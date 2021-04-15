import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import time


class Lab2PixV1Model(BaseModel):
    def name(self):
        return 'Lab2PixV1Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.verbose = opt.verbose
        self.use_gpu = False
        self.background_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16, 23]
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.use_gpu = True

        if 'cityscapes' in opt.dataroot:
            if 'crop' in opt.resize_or_crop:
                self.input_width = opt.fineSize
                self.input_height = opt.fineSize
            else:
                self.input_width = opt.loadSize
                self.input_height = opt.loadSize // 2
            self.task = 'semantic2image'
        elif 'edges2shoes' in opt.dataroot or 'edges2handbags' in opt.dataroot:
            self.input_height = 256
            self.input_width = 256
            self.task = 'sketch2image'
        else:
            print('Only support Dataset in \'cityscapes\', \'edges2shoes\', \'edges2handbags\'')
            exit()

        print('Input size:', self.input_width, 'x', self.input_height)

        self.input_nc = opt.label_class
        self.label_class = opt.label_class if self.task == 'semantic2image' else 1
        self.output_nc = opt.output_nc
        self.noise_dim = opt.noise_dim
        self.num_D = opt.num_D

        self.netG = networks.define_G(self.label_class, self.output_nc, opt.noise_dim, self.task, opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.use_encoder:
            self.netE = networks.define_E(self.output_nc, self.noise_dim, opt.norm, gpu_ids=self.gpu_ids)
        if self.isTrain:
            for i in range(self.num_D):
                netS = networks.define_S(self.output_nc, self.label_class, gpu_ids=self.gpu_ids)
                setattr(self, 'netS_' + str(i + 1), netS)
            netD_input_nc = self.output_nc
            for i in range(self.num_D):
                n_layers = 3
                this_addtion_layer = 1
                addtion_layer_list = [3, 2, 1]
                this_addtion_layer = addtion_layer_list[i]
                netD = networks.define_D(netD_input_nc, n_layers, opt.ndf, opt.norm, this_addtion_layer, gpu_ids=self.gpu_ids)
                setattr(self, 'netD_'+str(i+1), netD)

            if not self.opt.use_resnet:
                self.feature = networks.Vgg19()
                self.criterionFeature = networks.VGGLoss(self.feature)
            else:
                self.feature = networks.Resnet101()
                self.criterionFeature = networks.ResnetLoss(self.feature)
        print('---------- Networks initialized -------------')

        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            if hasattr(opt, 'load_pretrain'):
                pretrained_path = opt.load_pretrain if opt.load_pretrain else os.path.join(opt.checkpoints_dir, opt.name)
            else:
                pretrained_path = os.path.join(opt.checkpoints_dir, opt.name)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.opt.use_encoder:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            if self.isTrain:
                for i in range(self.num_D):
                    netD = getattr(self, 'netD_' + str(i + 1))
                    netS = getattr(self, 'netS_' + str(i + 1))
                    self.load_network(netD, 'D' + str(i + 1), opt.which_epoch, pretrained_path)
                    self.load_network(netS, 'S' + str(i + 1), opt.which_epoch, pretrained_path)
            print('---------- Networks loaded -------------')

        if self.isTrain:
            self.old_lr_G = opt.lr_G
            self.old_lr_D = opt.lr_D

            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionSeg = networks.SegLoss(self.task)
            self.criterionEncode = networks.KLLoss()

            if opt.linear_sharp:
                self.lambda_sharp = 0.0
            else:
                self.lambda_sharp = opt.lambda_sharp

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            if self.opt.use_encoder:
                params += list(self.netE.parameters())
            for i in range(self.num_D):
                netS = getattr(self, 'netS_' + str(i + 1))
                params += list(netS.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr_G, betas=(opt.beta1, 0.999))

            # optimizer D
            params = []
            for i in range(self.num_D):
                netD = getattr(self, 'netD_' + str(i + 1))
                params += list(netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr_D, betas=(opt.beta1, 0.999))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        if self.opt.use_encoder:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        for i in range(self.num_D):
            netD = getattr(self, 'netD_' + str(i + 1))
            netS = getattr(self, 'netS_' + str(i + 1))
            self.save_network(netD, 'D' + str(i + 1), which_epoch, self.gpu_ids)
            self.save_network(netS, 'S' + str(i + 1), which_epoch, self.gpu_ids)

    def update_lambda_sharp(self):
        self.lambda_sharp += self.opt.lambda_sharp / (self.opt.niter + self.opt.niter_decay)

    def update_learning_rate(self):
        lrd_G = self.opt.lr_G / self.opt.niter_decay
        lr_G = self.old_lr_G - lrd_G
        lrd_D = self.opt.lr_D / self.opt.niter_decay
        lr_D = self.old_lr_D - lrd_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr_D
        if self.opt.verbose:
            # print('update learning rate: %f -> %f' % (self.old_lr, lr))
            print('update learning rate')
        self.old_lr_G = lr_G
        self.old_lr_D = lr_D

    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, label_RGB, label_ID, image, infer=False):
        real_image = image.data.cuda()
        real_label_RGB = label_RGB.data.cuda()  # b x 3 x h x w
        real_label_ID = label_ID.data.cuda()  # b x h x w

        if self.task == 'semantic2image':
            b, h, w = real_label_ID.size()
            background_map = torch.zeros_like(real_label_ID, dtype=torch.float, device=real_label_ID.device)
            for i in self.background_list:
                mask = real_label_ID == i
                background_map[mask] = 1.0
            forground_map = 1.0 - background_map
            foreground_num = forground_map.sum((1, 2)).float()
            background_num = h * w - foreground_num
            lam = (foreground_num + background_num) / (self.opt.EnhanceRate * foreground_num + background_num)
            enhance_mask = forground_map * lam * self.opt.EnhanceRate + background_map * lam
        else:
            enhance_mask = None

        loss_G_GAN = 0.0
        loss_D_fake = 0.0
        loss_D_real = 0.0
        loss_G_feature = 0.0
        loss_S = 0.0
        loss_E = 0.0

        b, h, w = real_label_ID.size()
        noise = torch.randn(b, self.noise_dim, device=real_label_ID.device)

        if self.opt.use_encoder:
            encode_noise = self.netE(real_image)
            loss_E = loss_E + self.criterionEncode(encode_noise, noise)
            input_noise = encode_noise
        else:
            input_noise = noise

        loss_dict = {}

        if self.task == 'semantic2image':
            G_input = real_label_ID
        elif self.task == 'sketch2image':
            G_input = real_label_RGB[:, 0:1, :, :]
        fake_image = self.netG(G_input, input_noise)

        if self.opt.use_encoder:
            rec_noise = self.netE(fake_image[0])
            loss_E = loss_E + self.criterionEncode(rec_noise, noise)
            loss_E = loss_E + self.criterionFeat(rec_noise, encode_noise)

        for i in range(self.num_D):
            real_image_this = real_image if i == 0 else F.avg_pool2d(real_image, kernel_size=2**i)

            netD = getattr(self, 'netD_' + str(i + 1))
            netS = getattr(self, 'netS_' + str(i + 1))

            pred_fake = netD(fake_image[i])
            pred_fake_pool = netD(fake_image[i].detach())

            loss_G_GAN_this = self.criterionGAN(pred_fake, True, False, enhance_mask)
            loss_G_GAN = loss_G_GAN + loss_G_GAN_this
            loss_name = 'loss_G_GAN_stack_' + str(self.num_D - i)
            loss_dict[loss_name] = loss_G_GAN_this

            loss_D_fake_this = self.criterionGAN(pred_fake_pool, False, True, enhance_mask)
            loss_D_fake = loss_D_fake + loss_D_fake_this
            resized_real_this = F.avg_pool2d(real_image_this, kernel_size=2)
            resized_real_this = F.interpolate(resized_real_this, scale_factor=2)
            pred_fake_resized = netD(resized_real_this)
            loss_D_fake_resized_this = self.criterionGAN(pred_fake_resized, False, True)
            loss_D_fake = loss_D_fake + loss_D_fake_resized_this * self.lambda_sharp
            loss_name = 'loss_D_fake_stack_' + str(self.num_D - i)
            loss_dict[loss_name] = loss_D_fake_this

            pred_real = netD(real_image_this)
            loss_D_real_this = self.criterionGAN(pred_real, True, True)
            loss_D_real = loss_D_real + loss_D_real_this * (1.0 + self.lambda_sharp)
            loss_name = 'loss_D_real_stack_' + str(self.num_D - i)
            loss_dict[loss_name] = loss_D_real_this

            rec_label = netS(fake_image[i])
            loss_S_this = self.criterionSeg(rec_label, G_input)
            loss_S = loss_S + loss_S_this
            loss_name = 'loss_S_stack_' + str(self.num_D - i)
            loss_dict[loss_name] = loss_S_this

            if not self.opt.no_feature_loss:
                loss_G_feature_this = self.criterionFeature(fake_image[i], real_image_this)
                loss_G_feature = loss_G_feature + loss_G_feature_this * self.opt.lambda_feat

            if i > 0:
                loss_G_consistence = self.criterionFeature(F.avg_pool2d(fake_image[i-1], kernel_size=2), fake_image[i], True)
                loss_G_feature = loss_G_feature + loss_G_consistence * self.opt.lambda_consistence

        loss_dict['loss_G_GAN'] = loss_G_GAN
        loss_dict['loss_D_fake'] = loss_D_fake
        loss_dict['loss_D_real'] = loss_D_real
        loss_dict['loss_S'] = loss_S
        loss_dict['loss_G_feature'] = loss_G_feature
        if self.opt.use_encoder:
            loss_dict['loss_encode'] = loss_E
        else:
            loss_dict['loss_encode'] = None

        generate_image = [fake_image[i] for i in range(self.num_D)]

        return loss_dict, generate_image

    @torch.no_grad()
    def inference(self, label_RGB, label_ID, image):
        real_image = image.data.cuda()
        real_label_RGB = label_RGB.data.cuda()  # b x 3 x h x w
        real_label_ID = label_ID.data.cuda()  # b x h x w

        b, c, h, w = real_label_RGB.size()
        # noise = torch.randn(b, self.noise_dim, device=real_label_ID.device)
        noise = 0.0
        ave_time = 10
        for i in range(ave_time):
            noise_this = torch.randn(b, self.noise_dim, device=real_label_ID.device)
            noise = noise + noise_this
        noise = noise / ave_time

        if self.opt.use_encoder:
            encode_noise = self.netE(real_image)
            input_noise = encode_noise
        else:
            input_noise = noise

        if self.task == 'semantic2image':
            G_input = real_label_ID
        elif self.task == 'sketch2image':
            G_input = real_label_RGB[:, 0:1, :, :]
        fake_image = self.netG(G_input, input_noise)

        return fake_image[0]
