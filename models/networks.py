import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, noise_dim, task, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    activation = nn.ReLU(True)
    netG = LabelAdaptiveGenerator(input_nc, output_nc, noise_dim, task, norm_layer, activation)
    # print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_E(input_nc, noise_dim, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    activation = nn.LeakyReLU(0.2, True)
    netE = ImageEncoder(input_nc, noise_dim, norm_layer, activation)
    # print(netE)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netE.cuda(gpu_ids[0])
    netE.apply(weights_init)
    return netE


def define_D(input_nc, n_layers, ndf, norm='instance', n_addtion_layers=1, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    activation = nn.LeakyReLU(0.2, True)
    assert n_addtion_layers >= 1
    netD = HierarchicalPerceptualDiscriminator(input_nc, 1, n_layers, ndf, norm_layer, activation, n_addtion_layers)
    # print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_S(input_nc, output_nc, gpu_ids=[]):
    netS = ICNet(input_nc, output_nc)
    # print(netS)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netS.cuda(gpu_ids[0])
    netS.apply(weights_init)
    return netS


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.BCELoss(reduction='none')

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, discriminator, mask=None):
        loss = 0.0
        lambda_ = 1.0
        total_lambda = 0.0
        for input_i in input:
            pred = input_i
            target_tensor = self.get_target_tensor(pred, target_is_real)
            if mask is None:
                loss += self.loss(pred, target_tensor).mean() * lambda_
            else:
                b, c, h, w = pred.size()
                b, h_m, w_m = mask.size()
                kernel = (h_m // h, w_m // w)
                mask = F.avg_pool2d(mask.unsqueeze(1), kernel_size=kernel).squeeze(1)
                loss += (self.loss(pred, target_tensor) * mask).mean() * lambda_
            total_lambda += lambda_
            lambda_ /= 2.0
        loss = loss / total_lambda
        return loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.criterion = nn.KLDivLoss()

    def forward(self, x, y):
        x = F.log_softmax(x, 1)
        y = F.log_softmax(y, 1)
        loss = self.criterion(x, y)
        return loss


class SegLoss(nn.Module):
    def __init__(self, task):
        super(SegLoss, self).__init__()
        self.task = task
        if self.task == 'semantic2image':
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == 'sketch2image':
            self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x, y):
        data, data_8, data_16 = x
        label = y if y.dim() == 3 else y.squeeze(1)
        data = F.interpolate(data, size=label.size()[-2:])
        if self.task == 'semantic2image':
            loss = self.criterion(data, label)
        elif self.task == 'sketch2image':
            loss = self.criterion(torch.tanh(data), label)
        return loss


class VGGLoss(nn.Module):
    def __init__(self, vgg_net=None):
        super(VGGLoss, self).__init__()
        if vgg_net is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg_net
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0 / 16, 1.0 / 32]
        self.relate_weights = [1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0 / 16, 1.0]

    def forward(self, x, y, relate=False):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.
        for i in range(len(x_vgg)):
            x_f = x_vgg[i]
            y_f = y_vgg[i]
            if not relate:
                y_f = y_f.detach()
                loss += self.weights[i] * self.criterion(x_f, y_f)
            else:
                loss += self.criterion(x_f, y_f)
        return loss


class ResnetLoss(nn.Module):
    def __init__(self, resnet_net=None):
        super(ResnetLoss, self).__init__()
        if resnet_net is None:
            self.resnet = Resnet101().cuda()
        else:
            self.resnet = resnet_net
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0 / 16, 1.0 / 32]

    def forward(self, x, y, relate=False):
        x_resnet, y_resnet = self.resnet(x), self.resnet(y)
        loss = 0.
        for i in range(len(x_resnet)):
            x_f = x_resnet[i]
            y_f = y_resnet[i]
            if not relate:
                y_f = y_f.detach()
                loss += self.weights[i] * self.criterion(x_f, y_f)
            else:
                loss += self.criterion(x_f, y_f)
        return loss


##############################################################################
# Generator
##############################################################################
class LabelAdaptiveGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, noise_dim, task='semantic2image', norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(LabelAdaptiveGenerator, self).__init__()
        self.task = task
        if self.task == 'semantic2image':
            self.init_height = 2
            self.init_width = 4
        elif self.task == 'sketch2image':
            self.init_height = 2
            self.init_width = 2
        self.noise_process = nn.Linear(noise_dim, 1024*self.init_height*self.init_width)
        if self.task == 'semantic2image':
            self.SPADEResBlock_1 = SPADEResBlock(1024, 1024, input_dim, norm_layer, activation) # 2
            self.SPADEResBlock_2 = SPADEResBlock(1024, 1024, input_dim, norm_layer, activation) # 4
        elif self.task == 'sketch2image':
            self.encoder = Encoder(input_dim, 64, 1024, n_downsample=5, norm_layer=norm_layer, activation=activation)
        self.SPADEResBlock_3 = SPADEResBlock(1024, 1024, input_dim, norm_layer, activation) # 8
        self.SPADEResBlock_4 = SPADEResBlock(1024, 512, input_dim, norm_layer, activation)  # 16
        self.SPADEResBlock_5 = SPADEResBlock(512, 256, input_dim, norm_layer, activation)   # 32
        self.SPADEResBlock_6 = SPADEResBlock(256, 128, input_dim, norm_layer, activation)   # 64
        self.SPADEResBlock_7 = SPADEResBlock(128, 64, input_dim, norm_layer, activation)    # 128
        self.SPADEResBlock_5_fusion = LSCA(512, 256, input_dim, norm_layer, activation)
        self.SPADEResBlock_6_fusion = LSCA(256, 128, input_dim, norm_layer, activation)
        self.SPADEResBlock_7_fusion = LSCA(128, 64, input_dim, norm_layer, activation)
        self.out_7 = nn.Sequential(
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.out_6 = nn.Sequential(
            nn.Conv2d(128, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.out_5 = nn.Sequential(
            nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.num_up_layers = 7

    def forward(self, seg, noise):
        b = noise.size(0)
        input_ = self.noise_process(noise).view(b, 1024, self.init_height, self.init_width)
        if self.task == 'semantic2image':
            input_ = F.interpolate(self.SPADEResBlock_1(input_, seg), scale_factor=2)
            input_ = F.interpolate(self.SPADEResBlock_2(input_, seg), scale_factor=2)
        elif self.task == 'sketch2image':
            input_ = self.encoder(seg)
        input_ = F.interpolate(self.SPADEResBlock_3(input_, seg), scale_factor=2)
        input_ = F.interpolate(self.SPADEResBlock_4(input_, seg), scale_factor=2)
        input_5 = F.interpolate(self.SPADEResBlock_5(input_, seg), scale_factor=2)
        input_5, mask_5 = self.SPADEResBlock_5_fusion(input_, input_5, seg)
        out_5 = self.out_5(input_5)
        input_6 = F.interpolate(self.SPADEResBlock_6(input_5, seg), scale_factor=2)
        input_6, mask_6 = self.SPADEResBlock_6_fusion(input_5, input_6, seg)
        out_6 = self.out_6(input_6)
        input_7 = F.interpolate(self.SPADEResBlock_7(input_6, seg), scale_factor=2)
        input_7, mask_7 = self.SPADEResBlock_7_fusion(input_6, input_7, seg)
        out_7 = self.out_7(input_7)
        return out_7, out_6, out_5


class SPADE(nn.Module):
    def __init__(self, dim, seg_dim, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(SPADE, self).__init__()
        self.seg_dim = seg_dim
        common_conv_in = seg_dim
        self.normalization = norm_layer(dim)
        self.common_conv = [
            nn.Conv2d(common_conv_in, 128, kernel_size=3, stride=1, padding=1),
            activation,
        ]
        self.common_conv = nn.Sequential(*self.common_conv)
        self.mul_conv = nn.Conv2d(128, dim, kernel_size=3, stride=1, padding=1)
        self.add_conv = nn.Conv2d(128, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, seg):
        b, c, h, w = x.size()
        if len(seg.size()) == 3:
            b_, h_, w_ = seg.size()
            scale_h, scale_w = h_ // h, w_ // w
            seg = F.max_pool2d(seg.float().unsqueeze(1), kernel_size=(scale_h, scale_w)).long()
            seg_fea = torch.zeros(b, self.seg_dim, h, w, device=x.device).scatter_(1, seg, 1.0)
        else:
            b_, c_, h_, w_ = seg.size()
            scale_h, scale_w = h_ // h, w_ // w
            seg_fea = F.avg_pool2d(seg.float(), kernel_size=(scale_h, scale_w))
        seg_fea = self.common_conv(seg_fea)
        mul_fea = self.mul_conv(seg_fea)
        add_fea = self.add_conv(seg_fea)
        x = self.normalization(x)
        out = x * mul_fea + add_fea
        return out


class SPADEResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, seg_dim, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(SPADEResBlock, self).__init__()
        self.SPADE_1 = SPADE(input_dim, seg_dim, norm_layer, activation)
        model = [
            activation,
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
        ]
        self.SPADE_1_ = nn.Sequential(*model)
        self.SPADE_2 = SPADE(output_dim, seg_dim, norm_layer, activation)
        model = [
            activation,
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
        ]
        self.SPADE_2_ = nn.Sequential(*model)
        self.res = False
        if input_dim != output_dim:
            self.res = True
            self.SPADE_res = SPADE(input_dim, seg_dim, norm_layer, activation)
            model = [
                activation,
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            ]
            self.SPADE_res_ = nn.Sequential(*model)

    def forward(self, x, seg):
        out = self.SPADE_1(x, seg)
        out = self.SPADE_1_(out)
        out = self.SPADE_2(out, seg)
        out = self.SPADE_2_(out)
        if self.res:
            res = self.SPADE_res(x, seg)
            res = self.SPADE_res_(res)
            out = out + res
        return out


class LSCA(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, seg_dim, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(LSCA, self).__init__()
        self.seg_dim = seg_dim
        self.input1 = nn.Sequential(
            nn.Conv2d(input_dim_1, input_dim_2, kernel_size=3, stride=1, padding=1),
            activation,
        )
        self.norm1 = norm_layer(input_dim_2)
        self.norm2 = norm_layer(input_dim_2)
        self.seg_process = nn.Sequential(
            nn.Conv2d(seg_dim, input_dim_2, kernel_size=3, stride=1, padding=1),
            activation,
            norm_layer(input_dim_2)
        )
        self.mask = nn.Sequential(
            nn.Conv2d(input_dim_2 * 3, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2, seg):
        b, c, h, w = x_2.size()
        if len(seg.size()) == 3:
            b_, h_, w_ = seg.size()
            scale_h, scale_w = h_ // h, w_ // w
            seg = F.max_pool2d(seg.float().unsqueeze(1), kernel_size=(scale_h, scale_w)).long()
            seg_fea = torch.zeros(b, self.seg_dim, h, w, device=x.device).scatter_(1, seg, 1.0)
        else:
            b_, c_, h_, w_ = seg.size()
            scale_h, scale_w = h_ // h, w_ // w
            seg_fea = F.avg_pool2d(seg.float(), kernel_size=(scale_h, scale_w))
        x_1 = self.input1(F.interpolate(x_1, scale_factor=2))
        x_1_norm = self.norm1(x_1)
        x_2_norm = self.norm2(x_2)
        seg_fea = self.seg_process(seg_fea)
        mask = torch.cat((x_1_norm, x_2_norm, seg_fea), 1)
        mask = self.mask(mask)
        out = mask * x_1 + (1.0 - mask) * x_2
        return out, mask


class Encoder(nn.Module):
    def __init__(self, in_dim, ngf, max_dim, n_downsample, norm_layer, activation=nn.ReLU(True)):
        super(Encoder, self).__init__()
        model = []
        model += [
            nn.Conv2d(in_dim, ngf, kernel_size=5, stride=1, padding=2),
            norm_layer(ngf),
            activation,
        ]
        in_ = ngf
        for i in range(n_downsample):
            out_ = min(in_ * 2, max_dim)
            model += [
                nn.Conv2d(in_, out_, kernel_size=3, stride=2, padding=1),
                norm_layer(out_),
                activation,
            ]
            in_ = out_
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


##############################################################################
# Discriminator
##############################################################################
class HierarchicalPerceptualDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True), n_addition_layer=3):
        super(HierarchicalPerceptualDiscriminator, self).__init__()
        assert n_addition_layer > 0
        max_dim = 512
        self.n_layers = 3
        self.n_add_layer = n_addition_layer
        self.backboneNet = Vgg16backbone(block_num=min(5, 2+n_addition_layer), requires_grad=False)

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
            norm_layer(ndf),
            activation,
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ndf * 2),
            activation,
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ndf * 4),
            activation,
        ]
        model = nn.Sequential(*sequence)
        self.common = model

        for i in range(n_addition_layer):
            nf = max_dim if i > 0 else max_dim // 2
            sequence = [
                norm_layer(nf),
                activation,
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'pre' + str(i + 1), model)

            nf = 512 if i == 0 else 1024
            sequence = [
                nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1),
                norm_layer(512),
                activation,
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'gcb' + str(i + 1), model)

        for i in range(n_addition_layer):
            if i != n_addition_layer - 1:
                nf = 512
                sequence = [
                    InceptionBlock(nf, 512, stride=2, norm_layer=norm_layer, activation=activation),
                ]
                model = nn.Sequential(*sequence)
                setattr(self, 'conv' + str(i + 1) + '_line', model)

            nf = 512
            sequence = [
                norm_layer(nf),
                activation,
                nn.Conv2d(nf, output_nc, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid(),
            ]
            model = nn.Sequential(*sequence)
            setattr(self, 'conv' + str(i + 1) + '_out', model)

    def forward(self, input):
        out = []
        feat_pool = self.common(input)
        backbone_fea = self.backboneNet(input)
        for i in range(self.n_add_layer):
            backbone_fea_this = backbone_fea[self.n_layers-1+i]
            pre = getattr(self, 'pre' + str(i + 1))
            gcb = getattr(self, 'gcb' + str(i + 1))
            backbone_fea_this = pre(backbone_fea_this)
            fea = torch.cat((feat_pool, backbone_fea_this), 1)
            fea = gcb(fea)
            outBlock = getattr(self, 'conv' + str(i + 1) + '_out')
            out_this = outBlock(fea)
            out.append(out_this)
            if i < self.n_add_layer - 1:
                lineBlock = getattr(self, 'conv' + str(i + 1) + '_line')
                feat_pool = lineBlock(fea)
        return out


class Resnet_3_3_Block(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(Resnet_3_3_Block, self).__init__()
        model = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            norm_layer(dim),
            activation,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out


class Resnet_1_3_1_Block(nn.Module):
    def __init__(self, dim, norm_layer, hidden_dim=None, activation=nn.ReLU(True), use_dropout=False):
        super(Resnet_1_3_1_Block, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        model = [
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            norm_layer(hidden_dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            norm_layer(hidden_dim),
            activation,
        ]
        if use_dropout:
            model += [
                nn.Dropout(0.5),
            ]
        model += [
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            norm_layer(dim),
            activation,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm_layer, activation):
        super(InceptionBlock, self).__init__()
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            activation,
        )
        self.conv3_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            norm_layer(out_channels),
            activation,
        )
        self.conv5_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            norm_layer(out_channels),
            activation,
        )

    def forward(self, x):
        pool_b = self.pool_branch(x)
        conv3_b = self.conv3_branch(x)
        conv5_b = self.conv5_branch(x)
        out = pool_b + conv3_b + conv5_b
        return out


##############################################################################
# Other Networks
##############################################################################
class ImageEncoder(nn.Module):
    def __init__(self, input_dim, noise_dim, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            norm_layer(64),
            activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activation,
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation,
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation,
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation,
        )
        self.fc = nn.Linear(512*8*4, noise_dim)

    def forward(self, x):
        b = x.size(0)
        x = self.conv(x).view(b, -1)
        out = self.fc(x)
        return out


class ICNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True)):
        super(ICNet, self).__init__()
        self.upsample = torch.nn.Upsample(size=(256, 256), mode='nearest')
        # size/2
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0, count_include_pad=False)

        self.net_model = []
        # net_model1
        # size/4
        self.net_model1_1 = [nn.Conv2d(input_nc, 32, stride=2, kernel_size=3, padding=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 32, kernel_size=3, padding=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 64, kernel_size=3, padding=1),
                             norm_layer(64),
                             activation,
                             nn.MaxPool2d(3, stride=2, padding=1)]
        self.net_model1_1 = nn.Sequential(*self.net_model1_1)
        self.net_model1_2_1 = [nn.Conv2d(64, 32, kernel_size=1),
                               norm_layer(32),
                               activation,
                               nn.Conv2d(32, 32, kernel_size=3, padding=1),
                               norm_layer(32),
                               activation,
                               nn.Conv2d(32, 128, kernel_size=1),
                               norm_layer(128)]
        self.net_model1_2_1 = nn.Sequential(*self.net_model1_2_1)
        self.net_model1_2_2 = [nn.Conv2d(64, 128, kernel_size=1),
                               norm_layer(128)]
        self.net_model1_2_2 = nn.Sequential(*self.net_model1_2_2)
        self.net_model1_3 = activation
        # net_model2
        self.net_model2_1 = [nn.Conv2d(128, 32, kernel_size=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 32, kernel_size=3, padding=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 128, kernel_size=1),
                             norm_layer(128)]
        self.net_model2_1 = nn.Sequential(*self.net_model2_1)
        self.net_model2_2 = activation
        self.net_model2_3 = [nn.Conv2d(128, 32, kernel_size=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 32, kernel_size=3, padding=1),
                             norm_layer(32),
                             activation,
                             nn.Conv2d(32, 128, kernel_size=1),
                             norm_layer(128)]
        self.net_model2_3 = nn.Sequential(*self.net_model2_3)
        self.net_model2_4 = activation
        # net_model3
        # size/2
        self.net_model3_1 = [nn.Conv2d(128, 64, stride=2, kernel_size=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 256, kernel_size=1),
                             norm_layer(256)]
        self.net_model3_1 = nn.Sequential(*self.net_model3_1)
        self.net_model3_2 = [nn.Conv2d(128, 256, stride=2, kernel_size=1),
                             norm_layer(256)]
        self.net_model3_2 = nn.Sequential(*self.net_model3_2)
        self.net_model3_3 = activation
        self.net_model3_4 = [nn.Conv2d(256, 64, kernel_size=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 256, kernel_size=1),
                             norm_layer(256)]
        self.net_model3_4 = nn.Sequential(*self.net_model3_4)
        self.net_model3_5 = activation
        self.net_model3_6 = [nn.Conv2d(256, 64, kernel_size=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 256, kernel_size=1),
                             norm_layer(256)]
        self.net_model3_6 = nn.Sequential(*self.net_model3_6)
        self.net_model3_7 = activation
        self.net_model3_8 = [nn.Conv2d(256, 64, kernel_size=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                             norm_layer(64),
                             activation,
                             nn.Conv2d(64, 256, kernel_size=1),
                             norm_layer(256)]
        self.net_model3_8 = nn.Sequential(*self.net_model3_8)
        self.net_model3_9 = activation
        # net_model4
        self.net_model4_1 = [nn.Conv2d(256, 128, kernel_size=1),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 512, kernel_size=1),
                             norm_layer(512)]
        self.net_model4_1 = nn.Sequential(*self.net_model4_1)
        self.net_model4_2 = [nn.Conv2d(256, 512, kernel_size=1),
                             norm_layer(512)]
        self.net_model4_2 = nn.Sequential(*self.net_model4_2)
        self.net_model4_3 = activation
        self.net_model4_4 = [nn.Conv2d(512, 128, kernel_size=1),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 512, kernel_size=1),
                             norm_layer(512)]
        self.net_model4_4 = nn.Sequential(*self.net_model4_4)
        self.net_model4_5 = activation
        self.net_model4_6 = [nn.Conv2d(512, 128, kernel_size=1),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 512, kernel_size=1),
                             norm_layer(512)]
        self.net_model4_6 = nn.Sequential(*self.net_model4_6)
        self.net_model4_7 = activation
        self.net_model4_8 = [nn.Conv2d(512, 128, kernel_size=1),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                             norm_layer(128),
                             activation,
                             nn.Conv2d(128, 512, kernel_size=1),
                             norm_layer(512)]
        self.net_model4_8 = nn.Sequential(*self.net_model4_8)
        self.net_model4_9 = activation
        self.net_model4_10 = [nn.Conv2d(512, 128, kernel_size=1),
                              norm_layer(128),
                              activation,
                              nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                              norm_layer(128),
                              activation,
                              nn.Conv2d(128, 512, kernel_size=1),
                              norm_layer(512)]
        self.net_model4_10 = nn.Sequential(*self.net_model4_10)
        self.net_model4_11 = activation
        self.net_model4_12 = [nn.Conv2d(512, 128, kernel_size=1),
                              norm_layer(128),
                              activation,
                              nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                              norm_layer(128),
                              activation,
                              nn.Conv2d(128, 512, kernel_size=1),
                              norm_layer(512)]
        self.net_model4_12 = nn.Sequential(*self.net_model4_12)
        self.net_model4_13 = activation
        # net_model5
        self.net_model5_1 = [nn.Conv2d(512, 256, kernel_size=1),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 1024, kernel_size=1),
                             norm_layer(1024)]
        self.net_model5_1 = nn.Sequential(*self.net_model5_1)
        self.net_model5_2 = [nn.Conv2d(512, 1024, kernel_size=1),
                             norm_layer(1024)]
        self.net_model5_2 = nn.Sequential(*self.net_model5_2)
        self.net_model5_3 = activation
        self.net_model5_4 = [nn.Conv2d(1024, 256, kernel_size=1),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 1024, kernel_size=1),
                             norm_layer(1024)]
        self.net_model5_4 = nn.Sequential(*self.net_model5_4)
        self.net_model5_5 = activation
        self.net_model5_6 = [nn.Conv2d(1024, 256, kernel_size=1),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
                             norm_layer(256),
                             activation,
                             nn.Conv2d(256, 1024, kernel_size=1),
                             norm_layer(1024)]
        self.net_model5_6 = nn.Sequential(*self.net_model5_6)
        self.net_model5_7 = activation
        # size = 128 x 64
        self.net_model5_8_1 = [nn.AvgPool2d(kernel_size=(8, 8), stride=(8, 8), padding=0),
                               # size = 2 x 2
                               nn.Upsample(size=(8, 8), mode='nearest')
                               # size = 32 x 32
                               ]
        self.net_model5_8_1 = nn.Sequential(*self.net_model5_8_1)
        self.net_model5_8_2 = [nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0),
                               # size = 4 x 4
                               nn.Upsample(size=(8, 8), mode='nearest')
                               # size = 32 x 32
                               ]
        self.net_model5_8_2 = nn.Sequential(*self.net_model5_8_2)
        self.net_model5_8_3 = [nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0, count_include_pad=False),
                               # size = 6 x 6
                               nn.Upsample(size=(8, 8), mode='nearest')
                               # size = 32 x 32
                               ]
        self.net_model5_8_3 = nn.Sequential(*self.net_model5_8_3)
        self.net_model5_8_4 = [nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, count_include_pad=False),
                               # size = 12 x 12
                               nn.Upsample(size=(8, 8), mode='nearest')
                               # size = 32 x 32
                               ]
        self.net_model5_8_4 = nn.Sequential(*self.net_model5_8_4)

        '''
        self.net_model5_9 = [nn.Conv2d(1024, 256, kernel_size=1),
                             norm_layer(256),
                             activation,
                             nn.Upsample(scale_factor=2, mode='nearest'),
                             #size = 64 x 128
                             nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
                             norm_layer(128)]
        self.net_model5_9 = nn.Sequential(*self.net_model5_9)
        '''
        self.net_model5_9_1 = [nn.Conv2d(1024, 256, kernel_size=1),
                               norm_layer(256),
                               activation
                               ]
        self.net_model5_9_1 = nn.Sequential(*self.net_model5_9_1)
        self.loss_model_16 = [nn.Upsample(scale_factor=2, mode='nearest'),
                              # size = 64 x 128
                              nn.Conv2d(256, output_nc, kernel_size=1)
                              ]
        self.loss_model_16 = nn.Sequential(*self.loss_model_16)
        self.net_model5_9_2 = [nn.Upsample(scale_factor=2, mode='nearest'),
                               # size = 64 x 128
                               nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
                               norm_layer(128)]
        self.net_model5_9_2 = nn.Sequential(*self.net_model5_9_2)

        # net_model_sub2
        self.net_model_sub2_1 = [nn.Conv2d(256, 128, kernel_size=1),
                                 norm_layer(128)]
        self.net_model_sub2_1 = nn.Sequential(*self.net_model_sub2_1)
        self.net_model_sub2_2 = activation
        self.loss_model_8 = [nn.Upsample(scale_factor=2, mode='nearest'),
                             # 128 x 256
                             nn.Conv2d(128, output_nc, kernel_size=1)
                             ]
        self.loss_model_8 = nn.Sequential(*self.loss_model_8)
        self.net_model_sub2_3 = [nn.Upsample(scale_factor=2, mode='nearest'),
                                 # 128 x 256
                                 nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
                                 norm_layer(128)]
        self.net_model_sub2_3 = nn.Sequential(*self.net_model_sub2_3)
        # net_model_sub1
        self.net_model_sub1_1 = [nn.Conv2d(input_nc, 32, stride=2, kernel_size=3, padding=1),
                                 # 512 x 1024
                                 norm_layer(32),
                                 activation,
                                 nn.Conv2d(32, 32, stride=2, kernel_size=3, padding=1),
                                 # 256 x 512
                                 norm_layer(32),
                                 activation,
                                 nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
                                 # 128 x 256
                                 norm_layer(64),
                                 activation,
                                 nn.Conv2d(64, 128, kernel_size=1),
                                 norm_layer(128)]
        self.net_model_sub1_1 = nn.Sequential(*self.net_model_sub1_1)
        self.net_model_sub1_2 = activation
        self.net_model_sub1_3 = [nn.Upsample(scale_factor=4, mode='nearest'),
                                 # 256 x 512
                                 nn.Conv2d(128, output_nc, kernel_size=5, stride=1, padding=2),
                                 nn.Upsample(scale_factor=2, mode='nearest')
                                 # 1024 x 2048
                                 ]
        self.net_model_sub1_3 = nn.Sequential(*self.net_model_sub1_3)

    def forward(self, x):
        # input_sub0 = x
        # input_sub1 = self.downsample(input_sub0)
        x = self.upsample(x)
        data = self.downsample(x)  # input_sub2
        data = self.net_model1_1(data)  # pool1_3x3_s2
        data1 = self.net_model1_2_1(data)  # conv2_1_1x1_increase
        data = self.net_model1_2_2(data)  # conv2_1_1x1_proj
        data = self.net_model1_3(data + data1)  # conv2_1
        data1 = self.net_model2_1(data)  # conv2_2_1x1_increase
        data = self.net_model2_2(data + data1)  # conv2_2
        data1 = self.net_model2_3(data)  # conv2_3_1x1_increase
        data = self.net_model2_4(data + data1)  # conv2_3
        data1 = self.net_model3_1(data)  # conv3_1_1x1_increase
        data = self.net_model3_2(data)  # conv3_1_1x1_proj
        data = self.net_model3_3(data + data1)  # conv3_1
        conv3_1 = data
        data = self.downsample(data)  # conv3_1_sub4
        data1 = self.net_model3_4(data)  # conv3_2_1x1_increase
        data = self.net_model3_5(data + data1)  # conv3_2
        data1 = self.net_model3_6(data)  # conv3_3_1x1_increase
        data = self.net_model3_7(data + data1)  # conv3_3
        data1 = self.net_model3_8(data)  # conv3_4_1x1_increase
        data = self.net_model3_9(data + data1)  # conv3_4
        data1 = self.net_model4_1(data)  # conv4_1_1x1_increase
        data = self.net_model4_2(data)  # conv4_1_1x1_proj
        data = self.net_model4_3(data + data1)  # conv4_1
        data1 = self.net_model4_4(data)  # conv4_2_1x1_increase
        data = self.net_model4_5(data + data1)  # conv4_2
        data1 = self.net_model4_6(data)  # conv4_3_1x1_increase
        data = self.net_model4_7(data + data1)  # conv4_3
        data1 = self.net_model4_8(data)  # conv4_4_1x1_increase
        data = self.net_model4_9(data1)  # conv4_4
        data1 = self.net_model4_10(data)  # conv4_5_1x1_increase
        data = self.net_model4_11(data + data1)  # conv4_5
        data1 = self.net_model4_12(data)  # conv4_6_1x1_increase
        data = self.net_model4_13(data + data1)  # conv4_6
        data1 = self.net_model5_1(data)  # conv5_1_1x1_increase
        data = self.net_model5_2(data)  # conv5_1_1x1_proj
        data = self.net_model5_3(data + data1)  # conv5_1
        data1 = self.net_model5_4(data)  # conv5_2_1x1_increase
        data = self.net_model5_5(data + data1)  # conv5_2
        data1 = self.net_model5_6(data)  # conv5_3_1x1_increase
        data = self.net_model5_7(data + data1)  # conv5_3

        conv5_3_pool1_interp = self.net_model5_8_1(data)
        conv5_3_pool2_interp = self.net_model5_8_2(data)
        conv5_3_pool3_interp = self.net_model5_8_3(data)
        conv5_3_pool6_interp = self.net_model5_8_4(data)
        # conv5_3_sum
        data = conv5_3_pool1_interp + conv5_3_pool2_interp + conv5_3_pool3_interp + conv5_3_pool6_interp
        # data = self.net_model5_9(data) # conv_sub4
        data = self.net_model5_9_1(data)  # conv_sub4
        data_16 = self.loss_model_16(data)  # size / 16 for loss
        data = self.net_model5_9_2(data)  # conv_sub4

        data1 = self.net_model_sub2_1(conv3_1)  # conv3_1_sub2_proj
        del conv3_1
        data = self.net_model_sub2_2(data + data1)  # sub24_sum
        data_8 = self.loss_model_8(data)  # size / 8 for loss
        data1 = self.net_model_sub2_3(data)  # conv_sub2

        data = self.net_model_sub1_1(x)  # conv3_sub1_proj
        data = self.net_model_sub1_2(data + data1)  # sub12_sum
        data = self.net_model_sub1_3(data)  # conv6_interp
        return data, data_8, data_16


class Vgg16backbone(torch.nn.Module):
    def __init__(self, block_num, requires_grad=False):
        super(Vgg16backbone, self).__init__()
        assert block_num <= 5
        self.block_num = block_num
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        if block_num >= 1:
            self.slice1 = torch.nn.Sequential()
            for x in range(5):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 2:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 3:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 4:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if block_num >= 5:
            self.slice5 = torch.nn.Sequential()
            for x in range(24, 31):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = []
        pool = x
        for i in range(self.block_num):
            model = getattr(self, 'slice' + str(i + 1))
            pool = model(pool)
            out.append(pool)
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # for x in range(4):
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 18):
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(18, 27):
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(27, 36):
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Resnet101(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet101, self).__init__()
        resnet_pretrained = models.resnet101(pretrained=True)
        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice0.add_module('conv1', resnet_pretrained.conv1)
        self.slice0.add_module('bn1', resnet_pretrained.bn1)
        self.slice0.add_module('relu', resnet_pretrained.relu)
        self.slice0.add_module('maxpool', resnet_pretrained.maxpool)
        self.slice1.add_module('layer1', resnet_pretrained.layer1)
        self.slice2.add_module('layer2', resnet_pretrained.layer2)
        self.slice3.add_module('layer3', resnet_pretrained.layer3)
        self.slice4.add_module('layer4', resnet_pretrained.layer4)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu0 = self.slice0(X)
        h_relu1 = self.slice1(h_relu0)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        out = [h_relu0, h_relu1, h_relu2, h_relu3, h_relu4]
        return out
