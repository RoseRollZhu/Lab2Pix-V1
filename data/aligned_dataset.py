import os.path
from data.base_dataset import *
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.load_width = opt.loadSize
        self.load_height = opt.loadSize // 2
        if 'resize' in opt.resize_or_crop:
            self.load_width = opt.loadSize
            self.load_height = opt.loadSize
        elif 'scale_width' in opt.resize_or_crop:
            self.load_width = opt.loadSize
            self.load_height = opt.loadSize // 2
        if 'crop' in opt.resize_or_crop:
            self.load_width = opt.fineSize
            self.load_height = opt.fineSize
        self.input_size = str(self.load_width) + 'x' + str(self.load_height)  # like '512x256'

        if 'cityscapes' in opt.dataroot:
            self.task = 'semantic2image'
            labelIDFile = opt.dataroot + '/' + opt.phase + 'Labels.txt'
            self.dir_ID = [opt.dataroot + '/' + line.replace("Train", "").replace("\n", "") for line in open(labelIDFile)]
            self.dir_RGB_ID = [opt.dataroot + '/' + line.replace("labelTrainIds", "color").replace("\n", "") for line in open(labelIDFile)]
            realImgFile = opt.dataroot + '/' + opt.phase + 'Images.txt'
            self.dir_Image = [opt.dataroot + '/' + line.replace("Train", "").replace("\n", "") for line in open(realImgFile)]
            instanceFile = opt.dataroot + '/' + opt.phase + 'Instances.txt'
            self.dir_Instance = [opt.dataroot + '/' + line.replace("Train", "").replace("\n", "") for line in open(instanceFile)]
            
            assert len(self.dir_ID) == len(self.dir_RGB_ID)
            assert len(self.dir_Image) == len(self.dir_Instance)
            assert len(self.dir_ID) == len(self.dir_Image)

            if len(self.dir_ID) > opt.max_dataset_size:
                self.dir_ID = self.dir_ID[:opt.max_dataset_size]
                self.dir_RGB_ID = self.dir_RGB_ID[:opt.max_dataset_size]
                self.dir_Image = self.dir_Image[:opt.max_dataset_size]
                self.dir_Instance = self.dir_Instance[:opt.max_dataset_size]

            self.dataset_size = len(self.dir_ID)

        elif 'edges2shoes' in opt.dataroot or 'edges2handbags' in opt.dataroot:
            self.task = 'sketch2image'
            self.dir = os.listdir(os.path.join(opt.dataroot, opt.phase))
            self.dir.sort(key=lambda x:int(x[:-7]))
            self.dir = [os.path.join(opt.dataroot, opt.phase, f) for f in self.dir]
            if len(self.dir) > opt.max_dataset_size:
                self.dir = self.dir[:opt.max_dataset_size]

            self.dataset_size = len(self.dir)

        else:
            print('Only support Dataset in \'cityscapes\', \'edges2shoes\', \'edges2handbags\'')
            exit()

    def __getitem__(self, index):
        img_index = random.randint(0, self.dataset_size - 1)
        if self.task == 'semantic2image':
            ID = Image.open(self.dir_ID[index])
            ID_RGB = Image.open(self.dir_RGB_ID[index]).convert('RGB')
            Img = Image.open(self.dir_Image[img_index]).convert('RGB')

            params = get_params(self.opt, ID.size)
            transform_img = get_transform(self.opt, params)
            transform_id = get_transform(self.opt, params, normalize=False)

            ID_tensor = transform_id(ID).squeeze(0) * 255
            ID_RGB_tensor = transform_img(ID_RGB)
            Img_tensor = transform_img(Img)

            path = self.dir_Image[index]

        elif self.task == 'sketch2image':
            flip = random.random() > 0.5 if self.opt.isTrain and not self.opt.no_flip else False
            ID_RGB = Image.open(self.dir[index]).convert('RGB').crop((0, 0, 256, 256))
            Img = Image.open(self.dir[img_index]).convert('RGB').crop((256, 0, 512, 256))
            if flip:
                ID_RGB = ID_RGB.transpose(Image.FLIP_LEFT_RIGHT)
                Img = Img.transpose(Image.FLIP_LEFT_RIGHT)
            transform = transforms.Compose([transforms.ToTensor(), normalize()])
            ID_tensor = transform(ID_RGB)[0] * 0.5 + 0.5
            ID_RGB_tensor = transform(ID_RGB)
            Img_tensor = transform(Img)

            path = self.dir[index]

        input_dict = {
            'ID': ID_tensor.long(),
            'ID_RGB': ID_RGB_tensor.float(),
            'Image': Img_tensor.float(),
            'path': path,
        }
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
