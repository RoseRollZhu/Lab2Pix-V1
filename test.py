import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch


torch.backends.cudnn.benchmark = True

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

model = create_model(opt)
if opt.verbose:
    print(model)

for i, data in enumerate(dataset):
    if opt.max_test_num > 0:
        if i >= opt.max_test_num:
            break

    generated = model.inference(data['ID_RGB'], data['ID'], data['Image'])

    generated_image = util.tensor2im(generated.data[0])

    img_path = data['path'][0]
    base_name = os.path.basename(img_path)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    save_path = os.path.join(opt.results_dir, base_name)
    print('process image... %s' % img_path)
    util.save_image(generated_image, save_path)

