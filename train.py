import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import math
import numpy as np
import torch
from torch.autograd import Variable


torch.backends.cudnn.benchmark = True

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if os.path.exists(iter_path):
    opt.continue_train = True
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
need_steps = (opt.niter + opt.niter_decay) * dataset_size

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

if start_epoch > opt.niter:
    decay_time = start_epoch - opt.niter
    for i in range(decay_time):
        model.module.update_learning_rate()

loss_dict = {}
label_nc = 0
update_D = True
update_G_time = 0

print('Start Training...')
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    total = 0
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        losses, generated = model(
            label_RGB  =  data['ID_RGB'],
            label_ID   =  data['ID'],
            image      =  data['Image'],
            infer      =  save_fake
        )
        loss = {}

        for k, v in losses.items():
            if v is None:
                v = 0.0
            elif not isinstance(v, int):
                if isinstance(v, float):
                    print(k, v)
                    quit()
                v = torch.mean(v)
            loss[k] = v
        loss_G_GAN = loss['loss_G_GAN']
        loss_S = loss['loss_S']
        loss_G_feature = loss['loss_G_feature']
        loss_E = loss['loss_encode']
        loss_D_fake = loss['loss_D_fake']
        loss_D_real = loss['loss_D_real']

        loss_G = loss_G_GAN + loss_S * opt.lambda_seg + loss_G_feature + loss_E * opt.lambda_encode
        loss_D = (loss_D_fake + loss_D_real) * 0.5


        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()
        update_G_time += 1

        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        if update_D:
            model.module.optimizer_D.step()
            update_D = False

        if update_G_time >= opt.n_step_for_G:
            update_G_time = 0
            update_D = True

        loss_this = {}
        for num_D in range(opt.num_D):
            loss_name = 'loss_G_GAN_stack_' + str(opt.num_D - num_D)
            loss_this[loss_name] = loss[loss_name].item()
            loss_name = 'loss_D_fake_stack_' + str(opt.num_D - num_D)
            loss_this[loss_name] = loss[loss_name].item()
            loss_name = 'loss_D_real_stack_' + str(opt.num_D - num_D)
            loss_this[loss_name] = loss[loss_name].item()
            loss_name = 'loss_S_stack_' + str(opt.num_D - num_D)
            loss_this[loss_name] = loss[loss_name].item()
        loss_this['loss_G_Feature'] = loss_G_feature.item()

        if opt.use_encoder:
            loss_this['loss_encode'] = loss_E.item()
        for k, v in loss_this.items():
            if k in loss_dict:
                loss_dict[k] += v
            else:
                loss_dict[k] = v

        if total_steps % opt.print_freq == print_delta:
            errors = {k: (v / opt.print_freq) if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            loss_dict = {}

        if save_fake:
            visuals = OrderedDict()
            visuals['input_label'] = util.tensor2im(data['ID_RGB'][0])
            visuals['input_image'] = util.tensor2im(data['Image'][0])
            name = 'synthesized_image_'
            for id in range(len(generated)):
                name_this = name + str(id+1)
                generated_image = generated[id]
                visuals[name_this] = util.tensor2im(generated_image.data[0])
            visualizer.display_current_results(visuals, epoch, total_steps)

        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            start_save = time.time()
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            end_save = time.time()
            print('saving the model done, using time : %.3fs' % (end_save - start_save))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        start_save = time.time()
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        end_save = time.time()
        print('saving the model done, using time : %.3fs' % (end_save - start_save))

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
    if opt.linear_sharp:
        model.module.update_lambda_sharp()

