import torch


def create_model(opt):
    if opt.model == 'Lab2Pix-V1':
        from .Lab2PixV1_model import Lab2PixV1Model
        model = Lab2PixV1Model()
    else:
        exit()

    model.initialize(opt)
    if not opt.isTrain:
        model.eval()
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model.cuda()
