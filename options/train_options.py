from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for Generator adam')
        self.parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate for Discriminator adam')
        self.parser.add_argument('--n_step_for_G', type=int, default=1, help='# of steps to update generator parameters before update discriminator')

        # for discriminators
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--lambda_feat', type=float, default=1.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_encode', type=float, default=0.05, help='weight for encode matching loss')
        self.parser.add_argument('--lambda_sharp',  type=float, default=1.0, help='weight for sharpness enhancement loss')
        self.parser.add_argument('--linear_sharp', action='store_true', help='if specified, linear raise the lambda for sharpness enhancement loss')
        self.parser.add_argument('--lambda_seg', type=float, default=10.0, help='weight for segmentation matching loss')
        self.parser.add_argument('--lambda_consistence', type=float, default=0.000005, help='weight for segmentation matching loss')
        self.parser.add_argument('--no_feature_loss', action='store_true', help='if specified, do *not* use feature matching loss')
        self.parser.add_argument('--use_resnet', action='store_true', help='if specified, use Resnet feature matching loss')

        self.isTrain = True
