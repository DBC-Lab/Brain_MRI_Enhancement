from BME_X.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--input_path', type=str, help="Path to the input test images (e.g., ./Testing_subjects/)")
        self.parser.add_argument('--output_path', type=str, help="Path to the enhanced images (e.g., ./Testing_subjects/output/)")
        self.parser.add_argument("--suffix", type=str, help="Suffix (e.g., T1w)")
        self.parser.add_argument('--bids_root', type=str, help='BIDS dataset directory')
        self.parser.add_argument('--subject_id', default='', type=str, help='subject_id')
        self.parser.add_argument('--session_id', default='', type=str, help='session_id')
        self.parser.add_argument('--age_in_month', default='6', type=str, help="Age group of the test images (e.g., fetal, 0, 3, 6, 9, 12, 18, 24, adult)")
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.1, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate for adam')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--optimizer', type=str, default='SGD', help='the optimizer for training')
        self.parser.add_argument('--shuffle', type=str, default='True', help='if shuffle the datatset')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for training')
        self.parser.add_argument('--scheduler', type=str, default='StepLR')
        self.parser.add_argument('--scheduler_step', type=int, default=1)
        self.parser.add_argument('--scheduler_gamma', type=int, default=0.1)
        self.parser.add_argument('--epochs', type=int, default=30)
        self.parser.add_argument('--run_name', type=str, default='segmentation')
        self.parser.add_argument('--model_save_dir', type=str, default='./checkpoints/')
        self.isTrain = True

