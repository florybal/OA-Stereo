import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # model settings
        self.parser.add_argument(
            "--pyramid_levels",
            type=int,
            default=4,
            help="number of levels in the correlation and flsvolume pyramid",
        )
        self.parser.add_argument(
            "--lookup_radius",
            type=int,
            default=4,
            help="width of the correlation pyramid",
        )
        self.parser.add_argument(
            "--feature_chan",
            type=int,
            default=128,
            help="channels of feature maps in updator",
        )
        self.parser.add_argument(
            "--gru_layers", type=int, default=1, help="number of GRU levels"
        )
        self.parser.add_argument(
            "--agg_disp",
            type=int,
            default=192,
            help="number of candidate disp for aggregation",
        )
        self.parser.add_argument(
            "--flsvolume_size_k", type=int, default=64, help="size of flsvolume"
        )
        self.parser.add_argument(
            "--grid_size_ijk",
            type=int,
            nargs="+",
            default=[128, 128, 128],
            help="size of sonar grid",
        )
        self.parser.add_argument(
            "--use_flsvolume", action="store_true", help="use flsvolume fusion"
        )
        self.parser.add_argument(
            "--cat_flsvolume",
            action="store_true",
            help="use flsvolume fusion with concatenate",
        )

        # DDP settings
        self.parser.add_argument(
            "--use_ddp", action="store_true", help="use DistributedDataParallel"
        )

        # train settings
        self.parser.add_argument(
            "--name", type=str, default="default", help="name your experiment"
        )
        self.parser.add_argument(
            "--use_flsloss",
            action="store_true",
            help="use flsloss in unsupervised training",
        )
        self.parser.add_argument(
            "--device", type=str, default="cuda", help="device to train/valid models"
        )
        self.parser.add_argument(
            "--train_sets",
            type=str,
            nargs="+",
            default=["sceneflow"],
            help="datasets to train on, including 'sceneflow', ",
        )
        self.parser.add_argument(
            "--image_size",
            type=int,
            nargs="+",
            default=[384, 512],
            help="image size used during training.",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=6, help="batch size used during training."
        )
        self.parser.add_argument(
            "--loader_thread",
            type=int,
            default=8,
            help="number of thread of dataloader.",
        )
        self.parser.add_argument(
            "--learning_rate", type=float, default=2e-4, help="max learning rate."
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-5,
            help="weight decay in optimizer.",
        )
        self.parser.add_argument(
            "--train_step", type=int, default=50000, help="length of training steps."
        )
        self.parser.add_argument(
            "--update_iters",
            type=int,
            default=3,
            help="number of updates to the disparity field in each forward pass.",
        )
        self.parser.add_argument(
            "--valid_period",
            type=int,
            default=1000,
            help="number of steps in the validation cycle.",
        )
        self.parser.add_argument(
            "--no_compile", action="store_true", help="do not use torch.compile"
        )
        self.parser.add_argument(
            "--loss_weights",
            type=float,
            nargs="+",
            default=[1.0, 0.1, 1e-3, 0.1],
            help="weights of loss terms.",
        )
        self.parser.add_argument(
            "--border",
            type=int,
            default=10,
            help="boundary excluded in the calculation of loss.",
        )
        self.parser.add_argument(
            "--loss_gamma",
            type=float,
            default=0.90,
            help="weight decay in squence loss.",
        )
        self.parser.add_argument(
            "--sigma",
            type=float,
            default=0.03,
            help="standard deviation of depth estimate.",
        )

        # valid settings
        self.parser.add_argument(
            "--valid_sets",
            type=str,
            nargs="+",
            default=["dave_sonar"],
            help="datasets to valid on",
        )
        self.parser.add_argument(
            "--valid_multipath_shift",
            type=float,
            default=15,
            help="range for random shifted images added to imgs",
        )
        self.parser.add_argument(
            "--valid_noise_level",
            type=float,
            default=0.0,
            help="standard deviation of Gaussian noise added to imgs during validation",
        )
        self.parser.add_argument(
            "--valid_contrast_coff",
            type=float,
            default=1,
            help="contrast factor for validation",
        )
        self.parser.add_argument(
            "--valid_blur_sigma",
            type=float,
            default=0,
            help="sigma of Gaussian blur for validation",
        )
        self.parser.add_argument(
            "--valid_occlusion",
            type=float,
            default=0,
            help="percentage of occlusion in the right image for valid",
        )
        self.parser.add_argument(
            "--valid_fx_distortion",
            type=float,
            default=0,
            help="percentage of fx distortion for valid",
        )
        self.parser.add_argument(
            "--valid_displacement",
            type=float,
            default=0,
            help="percentage of baseline displacement for valid",
        )
        self.parser.add_argument(
            "--valid_fls_movement",
            type=float,
            nargs="+",
            default=[0, 0, 0],
            help="percentage of fls position displacement for valid",
        )
        self.parser.add_argument(
            "--weight_stereo", type=str, default=None, help="restore checkpoint"
        )

        # show settings
        self.parser.add_argument(
            "-l", "--left_images", help="path to all first (left) frames"
        )
        self.parser.add_argument(
            "-r", "--right_images", help="path to all second (right) frames"
        )
        self.parser.add_argument(
            "-s", "--sonar_images", help="path to all sonar frames"
        )
        self.parser.add_argument(
            "--cam_left_yaml", help="path to param yaml of left camera"
        )
        self.parser.add_argument(
            "--cam_right_yaml", help="path to param yaml of right camera"
        )
        self.parser.add_argument("--fls_yaml", help="path to param yaml of fls")
        self.parser.add_argument(
            "--fls_compensation",
            type=int,
            default=1,
            help="distance compensation for fls image",
        )
        self.parser.add_argument(
            "--fls_normalization",
            type=int,
            default=1,
            help="distance normalization for fls image",
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
