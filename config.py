import os
from datetime import datetime


class Configuration():
    def __init__(self):

        # dataset config
        self.train_gt_path = 'ds/train/gt/*.png'
        self.train_input_path = 'ds/train/input/*.png'
        self.train_batch_size = 8
        self.train_img_shape = [256, 256, 3]
        self.train_augmentation = True
        self.val_gt_path = 'ds/val/gt/*.png'
        self.val_input_path = 'ds/val/input/*.png'
        self.val_batch_size = 1
        self.val_img_shape = [1024, 2048, 3]
        self.val_augmentation = False

        # training config
        self.ckpt_dir = None  # assign None if starting from scratch
        self.train_mode = ['best', 'last'][1]  # continue training from last epoch or epoch with best accuracy
        self.n_epochs = 10000
        self.lr_boundaries = [8500, 19000]
        self.lr_values = [1e-4, 9e-5, 1e-5]
        self.weight_mse_loss = 1

        #visualization config
        self.val_freq = 4             # frequency of validation - assign high value to accelerate training
        self.display_frequency = 50   # frequency of printing sample predictions - must be a multiple of val_freq
        self.display_samples = 5      # number of samples printed
        self.log_dir = os.path.join('logs', str(datetime.now().strftime("%d%m%Y-%H%M%S")))  # Tensorboard logging

        # Model config
        self.dilation_rates = (3, 2, 1, 1, 1, 1)
        self.nFilters_enc = 16
        self.nFilters_dec = 64
        self.nPyramidFilters_enc = 16
        self.nPyramidFilters_dec = 64
