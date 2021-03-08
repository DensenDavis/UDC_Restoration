import os
from shutil import copytree
from datetime import datetime

class Configuration():
    def __init__(self):
        # dataset config
        self.train_gt_path = 'ds/train/gt/*.png'
        self.train_input_path = 'ds/train/input/*.png'
        self.train_batch_size = 2
        self.train_img_shape = [128,128,3]
        self.train_augmentation = False
        self.val_gt_path = 'ds/trainval/gt/*.png'
        self.val_input_path = 'ds/trainval/input/*.png'
        self.val_batch_size = 2
        self.val_img_shape = [128,128,3]
        self.val_augmentation = False

        # training config
        self.ckpt_dir = 'train_ckpts/08032021-123638' # assign None if starting from scratch
        self.n_epochs = 5000
        self.lr_boundaries = [500,2000]
        self.lr_values= [1e-4, 1e-5, 1e-6]
        self.weight_mae_loss = 1

        #visualization config
        self.display_frequency = 20
        self.display_samples = 3
        self.log_dir = os.path.join('logs', str(datetime.now().strftime("%d%m%Y-%H%M%S")))

        # validate configuration
        self.__validate_config()
    
    def __validate_config(self):

        #validate checkpoint
        new_ckpt = os.path.join('train_ckpts',str(datetime.now().strftime("%d%m%Y-%H%M%S")))

        if(self.ckpt_dir != None):
            assert os.path.exists(self.ckpt_dir)
            copytree(self.ckpt_dir,new_ckpt)
        self.ckpt_dir = new_ckpt
        return