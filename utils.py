import os
from shutil import copytree
from config import Configuration
cfg = Configuration()


def clone_checkpoint(ckpt_dir):
    '''
    Create a copy of the existing checkpoint to avoid overwriting
    the current checkpoint. This will be handy if you cannot afford continuous
    training (as in Colab).
    '''
    new_ckpt = os.path.join('train_ckpts', os.path.split(cfg.log_dir)[-1])
    if ckpt_dir is not None:
        assert os.path.exists(ckpt_dir)
        copytree(ckpt_dir, new_ckpt)
    ckpt_dir = new_ckpt
    return ckpt_dir
