import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from model import get_model
from config import Configuration
from dataset import Dataset
from train_loop import TrainLoop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as lr_decay
cfg = Configuration()

dataset = Dataset(cfg)
lr_schedule = lr_decay(
    boundaries=[i*dataset.num_train_batches for i in cfg.lr_boundaries],
    values=cfg.lr_values)

model = get_model(cfg.train_img_shape)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
tb_writer = tf.summary.create_file_writer(cfg.log_dir)
train_obj = TrainLoop(cfg, dataset, model, optimizer)

ckpt = tf.train.Checkpoint(
    model = train_obj.model,
    optimizer = train_obj.optimizer,
    train_dataset=train_obj.dataset.train_ds,
    epoch = tf.Variable(0, dtype=tf.dtypes.int64),
    max_psnr = tf.Variable(0.0))

chkpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_dir, max_to_keep=1, checkpoint_name='ckpt')
ckpt.restore(chkpt_manager.latest_checkpoint)
print(f"Initiating training from epoch {ckpt.epoch.numpy()}")
print("***Run tesnorboard to check training metrics***")

while(ckpt.epoch<cfg.n_epochs):

    ckpt.epoch.assign_add(1)
    train_obj.train_one_epoch(ckpt.epoch)

    save_prediction = ckpt.epoch%cfg.display_frequency==0
    display_batch = train_obj.run_validation(save_prediction)

    with tb_writer.as_default():
        tf.summary.scalar('train_loss', train_obj.train_loss.result(), step=ckpt.epoch)
        tf.summary.scalar('val_psnr', train_obj.val_psnr.result(), step=ckpt.epoch)
        if(save_prediction):
            tf.summary.image("val_images", display_batch, step=ckpt.epoch, max_outputs=cfg.display_samples) 

    if(ckpt.max_psnr<=train_obj.val_psnr.result()):
        ckpt.max_psnr.assign(train_obj.val_psnr.result())
        chkpt_manager.save(checkpoint_number=1)
        print("Checkpoint saved")