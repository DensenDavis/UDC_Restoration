import os
import tensorflow as tf
from model import get_model
from config import Configuration
from dataset import Dataset
from train_loop import TrainLoop
from utils import clone_checkpoint
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay as lr_decay
cfg = Configuration()

dataset = Dataset()
lr_schedule = lr_decay(
    boundaries=[i*dataset.num_train_batches for i in cfg.lr_boundaries],
    values=cfg.lr_values)

model = get_model([None, None, 3])
model.load_weights('best_model.hdf5')
optimizer = tf.keras.optimizers.Adam(lr_schedule)
tb_writer = tf.summary.create_file_writer(cfg.log_dir)
train_obj = TrainLoop(dataset, model, optimizer)

ckpt = tf.train.Checkpoint(
    model = train_obj.model,
    optimizer = train_obj.optimizer,
    train_dataset=train_obj.dataset.train_ds,
    epoch = tf.Variable(0, dtype=tf.dtypes.int64),
    max_psnr = tf.Variable(0.0))

ckpt_dir = clone_checkpoint(cfg.ckpt_dir)
chkpt_best = os.path.join(ckpt_dir,'best')
chkpt_best = tf.train.CheckpointManager(ckpt, chkpt_best, max_to_keep=1, checkpoint_name='ckpt')
chkpt_last = os.path.join(ckpt_dir,'last')
chkpt_last = tf.train.CheckpointManager(ckpt, chkpt_last, max_to_keep=1, checkpoint_name='ckpt')

if cfg.train_mode == 'best':
    ckpt.restore(chkpt_best.latest_checkpoint)
elif cfg.train_mode == 'last':
    ckpt.restore(chkpt_last.latest_checkpoint)
else:
    raise Exception('Error! invalid training mode, please check the config file.')

print(f"Initiating training from epoch {ckpt.epoch.numpy()}")
print("***Run tensorboard to check training metrics***")
print(f'best_psnr = {ckpt.max_psnr.numpy}')

while(ckpt.epoch < cfg.n_epochs):

    ckpt.epoch.assign_add(1)
    train_obj.train_one_epoch(ckpt.epoch)
    if (ckpt.epoch % cfg.val_freq) == 0:

        save_prediction = (ckpt.epoch % cfg.display_frequency) == 0
        display_batch = train_obj.run_validation(save_prediction)

        with tb_writer.as_default():
            tf.summary.scalar('train_loss', train_obj.train_loss.result(), step=ckpt.epoch)  # Average of stepwise loss of that epoch
            # tf.summary.scalar('train_psnr', train_obj.train_psnr.result(), step=ckpt.epoch)
            # tf.summary.scalar('val_loss', train_obj.val_loss.result(), step=ckpt.epoch)
            tf.summary.scalar('val_psnr', train_obj.val_psnr.result(), step=ckpt.epoch)
            if(save_prediction):
                tf.summary.image("val_images", display_batch, step=ckpt.epoch, max_outputs=cfg.display_samples) 

        if ckpt.max_psnr <= train_obj.val_psnr.result():
            ckpt.max_psnr.assign(train_obj.val_psnr.result())
            chkpt_best.save(checkpoint_number=1)
    chkpt_last.save(checkpoint_number=1)
    print(f'psnr : best/last = {ckpt.max_psnr.numpy()}/{train_obj.val_psnr.result().numpy()}')
