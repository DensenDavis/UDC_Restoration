import tensorflow as tf
import numpy as np
from tqdm import tqdm

class TrainLoop():
    def __init__(self, cfg, dataset, model, optimizer):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.mae_loss = tf.losses.MeanAbsoluteError()
        self.weight_mae_loss = self.cfg.weight_mae_loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_psnr = tf.keras.metrics.Mean(name='train_psnr')
        # self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_psnr = tf.keras.metrics.Mean(name='val_psnr')

    @tf.function
    def calculate_loss(self, pred, gt):
        return self.weight_mae_loss*self.mae_loss(pred, gt)

    @tf.function
    def train_step(self, input_batch, gt_batch):
        with tf.GradientTape(persistent=False) as tape:
            output_batch = self.model([input_batch], training=True)
            net_loss = self.calculate_loss(output_batch, gt_batch)
        gradients = tape.gradient(net_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.train_loss(net_loss)
        # self.train_psnr(tf.image.psnr(dec_outputs, self.gt_batch, max_val=1.0))
        return

    def train_one_epoch(self, epoch):
        self.train_loss.reset_states()
        # self.train_psnr.reset_states()
        pbar = tqdm(self.dataset.train_ds.take(self.dataset.num_train_batches), desc=f'Epoch : {epoch.numpy()}')
        for data_batch in pbar:
            input_batch, gt_batch = data_batch
            self.train_step(input_batch, gt_batch)
        return


    @tf.function
    def val_step(self, input_batch, gt_batch):
        output_batch = self.model([input_batch], training=False)
        self.val_psnr(tf.image.psnr(output_batch, gt_batch, max_val=1.0))
        return output_batch

    def generate_display_samples(self, display_batch, output_batch, gt_batch):
        padding_shape = (gt_batch.shape[0], gt_batch.shape[1], 20, gt_batch.shape[3])
        mini_display_batch = np.concatenate((output_batch,np.zeros(padding_shape),gt_batch), axis=2)
        if(type(display_batch)==type(None)):
            display_batch = mini_display_batch
        else:
            display_batch = np.concatenate((display_batch, mini_display_batch), axis=0)
        return display_batch

    def run_validation(self, save_prediction):
        # self.val_loss.reset_states()
        self.val_psnr.reset_states()
        display_batch = None
        for i,data_batch in enumerate(self.dataset.val_ds, start=1):
            input_batch, gt_batch = data_batch
            output_batch = self.val_step(input_batch, gt_batch)
            if(save_prediction):
                display_batch = self.generate_display_samples(display_batch, output_batch, gt_batch)
                if(display_batch.shape[0]>=self.cfg.display_samples):
                    save_prediction = False
        return display_batch
