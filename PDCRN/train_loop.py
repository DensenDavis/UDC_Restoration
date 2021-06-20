import tensorflow as tf
import numpy as np
from tqdm import tqdm
from config import Configuration
cfg = Configuration()

class TrainLoop():
    def __init__(self, dataset, model, optimizer):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_psnr = tf.keras.metrics.Mean(name='train_psnr')
        # self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_psnr = tf.keras.metrics.Mean(name='val_psnr')

    @tf.function
    def calculate_loss(self, y_true, y_pred):
        mse_loss = cfg.weight_mse_loss*self.mse_loss(y_true, y_pred)
        return mse_loss

    @tf.function
    def train_step(self, input_batch, gt_batch):
        with tf.GradientTape(persistent=False) as tape:
            output_batch = self.model([input_batch], training=True)
            net_loss = self.calculate_loss(gt_batch, output_batch)
        gradients = tape.gradient(net_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.train_loss(net_loss)
        '''
        If needed calculate training accuracy (psnr, ssim etc.) as well.
        This may result in slower training due to extra computations.
        For psnr calculation, apply a relu with upper limit to clip values in range [0,1]
        '''
        # output_batch = tf.keras.activations.relu(output_batch, max_value=1)
        # self.train_psnr(tf.image.psnr(output_batch, self.gt_batch, max_val=1.0))
        return

    def train_one_epoch(self, epoch):
        self.train_loss.reset_states()
        # self.train_psnr.reset_states()
        pbar = tqdm(self.dataset.train_ds, desc=f'Epoch : {epoch.numpy()}')
        for data_batch in pbar:
            input_batch, gt_batch = data_batch
            self.train_step(input_batch, gt_batch)
        return

    @tf.function
    def val_step(self, input_batch, gt_batch):
        output_batch = self.model([input_batch], training=False)
        output_batch = tf.keras.activations.relu(output_batch, max_value=1)
        self.val_psnr(tf.image.psnr(output_batch, gt_batch, max_val=1.0))
        return output_batch

    def generate_display_samples(self, display_batch, input_batch, output_batch, gt_batch):
        padding_shape = (gt_batch.shape[0], gt_batch.shape[1], 20, gt_batch.shape[3])
        mini_display_batch = np.concatenate((input_batch, np.zeros(padding_shape), output_batch, np.zeros(padding_shape), gt_batch), axis=2)
        if(type(display_batch) == type(None)):
            display_batch = mini_display_batch
        else:
            display_batch = np.concatenate((display_batch, mini_display_batch), axis=0)
        return display_batch

    def run_validation(self, save_prediction):
        '''
        If needed calculate validation loss as well.
        This may result in slower training due to extra computations.
        '''
        # self.val_loss.reset_states()
        self.val_psnr.reset_states()
        display_batch = None
        for i, data_batch in enumerate(self.dataset.val_ds, start=1):
            input_batch, gt_batch = data_batch
            output_batch = self.val_step(input_batch, gt_batch)
            if(save_prediction):
                display_batch = self.generate_display_samples(display_batch, input_batch, output_batch, gt_batch)
                if(display_batch.shape[0] >= cfg.display_samples):
                    save_prediction = False
        return display_batch
