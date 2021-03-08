import tensorflow as tf
import math
autotune = tf.data.experimental.AUTOTUNE


class Dataset():
    def __init__(self,cfg):
        self.cfg = cfg
        self.train_gt_files = sorted(tf.io.gfile.glob(self.cfg.train_gt_path))
        self.train_input_files = sorted(tf.io.gfile.glob(self.cfg.train_input_path))
        self.num_train_imgs = len(self.train_input_files)
        self.train_batch_size = self.cfg.train_batch_size
        self.num_train_batches = math.ceil(self.num_train_imgs/self.train_batch_size)
        self.train_aug = self.cfg.train_augmentation
        self.val_gt_files = sorted(tf.io.gfile.glob(self.cfg.val_gt_path))
        self.val_input_files = sorted(tf.io.gfile.glob(self.cfg.val_input_path))
        self.num_val_images = len(self.val_input_files)
        self.val_batch_size = self.cfg.val_batch_size
        self.val_aug = self.cfg.val_augmentation
        self.train_ds = self.get_train_data()
        self.val_ds = self.get_val_data()
        

    def read_file(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, dtype=tf.dtypes.float32)
        return image

    def create_pair(self, input_img_path, target_img_path):
        input_img = self.read_file(input_img_path)
        target_img = self.read_file(target_img_path)
        return tf.concat([input_img, target_img], axis=-1)

    def create_train_crop(self, img_pair):
        img_patch = tf.image.random_crop(img_pair, [self.cfg.train_img_shape[0], self.cfg.train_img_shape[1], self.cfg.train_img_shape[2]*2])
        return img_patch

    def create_val_crop(self, img_pair):
        img_patch = tf.image.random_crop(img_pair, [self.cfg.val_img_shape[0], self.cfg.val_img_shape[1], self.cfg.val_img_shape[2]*2])
        return img_patch

    def split_pair(self, image_pair):
        return image_pair[:,:,:self.cfg.train_img_shape[2]],image_pair[:,:,self.cfg.train_img_shape[2]:self.cfg.train_img_shape[2]*2]

    def train_augmentation(self, img_pair):
        img_pair = tf.image.random_flip_up_down(img_pair)
        img_pair = tf.image.random_flip_left_right(img_pair)
        img_pair = tf.image.rot90(img_pair, k=tf.random.uniform([], maxval=5, dtype=tf.int32))
        return img_pair

    def val_augmentation(self, ds):
        # ds = ds.map(lambda input_img, gt_img: tf.concat([input_img, gt_img], axis=-1), num_parallel_calls=autotune)
        ds.concatenate(ds.map(lambda img_pair: tf.image.flip_up_down(img_pair), num_parallel_calls=autotune))
        # self.num_val_images *= 2
        ds.concatenate(ds.map(lambda img_pair: tf.image.flip_left_right(img_pair), num_parallel_calls=autotune))
        # self.num_val_images *= 2
        # ds.concatenate(ds.map(lambda img_pair: tf.image.rot90(img_pair, k=1), num_parallel_calls=autotune))
        # self.num_val_images *= 2
        # ds = ds = ds.map(self.split_pair, num_parallel_calls=autotune)
        return ds

    def get_train_data(self):
        input_ds = tf.data.Dataset.from_tensor_slices(self.train_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.train_gt_files)
        ds = tf.data.Dataset.zip((input_ds, gt_ds))
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_train_crop, num_parallel_calls=autotune)
        if self.train_aug:
            ds = ds.map(self.train_augmentation, num_parallel_calls=autotune)
        ds = ds.map(self.split_pair, num_parallel_calls=autotune)
        ds = ds.batch(self.train_batch_size, drop_remainder=False)
        ds = ds.repeat().prefetch(buffer_size=autotune)
        return ds

    def get_val_data(self):
        input_ds = tf.data.Dataset.from_tensor_slices(self.val_input_files)
        gt_ds = tf.data.Dataset.from_tensor_slices(self.val_gt_files)
        ds = tf.data.Dataset.zip((input_ds, gt_ds))
        ds = ds.map(self.create_pair, num_parallel_calls=autotune)
        ds = ds.map(self.create_val_crop, num_parallel_calls=autotune)
        if self.val_aug:
            ds = self.val_augmentation(ds)
        ds = ds = ds.map(self.split_pair, num_parallel_calls=autotune)
        ds = ds.batch(self.val_batch_size, drop_remainder=False)
        ds = ds.prefetch(buffer_size=autotune)
        return ds
