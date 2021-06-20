import numpy as np
import tensorflow as tf


def Depth2Space(block_size):
    return lambda x: tf.nn.depth_to_space(x, block_size)


def Space2Depth(block_size):
    return lambda x: tf.nn.space_to_depth(x, block_size)


class DWT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DWT, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x01 = inputs[:, 0::2, :, :] / 4.0
        x02 = inputs[:, 1::2, :, :] / 4.0
        x1 = x01[:, :, 0::2, :]
        x2 = x01[:, :, 1::2, :]
        x3 = x02[:, :, 0::2, :]
        x4 = x02[:, :, 1::2, :]
        y1 = x1+x2+x3+x4
        y2 = x1-x2+x3-x4
        y3 = x1+x2-x3-x4
        y4 = x1-x2-x3+x4
        y = tf.keras.backend.concatenate([y1, y2, y3, y4], axis=-1)
        return y

    def compute_output_shape(self, input_shape):
        c = input_shape[-1]*4
        if(input_shape[1] is not None and input_shape[2] is not None):
            return (input_shape[0], input_shape[1] >> 1, input_shape[2] >> 1, c)
        else:
            return (None, None, None, c)


class IWT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IWT, self).__init__(**kwargs)

    def build(self, input_shape):
        c = input_shape[-1]
        out_c = c >> 2
        kernel = np.zeros((1, 1, c, c), dtype=np.float32)
        for i in range(0, c, 4):
            idx = i >> 2
            kernel[0, 0, idx::out_c, idx]          = [1, 1, 1, 1]
            kernel[0, 0, idx::out_c, idx+out_c]    = [1,-1, 1,-1]
            kernel[0, 0, idx::out_c, idx+out_c*2]  = [1, 1,-1,-1]
            kernel[0, 0, idx::out_c, idx+out_c*3]  = [1,-1,-1, 1]
        self.kernel = tf.keras.backend.variable(value=kernel, dtype='float32')

    def call(self, inputs, **kwargs):
        y = tf.keras.backend.conv2d(inputs, self.kernel, padding='same')
        y = tf.nn.depth_to_space(y, 2)
        return y

    def compute_output_shape(self, input_shape):
        c = input_shape[-1] >> 2
        if(input_shape[1] is not None and input_shape[2] is not None):
            return (input_shape[0], input_shape[1] << 1, input_shape[2] << 1, c)
        else:
            return (None, None, None, c)