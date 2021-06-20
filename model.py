import tensorflow as tf
from custom_layers import Space2Depth, Depth2Space, DWT, IWT
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Concatenate, Lambda
from config import Configuration
cfg = Configuration()


def dilation_pyramid(x_in, filters, dilation_rates):
    x_in = Conv2D(filters*2, 3, padding='same')(x_in)
    for dil_rate in dilation_rates:
        x_out = Conv2D(filters, 3, padding='same', dilation_rate=dil_rate)(x_in)
        x_in = Concatenate(axis=-1)([x_in, x_out])
    x_out = Conv2D(filters, 1, padding='same')(x_in)
    return x_out


def pyramid_block(x_in, nFilters, dilation_rates, nPyramidFilters):
    x_out = dilation_pyramid(x_in, nPyramidFilters, dilation_rates)
    x_out = Conv2D(nFilters, 3, padding='same')(x_out)
    x_out = Lambda(lambda x:x*0.1)(x_out)
    x_out = Add()([x_in, x_out])
    return x_out


def encoder(x_in, nFilters, dilation_rates, nPyramidFilters):
    x_out = DWT()(x_in)
    x_out = Lambda(Space2Depth(block_size=2))(x_out)
    x_out = Conv2D(nFilters, 5, padding='same')(x_out)
    x_out = Conv2D(nFilters, 3, padding='same')(x_out)
    x_out = pyramid_block(x_out, nFilters, dilation_rates, nPyramidFilters)
    x_out = Conv2D(nFilters*2, 5, padding='same', strides=(2, 2))(x_out)
    x_out = pyramid_block(x_out, nFilters*2, dilation_rates, nPyramidFilters*2)
    x_out = Conv2D(nFilters*4, 5, padding='same', strides=(2, 2))(x_out)
    x_out = pyramid_block(x_out, nFilters*4, dilation_rates, nPyramidFilters*4)
    return x_out


def decoder(x_in, nFilters, dilation_rates, nPyramidFilters):
    x_out = pyramid_block(x_in, nFilters, dilation_rates, nPyramidFilters)
    x_out = Conv2DTranspose(nFilters/2, 4, strides=(2, 2), padding='same')(x_out)
    x_out = pyramid_block(x_out, nFilters/2, dilation_rates, nPyramidFilters/2)
    x_out = Conv2DTranspose(nFilters/4, 4, strides=(2, 2), padding='same')(x_out)
    x_out = pyramid_block(x_out, nFilters/4, dilation_rates, nPyramidFilters/4)
    x_out = Lambda(Depth2Space(block_size=2))(x_out)
    x_out = Conv2D(3*4, 3, padding='same')(x_out)
    x_out = IWT()(x_out)
    return x_out


def get_model(input_shape):
    x_in = tf.keras.layers.Input(shape=input_shape)
    x_out = encoder(x_in, cfg.nFilters_enc, cfg.dilation_rates, cfg.nPyramidFilters_enc)
    x_out = decoder(x_out, cfg.nFilters_dec, cfg.dilation_rates, cfg.nPyramidFilters_dec)
    return tf.keras.Model(x_in, x_out, name="generator")