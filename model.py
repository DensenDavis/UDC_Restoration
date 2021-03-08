import layers
from layers import DWT, IWT
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Input, LeakyReLU, BatchNormalization, Conv2DTranspose, Concatenate, Add, Lambda, PReLU
from tensorflow.python.keras.models import Model

def conv_relu(x, filters, kernel, use_bias = True, dilation_rate=1):
	if dilation_rate == 0:
		y = tf.keras.layers.Conv2D(filters,1,padding='same',use_bias=use_bias,
			activation='relu')(x)
	else:
		y = tf.keras.layers.Conv2D(filters,kernel,padding='same',use_bias=use_bias,
			dilation_rate=dilation_rate,
			activation='relu')(x)
	return y
def conv(x, filters, kernel, use_bias=True, dilation_rate=1):
	y = tf.keras.layers.Conv2D(filters,kernel,padding='same',use_bias=use_bias,
		dilation_rate=dilation_rate)(x)
	return y


def pyramid_cell(x, filters, dilation_rates):
		for i in range(len(dilation_rates)):
			dilation_rate = dilation_rates[i]
			if i==0:
				t = conv_relu(x,filters,3,dilation_rate=dilation_rate)
				_t = tf.keras.layers.Concatenate(axis=-1)([x,t])
			else:
				t = conv_relu(_t,filters,3,dilation_rate=dilation_rate)
				_t = tf.keras.layers.Concatenate(axis=-1)([_t,t])
		return _t

def get_model(input_shape):
  input_batch = Input(input_shape)

  featureList_input = []

  x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None)(input_batch)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  buffer_layer = x
  
  for i in range(1,5):
    n_filter=int((2**i)*16)
    x = pyramid_cell(buffer_layer,n_filter,(3,2,1,1))
    x = Conv2D(filters=n_filter, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    featureList_input.append(x)
    x = Conv2D(filters=n_filter, kernel_size=(3,3), strides=(2,2), padding='SAME', activation=None)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    buffer_layer=x

  n_filter=int((2**(i+1))*16)
  x = pyramid_cell(buffer_layer,n_filter,(3,2,1,1))
  x = Conv2D(filters=n_filter, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  featureList_input.append(x)
  
  for i in range(1,5):
    n_filter = int(512*(2**-i))
    x = tf.keras.layers.Conv2DTranspose(filters = n_filter, kernel_size = (3, 3),strides=(2, 2),padding='SAME',activation=None)(x)
    x = Concatenate()([x, featureList_input[-i-1]])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=x.shape[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
  x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=tf.nn.tanh)(x)
  G = Model(inputs=input_batch, outputs=x, name="generator")
  return G
