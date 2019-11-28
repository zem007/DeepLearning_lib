""" keras utils
"""
from keras.layers import Conv3D, Activation, BatchNormalization, Deconvolution3D, UpSampling3D


def convolution_block_unet3d(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation='relu',
                                 padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis= -1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                        "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis= -1)(layer)
    return Activation(activation)(layer)
    
def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
    else:
        return UpSampling3D(size=pool_size)
    
    