# -*- encoding: utf-8 -*-
''' unet model base on KerasBaseModel in model.py

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

from .model import KerasModelBase
from keras.layers import Conv3D, Activation, BatchNormalization, Deconvolution3D, UpSampling3D

class Model3D(KerasModelBase):
    """ Unet model
    """
    def __init__(self):
        pass
        
    def build(self):
        pass
        
    def compile_model(self):
        pass
    
    def callbacks(self):
        pass
    
    def train(self):
        pass
    
    def train_generator(self):
        pass
    
    def load(self):
        pass
    
    def predict(self, test_img):
        pass
    
    def predict(self, x_test, labels_test):
        pass
    
    def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False):
        if deconvolution:
            return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
        else:
            return UpSampling3D(size=pool_size)

    def convolution_block(input_layer, 
                   n_filters, 
                   batch_normalization=True, 
                   kernel=(3, 3, 3), 
                   activation='relu',
                   padding='same', 
                   strides=(1, 1, 1)):
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis= -1)(layer)
        return Activation(activation)(layer)
        
        
