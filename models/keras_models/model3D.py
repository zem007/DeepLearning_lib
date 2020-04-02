# -*- encoding: utf-8 -*-
''' 3D model base on KerasBaseModel in model.py

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
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
        
    def compile(self):
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
    
    def get_up_convolution(self, n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=True):
        """ up conv structure
        Args:
            n_filters: int, the nums of filters that is used in this convolution layer
            pool_size: tuple of int, eg.(3,3,3)
            kernel_size: tuple of int, eg. (2,2,2)
            strides: tuple of int, eg. (2,2,2)
            deconvolution: bool, indicate whether to devconvolution or upsampling, default using deconvolution
        Returns:
            keras layer
        """
        if deconvolution:
            return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
        else:
            return UpSampling3D(size=pool_size)

    def convolution_block(self, 
                   input_layer, 
                   n_filters, 
                   batch_normalization=True, 
                   kernel=(3, 3, 3), 
                   activation='relu',
                   padding='same', 
                   strides=(1, 1, 1)):
        """ 3d conv block
        Args:
            input_layer: keras layer
            n_filters: int, the nums of filters that is used in this convolution layer
            batch_normalization: bool, indicate whether to use batch_norm after this conv layer
            kernel: tuple of int, eg. (2,2,2)
            activation: str, activation function for this layer, default by 'relu'
            padding: str, indicate whether to add padding, 'same' or 'none'
            strides: tuple of int, eg. (2,2,2)
        Returns:
            keras layer
        """
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis= -1)(layer)
        return Activation(activation)(layer)
        
        
