# -*- encoding: utf-8 -*-
''' unet model base on KerasBaseModel in model.py

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

from .model import KerasModelBase
from keras.layers import Deconvolution3D, UpSampling3D

class Model3D(KerasModelBase):
    """ Unet model
    """
    def __init__(self):
        pass
        
    def build_model(self):
        pass
        
    def compile_model(self):
        pass
    
    def callbacks(self):
        pass
    
    def train_model(self):
        pass
    
    def load_model(self):
        pass
    
    def predict(self, test_img):
        pass
    
    def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2), deconvolution=False):
        if deconvolution:
            return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
        else:
            return UpSampling3D(size=pool_size)
        
        