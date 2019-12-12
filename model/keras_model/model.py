# -*- encoding: utf-8 -*-
''' model base class for keras

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

import abc

class KerasModelBase(metaclass = abc.ABCMeta):
    """ keras base class
    """
    
    @abc.abstractmethod
    def build_model(self):
        pass
        
    @abc.abstractmethod
    def compile_model(self):
        pass
    
    @abc.abstractmethod
    def callbacks(self):
        pass
    
    @abc.abstractmethod
    def train_model(self):
        pass
    
    @abc.abstractmethod
    def load_model(self):
        pass
    
    @abc.abstractmethod
    def predict(self, test_img):
        pass

        
        
    