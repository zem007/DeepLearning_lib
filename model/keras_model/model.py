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
    def build(self):
        pass
        
    @abc.abstractmethod
    def compile_model(self):
        pass
    
    @abc.abstractmethod
    def callbacks(self):
        pass
    
    @abc.abstractmethod
    def train(self):
        pass
    
    @abc.abstractmethod
    def train_generator(self):
        pass
    
    @abc.abstractmethod
    def load(self):
        pass
    
    @abc.abstractmethod
    def predict(self, test_img):
        pass
    
    @abc.abstractmethod
    def evaluate(self, x_test, labels_test):
        pass

        
        
    