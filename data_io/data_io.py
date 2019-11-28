# -*- encoding: utf-8 -*-
''' module discription: data_io class

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import abc

class DataIoBase(metaclass = abc.ABCMeta):
    """ Data Io base class
    """
    
    @abc.abstractmethod
    def load(self, image_path):
        pass
    
    @abc.abstractmethod
    def save(self):
        pass
        
#     @abc.abstractmethod
#     def write(self):
#         pass