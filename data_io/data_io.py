# -*- encoding: utf-8 -*-
''' module discription: data_io class

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import abc
import os

class DataIoBase(metaclass = abc.ABCMeta):
    """ Data Io base class
    """
    
    @abc.abstractmethod
    def load(self, filename):
        pass
    
    @abc.abstractmethod
    def save(self, filename, data):
        pass
    
    def create_folder(self, filename):
        dirname = os.path.dirname(filename)
        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        else: 
            pass
        