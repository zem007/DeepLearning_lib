# -*- encoding: utf-8 -*-
''' metrics class for general case no matter for tf or keras

Author: Ze Ma, Minghui Hu
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

import abc

class MetricBase(metaclass = abc.ABCMeta):
    """ metrics base class
    
    """
        
    @abc.abstractmethod
    def get_input(self):
        
        
    @abc.abstractmethod
    def get_output(self):


    @abc.abstractmethod
    def plot(self):
        
        