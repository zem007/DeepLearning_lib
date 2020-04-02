# -*- encoding: utf-8 -*-
''' module discription: numpy io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: Novemeber 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
'''
import os
import numpy as np
from .data_io import DataIoBase

class NumpyIo(DataIoBase):
    """ manipulate np.array
    """
    def __init__(self):
        pass


    def load(self, filename):
        data_array = np.load(filename)
        print('load ', filename, ' completed!')
        return data_array
            
    def save(self, filename, data):
        super().create_folder(filename)
        np.save(filename, data)
        print('saved to: ', filename)

        
