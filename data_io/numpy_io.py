# -*- encoding: utf-8 -*-
''' module discription: numpy io, derived class from ImageIo in image_io.py

Author: Ze Ma
Date: Novemeber 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import os
import numpy as np
from .image_io import ImageIo

class NumpyIo(ImageIo):
    """ pd.dataframe io

        Args:
        
        Returns:
            
    """
    def __init__(DataIoBase):
        pass


    def load(self, path):
        data_array = np.load(path)
        print('load ', path, ' completed!')
        return data_array
            
    def save(self, path, data):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        np.save(path, data)
        print('saved to: ', path)

        
        
