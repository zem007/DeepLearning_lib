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
    def __init__(self):
#         ImageIo.__init__(self, image_path)
#         self.data_load_path = data_load_path
#         self.data_save_path = data_save_path
        pass


    def load(self, file_path, file_name):
        if os.path.exists(file_path):
            data_array = np.load(os.path.join(file_path, file_name))
            print('load ', file_name, ' completed!')
            return data_array
        else:
            print(file_path, ' does not exists!')
            
    def save(self, save_path, save_name, data):
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        np.save(os.path.join(save_path, save_name), data)
        print(save_name, ' saved to: ', str(save_path))

        
        
