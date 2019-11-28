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
    def __init__(self, data_save_path):
#         ImageIo.__init__(self, image_path)
#         self.data_load_path = data_load_path
        self.data_save_path = data_save_path
    
    def load_array(self):
        save_path = self.data_save_path
        if os.path.exists(save_path):
            print('load data from: ', save_path)
            x_train = np.load(os.path.join(save_path, 'x_train.npy'))
            labels_train = np.load(os.path.join(save_path, 'labels_train.npy'))
            x_test = np.load(os.path.join(save_path, 'x_test.npy'))
            labels_test = np.load(os.path.join(save_path, 'labels_test.npy'))
            data_seperate = np.load(os.path.join(save_path, 'data_seperate.npy'))
            print('load data completed!')
            return (x_train, labels_train), (x_test, labels_test), data_seperate
        else:
            print(save_path, ' does not exist!')
            
    def save_array(self, x_train, labels_train, x_test, labels_test, data_seperate):
        save_path = self.data_save_path
        os.mkdir(save_path)
        np.save(os.path.join(save_path, 'x_train.npy'), x_train)
        np.save(os.path.join(save_path, 'labels_train.npy'), labels_train)
        np.save(os.path.join(save_path, 'x_test.npy'), x_test)
        np.save(os.path.join(save_path, 'labels_test.npy'), labels_test)
        np.save(os.path.join(save_path, 'data_seperate.npy'), data_seperate)
        print('data saved to: ', str(save_path))
        
    def to_array(self, image_path):
        # get one image array
        img_array = ImageIo.load(image_path)
        return img_array
        
