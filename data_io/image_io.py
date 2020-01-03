# -*- encoding: utf-8 -*-
''' module discription: image io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: Novemeber 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import SimpleITK as sitk
from .data_io import DataIoBase

class ImageIo(DataIoBase):
    """ pd.dataframe io

        Args:
        
        Returns:
            
    """
    def __init__(self):
        pass
        
    def load(self, filename):
        img = sitk.ReadImage(filename)
        img_array = sitk.GetArrayFromImage(img)
        img_array = img_array.transpose(2,1,0) # zyx to xyz
        return img_array
    
    def save(self, filename, data):
        pass

    
    
    
    