# -*- encoding: utf-8 -*-
''' module discription: dataframe io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from .data_io import DataIoBase

class DataFrameIo(DataIoBase):
    """ pd.dataframe io

        Args:
        
        Returns:
            
    """
    def __init__(self, data_load_path, data_save_path):
        self.data_load_path = data_load_path
        self.data_save_path = data_save_path
        
    def load(self):
    
    
    def write(self):
    
    
    