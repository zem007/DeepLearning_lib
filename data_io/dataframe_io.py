# -*- encoding: utf-8 -*-
''' module discription: dataframe io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from .data_io import DataIoBase
import pandas as pd
import numpy as np

class DataFrameIo(DataIoBase):
    
    def __init__(self):
        pass
        
    def load(filename):
        df = pd.read_pickle(filename)
        return df
    
    def save(filename, df):
        super().create_folder(filename)
        df.to_pickle(filename)
        print('saved to: ', filename)
        
    