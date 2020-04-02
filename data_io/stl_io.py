# -*- encoding: utf-8 -*-
''' module discription: dataframe io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from .data_io import DataIoBase
import pandas as pd
import vtk

class StlIo(DataIoBase):
    
    def __init__(self):
        pass
        
    def load(filename):
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(filename)
        stl_reader.Update()
        return stl_reader.GetOutput()
    
    def save(self, filename, df):
#         super().create_folder(filename)
#         df.to_pickle(filename)
#         print('saved to: ', filename)
        pass