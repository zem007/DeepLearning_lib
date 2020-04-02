# -*- encoding: utf-8 -*-
''' module discription: ela io, base class is DataIoBase in data_io.py

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from .data_io import DataIoBase
import pandas as pd
import numpy as np

class ElaIo(DataIoBase):
    
    def __init__(self):
        pass
    
    def save(self):
        pass
    
    def load(filename):
        anData=[]
        with open(filename, 'r') as file_to_read:
            lineNum=0
            while True:
                line = file_to_read.readline()
    #             linelist.append(line)
                if not line:
                    break
                if lineNum==0 and ('ELA' not in line):
                    print('Wrong head!')
                    print(filename)
                    break;
                if lineNum>1:
                    lineData = [int(i) for i in line.split(' ')]
                    anData.append(lineData)
                    #print(lineData)
                lineNum = lineNum+1
        return np.array(anData)