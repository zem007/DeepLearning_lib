# -*- encoding: utf-8 -*-
''' loss functions for keras implementation

Author: Ze Ma, Minghui Hu
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from .metrics_keras import dice_coefficient, weighted_dice_coefficient

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)

