# -*- encoding: utf-8 -*-
''' metrics functions for keras implementation

Author: Ze Ma, Minghui Hu
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
from functools import partial
from keras import backend as K
from common.variables import *

def dice_coefficient(y_true, y_pred, smooth=DSC_SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def weighted_dice_coefficient(y_true, y_pred, axis=(-4, -3, -2), smooth=DSC_SMOOTH):
    """
    Weighted dice coefficient. Default axis assumes a "channels last" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2.*(K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    """ like a wrapper, show the dice coefficient function for each label
    """
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

