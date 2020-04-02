# -*- encoding: utf-8 -*-
''' metrics functions for tensorflow implementation

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
'''
import tensorflow as tf

def dice_coefficient(pred, label):
    """ pred: BxNxC,
       label: BxN, """
    pred = tf.argmax(pred, axis = 2) # B*N
    pred = tf.to_int32(pred)
    pred = tf.layers.flatten(pred)
    label = tf.layers.flatten(label)
    intersection = tf.reduce_sum(label * pred)
    union = tf.reduce_sum(label) + tf.reduce_sum(pred)
    dice = (2 * intersection) / (union)
    dice = tf.to_float(dice)
    return dice


