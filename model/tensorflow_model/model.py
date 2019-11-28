# -*- encoding: utf-8 -*-
''' model base class for tensorflow

Author: Ze Ma
Date: August 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

import abc
import numpy as np
import tensorflow as tf

# commonly used model building functions for tf
def conv1d_with_act(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

        Args:
            inputs: 3-D tensor variable BxLxC
            num_output_channels: int
            kernel_size: int
            scope: string
            stride: int
            padding: 'SAME' or 'VALID'
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            weight_decay: float
            activation_fn: function
            bn: bool, whether to use batch norm
            bn_decay: float or float tensor variable in [0,1]
            is_training: bool Tensor variable

        Returns:
            Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
    

def fully_connected_with_act(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

        Args:
            inputs: 2-D tensor BxN
            num_outputs: int
  
        Returns:
            Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)
     
        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
    

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
        Args:
            inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
            is_training:   boolean tf.Varialbe, true indicates training phase
            scope:         string, variable scope
            moments_dims:  a list of ints, indicating dimensions for moments calculation
            bn_decay:      float or float tensor variable, controling moving average weight

        Return:
            normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
    
        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.
  
        Args:
            inputs:      Tensor, 2D BxC input
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling moving average weight
            scope:       string, variable scope

        Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

        Args:
            inputs:      Tensor, 3D BLC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling moving average weight
            scope:       string, variable scope

        Return:
            normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.
        Args:
            inputs: tensor
            is_training: boolean tf.Variable
            scope: string
            keep_prob: float in [0,1]
            noise_shape: list of ints

        Returns:
            tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
        return outputs


class TfModelBase(metaclass = abc.ABCMeta):
    """ tensorflow base class
    """
    def __init__(self):
    
    @abc.abstractmethod
    def load_model(self):
        pass
        
    @abc.abstractmethod
    def get_losses(self):
        pass
    
    @abc.abstractmethod
    def set_optimizer(self):
        pass
    
    @abc.abstractmethod
    def set_placeholder(self):
        pass
    
    @abc.abstractmethod
    def train_model(self):
        pass
    
    @abc.abstractmethod
    def evaluate_model(self):
        pass