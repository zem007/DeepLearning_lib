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


class TfModelBase(metaclass = abc.ABCMeta):
    """ tensorflow base class
    """
    def __init__(self, 
             BASE_LEARNING_RATE = 0.001,
             BATCH_SIZE = 32,
             DECAY_STEP = 300000,
             DECAY_RATE = 0.5,
             BN_INIT_DECAY = 0.5,
             BN_DECAY_DECAY_RATE = 0.5,
             BN_DECAY_DECAY_STEP = float(300000),
             BN_DECAY_CLIP = 0.99):
        self.BASE_LEARNING_RATE = BASE_LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.DECAY_STEP = DECAY_STEP
        self.DECAY_RATE = DECAY_RATE
        self.BN_INIT_DECAY = BN_INIT_DECAY
        self.BN_DECAY_DECAY_RATE = BN_DECAY_DECAY_RATE
        self.BN_DECAY_DECAY_STEP = BN_DECAY_DECAY_STEP
        self.BN_DECAY_CLIP = BN_DECAY_CLIP
    
    @abc.abstractmethod
    def build(self):
        pass
    
    @abc.abstractmethod
    def compile(self):
        pass
    
    @abc.abstractmethod
    def train(self):
        pass
    
    @abc.abstractmethod
    def load(self):
        pass
    
    @abc.abstractmethod
    def predict_classes(self):
        pass
    
    @abc.abstractmethod
    def evaluate(self):
        pass
    
    def _variable_on_cpu(self, name, shape, initializer, use_fp16=False):
        """ Helper to create a Variable stored on CPU memory.

            Args:
                name: name of the variable
                shape: list of ints
                initializer: initializer for Variable

            Returns:
                Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float16 if use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, use_xavier=True):
        """ Helper to create an initialized Variable with weight decay.
            Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.

            Args:
                name: name of the variable
                shape: list of ints
                stddev: standard deviation of a truncated Gaussian
                wd: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.
                use_xavier: bool, whether to use xavier initializer

            Returns:
                Variable Tensor
        """
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = self._variable_on_cpu(name, shape, initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
    
    def _conv2d_with_act(self,
                   inputs,
                   num_output_channels,
                   kernel_size,
                   scope,
                   stride=[1, 1],
                   padding='SAME',
                   use_xavier=True,
                   stddev=1e-3,
                   weight_decay=0.0,
                   activation_fn=tf.nn.relu,
                   bn=False,
                   bn_decay=None,
                   is_training=None,
                   is_dist=False):
        """ 2D convolution with non-linear operation.

            Args:
                inputs: 4-D tensor variable BxHxWxC
                num_output_channels: int
                kernel_size: a list of 2 ints
                scope: string
                stride: a list of 2 ints
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
                kernel_h, kernel_w = kernel_size
                num_in_channels = inputs.get_shape()[-1].value
                kernel_shape = [kernel_h, kernel_w,
                                  num_in_channels, num_output_channels]
                kernel = self._variable_with_weight_decay('weights',
                                               shape=kernel_shape,
                                               use_xavier=use_xavier,
                                               stddev=stddev,
                                               wd=weight_decay)
                stride_h, stride_w = stride
                outputs = tf.nn.conv2d(inputs, kernel,
                                 [1, stride_h, stride_w, 1],
                                 padding=padding)
                biases = self._variable_on_cpu('biases', [num_output_channels],
                                      tf.constant_initializer(0.0))
                outputs = tf.nn.bias_add(outputs, biases)

                if bn:
                    outputs = self._batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist = is_dist)

                if activation_fn is not None:
                    outputs = activation_fn(outputs)

                return outputs
    
    def _max_pool2d(self,
               inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
        """ 2D max pooling.

            Args:
                inputs: 4-D tensor BxHxWxC
                kernel_size: a list of 2 ints
                stride: a list of 2 ints

            Returns:
                Variable tensor
        """
        with tf.variable_scope(scope) as sc:
            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = stride
            outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
            return outputs
        
    def _fully_connected_with_act(self,
                        inputs,
                        num_outputs,
                        scope,
                        use_xavier=True,
                        stddev=1e-3,
                        weight_decay=0.0,
                        activation_fn=tf.nn.relu,
                        bn=False,
                        bn_decay=None,
                        is_training=None,
                        is_dist=False):
        """ Fully connected layer with non-linear operation.

            Args:
                inputs: 2-D tensor BxN
                num_outputs: int

            Returns:
                Variable tensor of size B x num_outputs.
        """
        with tf.variable_scope(scope) as sc:
            num_input_units = inputs.get_shape()[-1].value
            weights = self._variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
            outputs = tf.matmul(inputs, weights)
            biases = self._variable_on_cpu('biases', [num_outputs],
                                 tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

            if bn:
                outputs = self._batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist = is_dist)

            if activation_fn is not None:
                outputs = activation_fn(outputs)
            return outputs
        
    def _get_learning_rate(self, batch, base_learning_rate, batch_size):
        learning_rate = tf.train.exponential_decay(
                            base_learning_rate,  # self.Base learning rate.
                            batch * batch_size,  # self.BATCH_SIZE
                            300000,          # self.Decay step.
                            0.5,          # self.Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate        

    def _get_bn_decay(self, batch, batch_size):
        bn_momentum = tf.train.exponential_decay(
                          0.5,    #BN_INIT_DECAY
                          batch*batch_size,    # self.BATCH_SIZE
                          float(300000),    # self.BN_DECAY_DECAY_STEP
                          0.5,    # self.BN_DECAY_DECAY_RATE
                          staircase=True)
        bn_decay = tf.minimum(0.99, 1 - bn_momentum)    #self.BN_DECAY_CLIP
        return bn_decay
    
    def _dropout(self,
            inputs,
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

    def _batch_norm_template(self, inputs, is_training, scope, moments_dims, bn_decay):
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


    def _batch_norm_dist_template(self, inputs, is_training, scope, moments_dims, bn_decay):
        """ The batch normalization for distributed training.
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
            beta = self._variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
            gamma = self._variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

            pop_mean = self._variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer())  # trainable=False
            pop_var = self._variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer())    # trainable=False

            def train_bn_op():
                batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
                decay = bn_decay if bn_decay is not None else 0.9
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

            def test_bn_op():
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

            normed = tf.cond(is_training, train_bn_op, test_bn_op)
            return normed
        
    def _batch_norm_for_conv2d(self, inputs, is_training, bn_decay, scope, is_dist):
        """ Batch normalization on 2D convolutional maps.

            Args:
                inputs:      Tensor, 4D BHWC input maps
                is_training: boolean tf.Varialbe, true indicates training phase
                bn_decay:    float or float tensor variable, controling moving average weight
                scope:       string, variable scope

            Return:
                normed:      batch-normalized maps
        """
        if is_dist:
            return self._batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
        else:
            return self._batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)
        
    def _batch_norm_for_fc(self, inputs, is_training, bn_decay, scope, is_dist):
        """ Batch normalization on FC data.

            Args:
                inputs:      Tensor, 2D BxC input
                is_training: boolean tf.Varialbe, true indicates training phase
                bn_decay:    float or float tensor variable, controling moving average weight
                scope:       string, variable scope

            Return:
                normed:      batch-normalized maps
        """
        if is_dist:
            return self._batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
        else:
            return self._batch_norm_template(inputs, is_training, scope, [0,], bn_decay)
        