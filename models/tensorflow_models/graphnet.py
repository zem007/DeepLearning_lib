# -*- encoding: utf-8 -*-
''' graphnet model for tensorflow

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
'''
import logging
logging.basicConfig(level = logging.INFO)
from .graph_base import *
from evaluation.metrics_tf import dice_coefficient
import tensorflow as tf
import time
import math


class Graphnet(GraphBase):
    """ graphnet model based on TfModelBase in model.py
    """
    def __init__(self,
             saver_folder_name,
             num_class, 
             batch_size, 
             num_point, 
             num_coordinate, 
             nums_info, 
             with_adj, 
             with_sample_weight):
        """ initialization
           Args:
               
        """
        self.saver_folder_name = saver_folder_name
        self.num_class = num_class
        self.pred_seg = None
        self.pred_cls = None
        self.sess = None
        self.logits = None
        self._get_placeholders(batch_size, num_point, num_coordinate, nums_info, with_adj, with_sample_weight)
        
    def _get_placeholders(self, batch_size, num_point, num_coordinate, nums_info, with_adj, with_sample_weight):
        x, labels_seg, labels_cls, info, training_flag, adjs, sample_weight = super()._get_placeholders(
                     batch_size, num_point, num_coordinate, nums_info, with_adj, with_sample_weight)
        
        self.x = x
        self.labels_seg = labels_seg
        self.labels_cls = labels_cls
        self.info = info
        self.training_flag = training_flag
        self.adjs = adjs
        self.sample_weight = sample_weight
    
    
    def build(self, bn_decay=False):
        """ Graphnet structure
           input is BxNx3, adjs is BxNxN, output BxNx50 
           Args:
               point_cloud: np.array of B x point_num x coordinate_num, coordinate_nums is 3 for point cloud
               info_arr: np.array of B x feature_num
               is_training: bool, training or evaluation
               adjs: tf.tensor, if adjacent matrix is not added, then adjs = False
               bn_decay: batchnorm decay
           Returns:
               net_cls: output tf.tensor for the classification output
               net_seg: output tf.tensor for the classification output
        """
        point_cloud = self.x
        adjs = self.adjs
        is_training = self.training_flag
        info_arr = self.info
        
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        num_coordinate = point_cloud.get_shape()[2].value
        
        if bn_decay == True:
            bn_decay = super()._get_bn_decay(tf.Variable(0), batch_size)
        else: bn_decay = None

        input_image = tf.expand_dims(point_cloud, -1)
        if adjs != False:
            net = tf.matmul(adjs, point_cloud)
            input_image = tf.expand_dims(net, -1)
        # if adjacent matrix is not added, the model is reduced to pointnet
        net = super()._conv2d_with_act(input_image, 64, [1, num_coordinate],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        point_feat = net

        if adjs != False:
            point_feat = tf.matmul(adjs, tf.squeeze(point_feat, axis=[2]))
            point_feat = tf.expand_dims(point_feat, 2)
        net = super()._conv2d_with_act(point_feat, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        global_feat = super()._max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')

        # classification
        net_cls = tf.reshape(global_feat, [batch_size, -1])   # N * 1024
        net_cls = tf.concat([net_cls, info_arr], axis = 1) # N*(1024+36)
        net_cls = super()._fully_connected_with_act(net_cls, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net_cls = super()._fully_connected_with_act(net_cls, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net_cls = super()._fully_connected_with_act(net_cls, self.num_class, activation_fn=None, scope='fc3')

        # segmentation
        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        concat_feat = tf.concat([point_feat, global_feat_expand], 3)

        if adjs != False:
            concat_feat = tf.matmul(adjs, tf.squeeze(concat_feat, axis=[2]))
            concat_feat = tf.expand_dims(concat_feat, 2)
        net = super()._conv2d_with_act(concat_feat, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        if adjs != False:
            net = tf.matmul(adjs, tf.squeeze(net, axis=[2]))
            net = tf.expand_dims(net, 2)
        net = super()._conv2d_with_act(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay)
        net = super()._conv2d_with_act(net, self.num_class, [1,1],
                             padding='VALID', stride=[1,1], activation_fn=None,
                             scope='conv10')
        net_seg = tf.squeeze(net, [2]) # BxNxC

        self.pred_cls = net_cls
        self.pred_seg = net_seg
        logits = tf.nn.sigmoid(net_cls) # B*2
        self.logits = logits
 
    def compile(self, input_optimizer, batch, batch_size, base_learning_rate, momentum):
        labels_seg = self.labels_seg
        labels_cls = self.labels_cls
        sample_weight = self.sample_weight
        
        loss_seg, loss_cls, total_loss, train_op, dice = super().compile(
        labels_seg, labels_cls, input_optimizer, batch, batch_size, base_learning_rate, momentum, sample_weight)
        
        self.loss_seg = loss_seg
        self.loss_cls = loss_cls
        self.total_loss = total_loss
        self.train_op = train_op
        self.dice = dice

    def train(self,
           sess,
           saver, 
           x_train, 
           labels_train_cls, 
           labels_train_seg,  
           info_train,
           adj_train,
           num_train_sample, 
           x_val, 
           labels_val_cls, 
           labels_val_seg, 
           info_val,
           adj_val,
           num_val_sample, 
           batch_size, 
           num_point, 
           learning_rate,
           epochs,  
           input_generator,
           shuffle = True, 
           with_aug = False):
        
        x = self.x
        labels_cls = self.labels_cls
        labels_seg = self.labels_seg
        info = self.info
        adjs = self.adjs
        training_flag = self.training_flag
        
        super().train(sess,
                   saver, 
                   x,
                   labels_cls, 
                   labels_seg, 
                   info, 
                   adjs, 
                   training_flag, 
                   x_train, 
                   labels_train_cls, 
                   labels_train_seg,  
                   info_train,
                   adj_train,
                   num_train_sample, 
                   x_val, 
                   labels_val_cls, 
                   labels_val_seg, 
                   info_val,
                   adj_val,
                   num_val_sample, 
                   batch_size, 
                   num_point, 
                   learning_rate,
                   epochs,  
                   input_generator,
                   shuffle, 
                   with_aug)
    
    def load(self, sess, saver):
        super().load(sess, saver)
    
    def predict_classes(self, x_test, info_test, adj_test):
        x = self.x
        info = self.info
        adjs = self.adjs
        training_flag = self.training_flag
        
        pred_test = super().predict_classes(self.logits, x, info, adjs, training_flag, x_test, info_test, adj_test)
        return pred_test

    def evaluate(self, x_val, labels_val_cls, info_val, adj_val):
        x = self.x
        info = self.info
        adjs = self.adjs
        training_flag = self.training_flag
        
        super().evaluate(self.logits, x, info, adjs, training_flag, x_val, labels_val_cls, info_val, adj_val)
    