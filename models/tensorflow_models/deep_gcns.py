# -*- encoding: utf-8 -*-
''' DeepGCNs model for tensorflow

source:
https://sites.google.com/view/deep-gcns
http://arxiv.org/abs/1904.03751

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
'''
import logging
logging.basicConfig(level = logging.INFO)
from .graph_base import *
from .model import TfModelBase
from evaluation.metrics_tf import dice_coefficient
import tensorflow as tf
import time
import math
import numpy as np
from utils.tf_utils import *


class DeepGCNs(GraphBase):
    def __init__(self, 
             saver_folder_name, 
             batch_size, 
             num_point, 
             num_coordinate, 
             nums_info, 
             with_adj, 
             with_sample_weight):
        
        self.saver_folder_name = saver_folder_name
        self.batch_size = batch_size
        # output for cls and seg
        self.pred_cls = None
        self.pred_seg = None
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
    
    
    def build(self, 
           kernel_size, 
           padding, 
           stride, 
           bn,
           weight_decay,
           bn_decay, 
           is_dist, 
           num_point,
           num_layers,
           num_neighbors,
           num_filters,
           num_class, 
           skip_connect = None,
           dilations = None):
        
        input_graph = self.x
        is_training = self.training_flag
        info_arr = self.info
        adjs = self.adjs
        if bn_decay == True:
            bn_decay = super()._get_bn_decay(tf.Variable(0, trainable = False), self.batch_size)
        else: bn_decay = None
        
        self.kernel_size = kernel_size
        self.padding= padding
        self.stride= stride
        self.bn= bn
        self.weight_decay= weight_decay
        self.bn_decay= bn_decay
        self.is_dist= is_dist
        
        ### build gcn_backbone_block
        '''Build the gcn backbone block'''
        input_graph = tf.expand_dims(input_graph, -2)
        graphs = []

        for i in range(num_layers):
            #
            if i == 0:
                if adjs == False:
                    neigh_idx = knn_graph(input_graph[:, :, :, :], num_neighbors[i])
                    edge_features = get_edge_feature(input_graph, neigh_idx, num_neighbors[i])
                else:
                    input_graph = tf.squeeze(input_graph, axis = [-2])
                    edge_features = tf.matmul(adjs, input_graph)
                    edge_features = tf.expand_dims(edge_features, -2)
                logging.info('edge features # %d: '%i, edge_features)
                    
                out = super()._conv2d_with_act(inputs = edge_features,
                               num_output_channels = num_filters[i],
                               scope='adj_conv_'+str(i),
                               is_training=is_training, 
                               kernel_size = self.kernel_size,
                               padding=self.padding,
                               stride=self.stride,
                               bn=self.bn,
                               weight_decay=self.weight_decay,
                               bn_decay=self.bn_decay,
                               is_dist=self.is_dist)
                logging.info('out # %d: '%i, out)
                vertex_features = tf.reduce_max(out, axis=-2, keep_dims=True)

                graph = vertex_features
                logging.info('graph layer # %d: '%i, graph)
                graphs.append(graph)
            else:
                if adjs == False:
                    neigh_idx = knn_graph(graphs[-1], num_neighbors[i])
                    edge_features = get_edge_feature(graphs[-1], neigh_idx, num_neighbors[i])
                else:
                    edge_features = tf.matmul(adjs, tf.squeeze(graphs[-1], axis=[-2]))
                    edge_features = tf.expand_dims(edge_features, -2)
                logging.info('edge features # %d: '%i, edge_features)
                
                out = super()._conv2d_with_act(inputs = edge_features,
                               num_output_channels = num_filters[i],
                               scope='adj_conv_'+str(i),
                               is_training=is_training, 
                               kernel_size = self.kernel_size,
                               padding=self.padding,
                               stride=self.stride,
                               bn=self.bn,
                               weight_decay=self.weight_decay,
                               bn_decay=self.bn_decay,
                               is_dist=self.is_dist)
                logging.info('out # %d: '%i, out)
                vertex_features = tf.reduce_max(out, axis=-2, keep_dims=True)

                graph = vertex_features
                logging.info('graph layer # %d: '%i, graph)
                if skip_connect == 'residual':
                    graph = graph + graphs[-1]
                elif skip_connect == 'dense':
                    graph = tf.concat([graph, graphs[-1]], axis=-1)
                elif skip_connect == 'none':
                    graph = graph
                else:
                    raise Exception('Unknown connections')
                graphs.append(graph)
        
        ### build_gcn_backbone_block end, return graphs
        
        ### build_fusion_block start
        out = super()._conv2d_with_act(inputs = tf.concat(graphs, axis=-1),
                       num_output_channels = 1024,
                       scope='adj_conv_'+'final',
                       is_training=is_training, 
                       kernel_size = self.kernel_size,
                       padding=self.padding,
                       stride=self.stride,
                       bn=self.bn,
                       weight_decay=self.weight_decay,
                       bn_decay=self.bn_decay,
                       is_dist=self.is_dist)
        logging.info('out: ', out)
        out_max = super()._max_pool2d(out, [num_point, 1], padding='VALID', scope='maxpool')
        logging.info('out_max: ', out_max)
        
        # classification
#         pred_cls = tf.reshape(out_max, [batch_size, -1])   # N * num_point
        pred_cls = tf.squeeze(out_max)   # N * num_point
        pred_cls = tf.concat([pred_cls, info_arr], axis = 1) # N*(num_point + num_clinic_feature + num_morphological_feature)
        pred_cls = super()._fully_connected_with_act(pred_cls, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=self.bn_decay, is_dist=self.is_dist)
        pred_cls = super()._fully_connected_with_act(pred_cls, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=self.bn_decay, is_dist=self.is_dist)
        pred_cls = super()._fully_connected_with_act(pred_cls, num_class, activation_fn=None, scope='fc3', is_dist=self.is_dist)
        self.pred_cls = pred_cls
        
        expand = tf.tile(out_max, [1, num_point, 1, 1])
        logging.info('expand: ', expand)
        fusion = tf.concat(axis=3, values=[expand]+graphs)
        logging.info('fusion: ', fusion)
        ### build_fusion_block end, return pred_cls and fusion
        
        ### build_mlp_pred_block start
        self.bn_decay = None
        out = super()._conv2d_with_act(inputs = fusion,
                       num_output_channels = 512,
                       scope='seg/conv1',
                       is_training=is_training, 
                       kernel_size = self.kernel_size,
                       padding=self.padding,
                       stride=self.stride,
                       bn=self.bn,
                       weight_decay=self.weight_decay,
                       bn_decay=self.bn_decay,
                       is_dist=self.is_dist)
        out = super()._conv2d_with_act(inputs = out,
                       num_output_channels = 256,
                       scope='seg/conv2',
                       is_training=is_training, 
                       kernel_size = self.kernel_size,
                       padding=self.padding,
                       stride=self.stride,
                       bn=self.bn,
                       weight_decay=self.weight_decay,
                       bn_decay=self.bn_decay,
                       is_dist=self.is_dist)
        out = super()._dropout(out, keep_prob=0.7, scope='dp1', is_training=is_training)
        self.bn = False
        out = super()._conv2d_with_act(inputs = out,
                       num_output_channels = num_class,
                       scope='seg/conv3',
                       activation_fn=None, 
                       kernel_size = self.kernel_size,
                       padding=self.padding,
                       stride=self.stride,
                       bn=self.bn,
                       weight_decay=self.weight_decay,
                       bn_decay=self.bn_decay,
                       is_dist=self.is_dist)
        logging.info('final out: ', out)
        pred = tf.squeeze(out, [2])
        self.pred_seg = pred
        logits = tf.nn.sigmoid(pred_cls) # B*2
        self.logits = logits
        ### end 

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
    