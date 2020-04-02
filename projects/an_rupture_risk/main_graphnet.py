# -*- encoding: utf-8 -*-
''' generate, load, or save data for 3D An segmentation 3DUnet model

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import sys
sys.path.append('/home/maze/USAL/libs/')
import os
import logging
logging.basicConfig(level = logging.INFO)
import numpy as np
import tensorflow as tf
import math
import argparse
from models.tensorflow_models.graphnet import Graphnet
from models.tensorflow_models.model import TfModelBase
from utils_project import *
from data_io.dataframe_io import DataFrameIo
from data_io.stl_io import StlIo
from data_io.ela_io import ElaIo
from utils.data_io_utils import split_dataset
from mappers.generator_tensorflow import GeneratorPointCloud
import scipy.sparse as sp
from sklearn import preprocessing
import vtk

# predefine something
which_gpu = '0'
data_saver_filename = '/home/maze/USAL_backup/df_1024_triangle.pkl'
saver_folder_name = '/home/maze/USAL/libs/model_saver'
num_class = 2
with_adj = True
with_sample_weight = None
batch_size = 32
epochs = 100
learning_rate = 0.001
momentum = 0.9
shuffle = True
with_aug = False
input_optimizer = 'adam'
input_generator = GeneratorPointCloud(rotate=True, jitter=True, random_scale=False, 
                         scale_low = 0.8, scale_high = 1.25, rotate_perturbation=False, shift=True)


def main(mode, df_original, data_generation = None):
    tf.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
    ### if need to generate data, then generator it, if not then load it
    if data_generation == None:
        df = DataFrameIo.load(data_saver_filename)
    elif data_generation == 'point data':
        df = generate_point_data(df_original, decimate_ratio = 0.65, num_point = 1024, save = False, filename = None)
    elif data_generation == 'triangle data':
        df = generate_triangle_data(df_original, decimate_ratio = 0.65, num_triangle = 1024, save = False, filename = None)
    df = df.drop(['Rupture'], axis = 1)
    # split train and val dataset
    dir_list = list(df['file_name'])
    train_list, val_list, test_list = split_dataset(img_list = dir_list, split_id = -11, test_id=[0], val_id=[1])
    for i in range(len(df)):
        if df.iloc[i]['file_name'] in train_list:
            df.iloc[i]['data_set'] = 'train_set'
        elif df.iloc[i]['file_name'] in val_list:
            df.iloc[i]['data_set'] = 'validation_set'
        elif df.iloc[i]['file_name'] in test_list:
            df.iloc[i]['data_set'] = 'test_set'
    df_train = df[df.data_set == 'train_set']
    df_val = df[df.data_set == 'validation_set']
    df_test = df[df.data_set == 'test_set']
                      
    # to array
    x_train, labels_train, adj_train, info_train = read_all_info(df_train)
    minmaxscaler = preprocessing.MinMaxScaler()
    info_train = minmaxscaler.fit_transform(info_train)
    # to array
    x_val, labels_val, adj_val, info_val = read_all_info(df_val)
    info_val = minmaxscaler.transform(info_val)
    
    #convert to cls and seg labels
    labels_train_cls = convert_to_cls_labels(labels_train)
    labels_train_seg = convert_to_seg_labels(labels_train)
    labels_val_cls = convert_to_cls_labels(labels_val)
    labels_val_seg = convert_to_seg_labels(labels_val)
    
    #define placeholders shapes
    num_train_sample = x_train.shape[0]
    num_val_sample = x_val.shape[0]
    num_point = x_train.shape[1]
    num_coordinate = x_train.shape[2]
    nums_info = info_train.shape[1]
    
    
    if mode == 'train':
        graph = tf.Graph()
        with graph.as_default():
            ### import graphnet
            graphnet_class = Graphnet(saver_folder_name = saver_folder_name, 
                              num_class = num_class, 
                              batch_size = batch_size, 
                              num_point = num_point, 
                              num_coordinate = num_coordinate, 
                              nums_info = nums_info, 
                              with_adj = with_adj, 
                              with_sample_weight = with_sample_weight)
            # build model
            graphnet_class.build(bn_decay = True)
            # compile 
            graphnet_class.compile(input_optimizer, tf.Variable(0), batch_size, learning_rate, momentum)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep = 1)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = True
            
        # open a new session
        with tf.Session(graph = graph, config = config) as sess:
            graphnet_class.train(sess, 
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
                          shuffle, 
                          with_aug)
        
        
    if mode == 'evaluate':
        graph = tf.Graph()
        with graph.as_default():
            ### import graphnet
            graphnet_class = Graphnet(saver_folder_name = saver_folder_name, 
                              num_class = num_class, 
                              batch_size = num_val_sample, 
                              num_point = num_point, 
                              num_coordinate = num_coordinate, 
                              nums_info = nums_info, 
                              with_adj = with_adj, 
                              with_sample_weight = with_sample_weight)
            # Get model and loss
            graphnet_class.build()

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        with tf.Session(graph = graph) as sess:
            graphnet_class.load(sess, saver)
            graphnet_class.evaluate(x_val, labels_val_cls, info_val, adj_val)

    if mode == 'predict_classes':
        graph = tf.Graph()
        with graph.as_default():
            ### import graphnet
            graphnet_class = Graphnet(saver_folder_name = saver_folder_name, 
                              num_class = num_class, 
                              batch_size = num_val_sample, 
                              num_point = num_point, 
                              num_coordinate = num_coordinate, 
                              nums_info = nums_info, 
                              with_adj = with_adj, 
                              with_sample_weight = with_sample_weight)
            # Get model and loss
            graphnet_class.build()

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        with tf.Session(graph = graph) as sess:
            graphnet_class.load(sess, saver)
            pred_test = graphnet_class.predict_classes(x_test, info_test, adj_test)
            return pred_test


