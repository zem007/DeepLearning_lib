# -*- encoding: utf-8 -*-
''' generate, load, or save data for segmentation, 3DUnet model

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''

import numpy as np
import argparse
from data_io.numpy_io import NumpyIo
from model.keras_model.unet3D import Unet3D
from utils.data_io_utils import get_file_dir_lists, random_seperate_dataset, small_image_mask_block

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default = '/data/username/Image/')
parser.add_argument('--an_path', default = '/data/username/Mask1/')
parser.add_argument('--vessel_path', default = '/data/username/Mask2/')
parser.add_argument('--data_save_path', default = '/home/username/data')
parser.add_argument('--generate', default = False)
parser.add_argument('--save', default = False)
parser.add_argument('--load', default = True)
parser.add_argument('--which_gpu', default = '0')
parser.add_argument('--mode', default = 'predict')
parser.add_argument('--n_labels', default = 3)
parser.add_argument('--input_shape', default = (64,64,64,1))
parser.add_argument('--pool_size', default = (2,2,2))
parser.add_argument('--deconvolution', default = True)
parser.add_argument('--depth', default = 4)
parser.add_argument('--n_base_filters', default = 32)
parser.add_argument('--batch_normalization', default = True)
parser.add_argument('--activation_name', default = 'softmax')
parser.add_argument('--model_save_path', default = '/home/username/model_save_path')
parser.add_argument('--include_label_wise_dice_coefficients', default = True)
parser.add_argument('--initial_learning_rate', default = 1e-5)
parser.add_argument('--metrics', default = 'weighted_dice_coefficient')
parser.add_argument('--loss_fun', default = 'weighted_dice_coefficient_loss')
parser.add_argument('--monitor', default = 'val_loss')
parser.add_argument('--save_best_only', default = True)
parser.add_argument('--using_data_augmentation', default = False)
parser.add_argument('--batch_size', default = 8)
parser.add_argument('--epochs', default = 100)
parser.add_argument('--verbose', default = 1)
parser.add_argument('--shuffle', default = True)


def generate_data(numpyIo, data_load_path, drop_data = True, block_size = 64):
        print('load data from the source dir: ', data_load_path)
        img_dir_list, an_dir_list, vessel_dir_list = get_file_dir_lists(data_load_path, drop_data)
        x_train = []
        labels_train = []
        x_test = []
        labels_test = []
        data_seperate = []
        for i in range(len(img_dir_list)):
            print('processing: ', an_dir_list[i]) #print name
            # random seperate train and test set 80% and 20%
            data_set = random_seperate_dataset(ratio = 0.8)
            # read the whole image, an_mask and vessel as array
            img_array = numpyIo.to_array(img_dir_list[i])
            mask_array = numpyIo.to_array(an_dir_list[i])
            vessel_array = numpyIo.to_array(vessel_dir_list[i])
            # get the small image and mask block array for that image and mask, output shape [n,64,64,64]
            small_img_array, small_mask_array = small_image_mask_block(img_array, mask_array, vessel_array, block_size)
            if data_set == 'train_set':
                x_train.append(small_img_array)
                labels_train.append(small_mask_array)
                data_seperate.append([an_dir_list[i][-31:], 'train_set'])
            elif data_set == 'test_set':
                x_test.append(small_img_array)
                labels_test.append(small_mask_array)
                data_seperate.append([an_dir_list[i][-31:], 'test_set'])
        # list to array
        x_train = np.concatenate(x_train, axis = 0)
        labels_train = np.concatenate(labels_train, axis = 0)
        x_test = np.concatenate(x_test, axis = 0)
        labels_test = np.concatenate(labels_test, axis = 0)
        data_seperate = np.array(data_seperate)
        
        return (x_train, labels_train), (x_test, labels_test), data_seperate
    
def load_or_save_data(generate, save, load, data_save_path, data_load_path):
    numpyIo = NumpyIo(data_save_path = data_save_path)
    
    if generate:
        # generate numpy data
        (x_train, labels_train), (x_test, labels_test), data_seperate = generate_data(numpyIo, data_load_path, drop_data = True, block_size = 64)
        if save:
            # save numpy data
            numpyIo.save_array(x_train, labels_train, x_test, labels_test, data_seperate)
        return (x_train, labels_train), (x_test, labels_test), data_seperate
    if load:
        # load numpy data
        (x_train, labels_train), (x_test, labels_test), data_seperate = numpyIo.load_array()
        return (x_train, labels_train), (x_test, labels_test), data_seperate

def main():
    FLAGS = parser.parse_args()
    
    image_path = FLAGS.image_path
    an_path = FLAGS.an_path
    vessel_path = FLAGS.vessel_path
    data_load_path = [image_path, an_path, vessel_path]
    data_save_path = FLAGS.data_save_path
    generate = FLAGS.generate
    save = FLAGS.save
    load = FLAGS.load
    which_gpu = FLAGS.which_gpu
    mode = FLAGS.mode
    n_labels = FLAGS.n_labels
    input_shape = FLAGS.input_shape
    pool_size = FLAGS.pool_size
    deconvolution = FLAGS.deconvolution
    depth = FLAGS.depth
    n_base_filters = FLAGS.n_base_filters
    batch_normalization = FLAGS.batch_normalization
    activation_name = FLAGS.activation_name
    model_save_path = FLAGS.model_save_path
    include_label_wise_dice_coefficients = FLAGS.include_label_wise_dice_coefficients
    initial_learning_rate = FLAGS.initial_learning_rate
    metrics = FLAGS.metrics
    loss_fun = FLAGS.loss_fun
    monitor = FLAGS.monitor
    save_best_only = FLAGS.save_best_only
    using_data_augmentation = FLAGS.using_data_augmentation
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    verbose = FLAGS.verbose
    shuffle = FLAGS.shuffle
    
    # load dataset
    (x_train, labels_train), (x_test, labels_test), _ = load_or_save_data(generate, save, load, data_save_path, data_load_path)
    # class initialize
    unet3d_class = Unet3D(which_gpu = which_gpu, 
                   n_labels = n_labels,
                   input_shape = input_shape,
                   pool_size = pool_size, 
                   deconvolution = deconvolution,
                   depth = depth,
                   n_base_filters = n_base_filters,
                   batch_normalization = batch_normalization,
                   activation_name = activation_name,
                   model_save_path = model_save_path)
    # build model structure
    built_model = unet3d_class.build_model()
    # compile model
    compiled_model = unet3d_class.compile_model(built_model = built_model, 
                                  include_label_wise_dice_coefficients = include_label_wise_dice_coefficients, 
                                  initial_learning_rate = initial_learning_rate, 
                                  metrics = metrics, 
                                  loss_fun = loss_fun)
    # train
    if mode == 'train':
        # callbacks
        model_callbacks = unet3d_class.callbacks(monitor = monitor, save_best_only = save_best_only)
        # train model
        unet3d_class.train_model(compiled_model = compiled_model, 
                         model_callbacks = model_callbacks, 
                         x_train = x_train, 
                         labels_train = labels_train, 
                         x_test = x_test, 
                         labels_test = labels_test, 
                         using_data_augmentation = using_data_augmentation, 
                         batch_size = batch_size, 
                         epochs = epochs, 
                         verbose = verbose, 
                         shuffle = shuffle)
    # predict
    elif mode == 'predict':
        # get test img
        test_img = x_test[0]
        test_img = np.expand_dims(test_img, axis = -1)
        test_img = np.expand_dims(test_img, axis = 0) # shape = (1,64,64,64,1)
        # predict
        pred_mask = unet3d_class.predict(compiled_model = compiled_model, test_img = test_img)
#         print(pred_mask.shape)
        
        
if __name__ == '__main__':
    main()
    
    
