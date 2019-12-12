# -*- encoding: utf-8 -*-
''' generate, load, or save data for 3D An segmentation 3DUnet model

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import sys
sys.path.append('/home/maze/USAL/libs/')
import numpy as np
import argparse
from data_io.numpy_io import NumpyIo
from data_io.image_io import ImageIo
from mapper.keras_generator import keras_image_generator
from model.keras_model.unet3D import Unet3D
from utils.data_io_utils import get_file_dir_lists, split_dataset, img_norm
from utils.image_processing_util import cut_around
from evaluation.metrics_keras import *
from keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default = '/data/maze/3D_An_Seg_20190628/01.CVAImage/')
parser.add_argument('--an_path', default = '/data/maze/3D_An_Seg_20190628/03.Aneurysms/')
parser.add_argument('--vessel_path', default = '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/')
parser.add_argument('--data_save_path', default = '/home/maze/USAL/libs/data')
parser.add_argument('--which_gpu', default = '0')
parser.add_argument('--mode', default = 'evaluation')
parser.add_argument('--n_labels', default = 3)
parser.add_argument('--input_shape', default = (64,64,64,1))
parser.add_argument('--pool_size', default = (2,2,2))
parser.add_argument('--deconvolution', default = True)
parser.add_argument('--depth', default = 4)
parser.add_argument('--n_base_filters', default = 32)
parser.add_argument('--batch_normalization', default = True)
parser.add_argument('--activation_name', default = 'softmax')
parser.add_argument('--model_save_path', default = '/home/maze/USAL/libs/model_save_path')
parser.add_argument('--include_label_wise_dice_coefficients', default = True)
parser.add_argument('--initial_learning_rate', default = 1e-5)
parser.add_argument('--metrics', default = weighted_dice_coefficient)
parser.add_argument('--loss_fun', default = weighted_dice_coefficient_loss)
parser.add_argument('--input_optimizer', default = Adam)
parser.add_argument('--monitor', default = 'val_loss')
parser.add_argument('--save_best_only', default = True)
parser.add_argument('--using_data_augmentation', default = False)
parser.add_argument('--input_generator', default = keras_image_generator)
parser.add_argument('--batch_size', default = 8)
parser.add_argument('--epochs', default = 100)
parser.add_argument('--verbose', default = 1)
parser.add_argument('--shuffle', default = True)

drop_data_list = ['/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000105.T0002CVAImage.mha', 
            '/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000105.T0002Aneurysms.mha', 
            '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000105.T0002CVAVesselMask.mha', 
            '/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000208.T0001CVAImage.mha', 
            '/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000208.T0001Aneurysms.mha', 
            '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000208.T0001CVAVesselMask.mha', 
            '/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000288.T0001CVAImage.mha', 
            '/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000288.T0001Aneurysms.mha', 
            '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000288.T0001CVAVesselMask.mha']

def get_small_image_mask_block(img_array, mask_array, vessel_array, size):
    '''
    Args:
        img_array: npy, of shape [length, width, channel]. eg.[512, 512, 510]
        mask_array: npy, same shape with img_array, the label metrix of an. contained value 0 and 1
        vessel_array: npy, same shape with img_array, the label metrix of vessel. contained value 0, 1, and 2.
        size: int, eg.64,128,256..., the wanted block size from the whole image
    Return:
        small_img_array: npy, of shape [an_nums,length, width, channel]. eg.[2, 64, 64, 64].an_nums = int(1,2,3,...)
        small_mask_array: npy, same shape with small_img_array.
    '''
    small_img_list = []
    small_mask_list = []
    # normlization raw image array first
    img_array = img_norm(img_array)
    # nums of An. 1 or 2 or 3
    an_nums = int(np.max(mask_array))
    for i in range(an_nums):
        print('an #: ', str(i+1))
        # all index for one an            
        one_an_index = np.where(mask_array == (i+1))
        one_an_array = np.array(one_an_index)    # tuple to array
        one_an_array = np.transpose(one_an_array)    # transpose to (n, 3)
        for j in range(len(one_an_array)):
            x = one_an_array[j][0]
            y = one_an_array[j][1]
            z = one_an_array[j][2]
            vessel_array[x, y, z] = 2
        # x axis max and min index
        x_max = np.max(one_an_index[0])
        x_min = np.min(one_an_index[0])
        # y axis max and min index
        y_max = np.max(one_an_index[1])
        y_min = np.min(one_an_index[1])
        # z axis max and min index
        z_max = np.max(one_an_index[2])
        z_min = np.min(one_an_index[2])
        # if max dimention >= size. eg.64
        max_an_dimention = max((x_max-x_min), (y_max-y_min), (z_max-z_min))
        # print the over sized an, whose size is > block size
#         if max_an_dimention >= size:
#             print((x_max-x_min), (y_max-y_min), (z_max-z_min))
        # an index center
        x_mean = np.mean(one_an_index[0])
        y_mean = np.mean(one_an_index[1])
        z_mean = np.mean(one_an_index[2])
        one_an_center = [x_mean, y_mean, z_mean]
        center_int = [int(round(j)) for j in one_an_center]
        # get the 64*64*64 mask and image block
        one_small_mask = cut_around(vessel_array, center_int, size)
        one_small_img = cut_around(img_array, center_int, size)
        small_img_list.append(one_small_img)
        small_mask_list.append(one_small_mask)
        
    small_img_array = np.array(small_img_list)
    small_mask_array = np.array(small_mask_list)
    assert(small_img_array.shape == (an_nums, size, size, size))
    assert(small_mask_array.shape == (an_nums, size, size, size))
    
    return small_img_array, small_mask_array

def generate_data(ImageIo, NumpyIo, data_load_path, drop_data_list = drop_data_list, block_size = 64, save = False, save_path = None):
        print('load data from the source dir: ', data_load_path)
        img_dir_list, an_dir_list, vessel_dir_list = get_file_dir_lists(data_load_path, drop_data_list)
        
        (img_train_list, an_train_list, vessel_train_list), (img_val_list, an_val_list, vessel_val_list), (img_test_list, an_test_list,                   vessel_test_list) = split_dataset(img_dir_list, an_dir_list, vessel_dir_list, split_id = 0, val_ratio = 0.1)
        
        x_train = []
        labels_train = []
        x_test = []
        labels_test = []
        x_val = []
        labels_val = []
        data_seperate = []
        for i in range(len(img_train_list)):
            print('processing: ', an_train_list[i])
            # read the whole image, an_mask and vessel as array
            img_array = ImageIo().load(img_train_list[i])
            mask_array = ImageIo().load(an_train_list[i])
            vessel_array = ImageIo().load(vessel_train_list[i])
            # get the small image and mask block array for that image and mask, output shape [n,64,64,64]
            small_img_array, small_mask_array = get_small_image_mask_block(img_array, mask_array, vessel_array, block_size)
            x_train.append(small_img_array)
            labels_train.append(small_mask_array)
            data_seperate.append([an_dir_list[i][-31:], 'train_set'])
        for i in range(len(img_val_list)):
            print('processing: ', an_val_list[i])
            # read the whole image, an_mask and vessel as array
            img_array = ImageIo().load(img_val_list[i])
            mask_array = ImageIo().load(an_val_list[i])
            vessel_array = ImageIo().load(vessel_val_list[i])
            # get the small image and mask block array for that image and mask, output shape [n,64,64,64]
            small_img_array, small_mask_array = get_small_image_mask_block(img_array, mask_array, vessel_array, block_size)
            x_val.append(small_img_array)
            labels_val.append(small_mask_array)
            data_seperate.append([an_dir_list[i][-31:], 'val_set'])
        for i in range(len(img_test_list)):
            print('processing: ', an_test_list[i])
            # read the whole image, an_mask and vessel as array
            img_array = ImageIo().load(img_test_list[i])
            mask_array = ImageIo().load(an_test_list[i])
            vessel_array = ImageIo().load(vessel_test_list[i])
            # get the small image and mask block array for that image and mask, output shape [n,64,64,64]
            small_img_array, small_mask_array = get_small_image_mask_block(img_array, mask_array, vessel_array, block_size)
            x_test.append(small_img_array)
            labels_test.append(small_mask_array)
            data_seperate.append([an_dir_list[i][-31:], 'test_set'])
        # list to array
        x_train = np.concatenate(x_train, axis = 0)
        labels_train = np.concatenate(labels_train, axis = 0)
        x_val = np.concatenate(x_val, axis = 0)
        labels_val = np.concatenate(labels_val, axis = 0)
        x_test = np.concatenate(x_test, axis = 0)
        labels_test = np.concatenate(labels_test, axis = 0)
        data_seperate = np.array(data_seperate)
        
        if save == True:
            numpyIo = NumpyIo()
            numpyIo.save(save_path = save_path, save_name = 'x_train.npy', data = x_train)
            numpyIo.save(save_path = save_path, save_name = 'labels_train.npy', data = labels_train)
            numpyIo.save(save_path = save_path, save_name = 'x_test.npy', data = x_test)
            numpyIo.save(save_path = save_path, save_name = 'labels_test.npy', data = labels_test)
            numpyIo.save(save_path = save_path, save_name = 'x_val.npy', data = x_val)
            numpyIo.save(save_path = save_path, save_name = 'labels_val.npy', data = labels_val)
            numpyIo.save(save_path = save_path, save_name = 'data_seperate.npy', data = data_seperate)
        
        return (x_train, labels_train), (x_test, labels_test), (x_val, labels_val), data_seperate

def main():
    FLAGS = parser.parse_args()
    
    image_path = FLAGS.image_path
    an_path = FLAGS.an_path
    vessel_path = FLAGS.vessel_path
    data_load_path = [image_path, an_path, vessel_path]
    data_save_path = FLAGS.data_save_path
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
    input_optimizer = FLAGS.input_optimizer
    monitor = FLAGS.monitor
    save_best_only = FLAGS.save_best_only
    using_data_augmentation = FLAGS.using_data_augmentation
    input_generator = FLAGS.input_generator
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    verbose = FLAGS.verbose
    shuffle = FLAGS.shuffle
    
    # load dataset
    numpyIo = NumpyIo()
    x_train = numpyIo.load(file_path = data_save_path, file_name = 'x_train.npy')
    labels_train = numpyIo.load(file_path = data_save_path, file_name = 'labels_train.npy')
    x_test = numpyIo.load(file_path = data_save_path, file_name = 'x_val.npy')
    labels_test = numpyIo.load(file_path = data_save_path, file_name = 'labels_val.npy')
    # class initialize
    unet3d_class = Unet3D(which_gpu = which_gpu, 
                   n_labels = n_labels,
                   input_shape = input_shape,
                   pool_size = pool_size, 
                   deconvolution = deconvolution,
                   depth = depth,
                   n_base_filters = n_base_filters,
                   batch_normalization = batch_normalization,
                   activation_name = activation_name)
    # build model structure
    unet3d_class.build_model()
    # compile model
    unet3d_class.compile_model(include_label_wise_dice_coefficients = include_label_wise_dice_coefficients, 
                      initial_learning_rate = initial_learning_rate, 
                      metrics = metrics, 
                      loss_fun = loss_fun,
                      input_optimizer = input_optimizer)
    # train
    if mode == 'train':
        # callbacks
        model_callbacks = unet3d_class.callbacks(model_save_path = model_save_path, monitor = monitor, save_best_only = save_best_only)
        # train model
        unet3d_class.train_model(model_callbacks = model_callbacks, 
                         x_train = x_train, 
                         labels_train = labels_train, 
                         x_test = x_test, 
                         labels_test = labels_test, 
                         using_data_augmentation = using_data_augmentation, 
                         input_generator = input_generator, 
                         batch_size = batch_size, 
                         epochs = epochs, 
                         verbose = verbose, 
                         shuffle = shuffle)
    # predict
    elif mode == 'predict':
        unet3d_class.load_model(model_save_path = model_save_path)
        # get test img
        test_img = x_test[0]
        test_img = np.expand_dims(test_img, axis = -1)
        test_img = np.expand_dims(test_img, axis = 0) # shape = (1,64,64,64,1)
        # predict
        pred_mask = unet3d_class.predict(test_img = test_img)
#         print(pred_mask.shape)

    elif mode == 'evaluation':
        unet3d_class.load_model(model_save_path)
        unet3d_class.evaluate_model(x_test = x_test, labels_test = labels_test)
        
        
if __name__ == '__main__':
    main()
    
    