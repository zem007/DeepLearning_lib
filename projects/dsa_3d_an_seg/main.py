# -*- encoding: utf-8 -*-
''' generate, load, or save data for 3D An segmentation 3DUnet model

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import sys
sys.path.append('/home/maze/USAL/libs/')
import os
import numpy as np
import math
import argparse
from data_io.numpy_io import NumpyIo
from data_io.image_io import ImageIo
from mappers.generator_keras import keras_image_generator_2D, image_generator_2Dto3D
from evaluation.metrics_keras import get_label_dice_coefficient_function
from models.keras_models.unet3D import Unet3D
from utils.data_io_utils import get_file_dir_lists, split_dataset
from utils.image_processing_utils import cut_around_3D, get_label_mean, normalize_image
from evaluation.metrics_keras import *
from evaluation.losses_keras import *
from keras.optimizers import Adam
from keras.utils import np_utils

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default = '/data/maze/04.AN-AI/01.CVAImage/')
parser.add_argument('--an_path', default = '/data/maze/04.AN-AI/03.Aneurysms/')
parser.add_argument('--vessel_path', default = '/data/maze/04.AN-AI/02.CVAVesselMask/')
parser.add_argument('--data_save_path', default = '/home/maze/USAL/libs/new_data')
parser.add_argument('--which_gpu', default = '0')
parser.add_argument('--mode', default = 'predict')
parser.add_argument('--class_weight', default = None)
parser.add_argument('--sample_weight', default = None)
parser.add_argument('--n_outputs', default = 3)
parser.add_argument('--input_shape', default = (64,64,64,1))
parser.add_argument('--pool_size', default = (2,2,2))
parser.add_argument('--deconvolution', default = True)
parser.add_argument('--depth', default = 4)
parser.add_argument('--n_base_filters', default = 32)
parser.add_argument('--batch_normalization', default = True)
parser.add_argument('--activation_name', default = 'softmax')
parser.add_argument('--initial_learning_rate', default = 1e-5)
parser.add_argument('--metrics', default = weighted_dice_coefficient)
parser.add_argument('--loss_fun', default = weighted_dice_coefficient_loss)
parser.add_argument('--input_optimizer', default = Adam)
parser.add_argument('--callbacks', default = True)
parser.add_argument('--model_save_path', default = '/home/maze/USAL/libs/model_save_path')
parser.add_argument('--monitor', default = 'val_loss')
parser.add_argument('--save_best_only', default = True)
parser.add_argument('--input_generator', default = keras_image_generator_2D)
parser.add_argument('--aug_planes', default = ['xy', 'yz', 'xz'])
parser.add_argument('--batch_size', default = 8)
parser.add_argument('--epochs', default = 100)
parser.add_argument('--verbose', default = 1)
parser.add_argument('--shuffle', default = True)

drop_list = ['A002.P000208', 'A003.P000143']
# drop_img_list = ['/data/maze/04.AN-AI/01.CVAImage/A002.P000208.D01.S004CVAImage.mha', 
#             '/data/maze/04.AN-AI/01.CVAImage/A003.P000143.D01.S001CVAImage.mha']
# drop_an_list = ['/data/maze/04.AN-AI/03.Aneurysms/A002.P000208.D01.S004Aneurysms.mha', 
#            '/data/maze/04.AN-AI/03.Aneurysms/A003.P000143.D01.S001Aneurysms.mha']
# drop_vessel_list = ['/data/maze/04.AN-AI/02.CVAVesselMask/A002.P000208.D01.S004CVAVesselMask.mha', 
#               '/data/maze/04.AN-AI/02.CVAVesselMask/A003.P000143.D01.S001CVAVesselMask.mha']

# drop_img_list = ['/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000105.T0002CVAImage.mha', 
#             '/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000208.T0001CVAImage.mha', 
#             '/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000288.T0001CVAImage.mha']
# drop_an_list = ['/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000105.T0002Aneurysms.mha', 
#            '/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000208.T0001Aneurysms.mha', 
#            '/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000288.T0001Aneurysms.mha']
# drop_vessel_list = ['/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000105.T0002CVAVesselMask.mha', 
#               '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000208.T0001CVAVesselMask.mha', 
#               '/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000288.T0001CVAVesselMask.mha']


def get_small_image_mask_block(x_image, mask_image, vessel_image, size):
    small_img_list = []
    small_mask_list = []
    # normlization raw image array first
    img_array = normalize_image(x_image)
    mean_dict = get_label_mean(mask_image)
    modified_image = np.copy(vessel_image)
    modified_image[mask_image > 0] = 2
    for label, center in mean_dict.items():
        # get the 64*64*64 mask and image block
        one_small_mask = cut_around_3D(modified_image, center, size)
        one_small_img = cut_around_3D(x_image, center, size)
        small_img_list.append(one_small_img)
        small_mask_list.append(one_small_mask)
    small_img_array = np.array(small_img_list)
    small_mask_array = np.array(small_mask_list)
    
    return small_img_array, small_mask_array

def generate_data(ImageIo, NumpyIo, image_path, an_path, vessel_path, drop_list, block_size = 64, save = False, save_path = None):
        print('load img from the source dir: ', image_path)
        img_dir_list = get_file_dir_lists(image_path, drop_list)
        print('load an from the source dir: ', an_path)
        an_dir_list = get_file_dir_lists(an_path, drop_list)
        print('load vessel from the source dir: ', vessel_path)
        vessel_dir_list = get_file_dir_lists(vessel_path, drop_list)
        
        img_train_list, img_val_list, img_test_list = split_dataset(img_list = img_dir_list, split_id = -22, test_id=[0], val_id=[1])
        an_train_list, an_val_list, an_test_list = split_dataset(img_list = an_dir_list, split_id = -23, test_id=[0], val_id=[1])
        vessel_train_list, vessel_val_list, vessel_test_list = split_dataset(img_list = vessel_dir_list, split_id = -27, test_id=[0], val_id=[1])
        
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
            numpyIo.save(filename = os.path.join(save_path, 'x_train.npy'), data = x_train)
            numpyIo.save(filename = os.path.join(save_path, 'labels_train.npy'), data = labels_train)
            numpyIo.save(filename = os.path.join(save_path, 'x_test.npy'), data = x_test)
            numpyIo.save(filename = os.path.join(save_path, 'labels_test.npy'), data = labels_test)
            numpyIo.save(filename = os.path.join(save_path, 'x_val.npy'), data = x_val)
            numpyIo.save(filename = os.path.join(save_path, 'labels_val.npy'), data = labels_val)
            numpyIo.save(filename = os.path.join(save_path, 'data_seperate.npy'), data = data_seperate)
        
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
    class_weight = FLAGS.class_weight
    sample_weight = FLAGS.sample_weight
    n_outputs = FLAGS.n_outputs
    input_shape = FLAGS.input_shape
    pool_size = FLAGS.pool_size
    deconvolution = FLAGS.deconvolution
    depth = FLAGS.depth
    n_base_filters = FLAGS.n_base_filters
    batch_normalization = FLAGS.batch_normalization
    activation_name = FLAGS.activation_name
    model_save_path = FLAGS.model_save_path
    initial_learning_rate = FLAGS.initial_learning_rate
    metrics = FLAGS.metrics
    loss_fun = FLAGS.loss_fun
    input_optimizer = FLAGS.input_optimizer
    callbacks = FLAGS.callbacks
    monitor = FLAGS.monitor
    save_best_only = FLAGS.save_best_only
    input_generator = FLAGS.input_generator
    aug_planes = FLAGS.aug_planes
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    verbose = FLAGS.verbose
    shuffle = FLAGS.shuffle
    
    # load dataset
    numpyIo = NumpyIo()
    x_train = numpyIo.load(filename = os.path.join(data_save_path, 'x_train.npy'))
    labels_train = numpyIo.load(filename = os.path.join(data_save_path, 'labels_train.npy'))
    x_val = numpyIo.load(filename = os.path.join(data_save_path, 'x_val.npy'))
    labels_val = numpyIo.load(filename = os.path.join(data_save_path, 'labels_val.npy'))

    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)
    # class initialize
    unet3d_class = Unet3D(which_gpu = which_gpu, 
                   n_outputs = n_outputs,
                   input_shape = input_shape,
                   pool_size = pool_size, 
                   deconvolution = deconvolution,
                   depth = depth,
                   n_base_filters = n_base_filters,
                   batch_normalization = batch_normalization,
                   activation_name = activation_name)
    # build model structure
    unet3d_class.build()
    # compile model
    metrics = [metrics]
    if n_outputs > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_outputs)]
        metrics = metrics + label_wise_dice_metrics
    input_optimizer = input_optimizer(lr = initial_learning_rate) 
    unet3d_class.compile(input_optimizer = input_optimizer, 
                  loss_fun = loss_fun, 
                  metrics = metrics, 
                  initial_learning_rate = initial_learning_rate)
  
    # train
    if mode == 'train':
        # callbacks
        if callbacks == True:
            callbacks = unet3d_class.callbacks(path = os.path.join(model_save_path, '3dUnet_64.hdf5'), monitor = monitor, save_best_only =                                          save_best_only)
        else: callbacks = None
        # train model
        labels_train = np_utils.to_categorical(labels_train, n_outputs)
        labels_val = np_utils.to_categorical(labels_val, n_outputs)
        unet3d_class.train(x_train = x_train, 
                         labels_train = labels_train, 
                         x_val = x_val, 
                         labels_val = labels_val, 
                         batch_size = batch_size, 
                         epochs = epochs, 
                         verbose = verbose, 
                         callbacks = callbacks,  
                         shuffle = shuffle, 
                         class_weight = class_weight, 
                         sample_weight = sample_weight)
    # train
    if mode == 'train_generator':
        # callbacks
        if callbacks == True:
            callbacks = unet3d_class.callbacks(path = os.path.join(model_save_path, '3dUnet_64.hdf5'), monitor = monitor, save_best_only =                                          save_best_only)
        else: callbacks = None
        # input genertor
        datagen = input_generator()
        gen = image_generator_2Dto3D(data = x_train, label = labels_train, input_generator = datagen, 
                            class_num = n_outputs, aug_planes = aug_planes, batch_size = batch_size, shuffle=shuffle)
        steps_per_epoch = math.ceil(len(x_train)/batch_size)  # define how many batches for one epoch
        labels_val = np_utils.to_categorical(labels_val, n_outputs)
        # train model
        unet3d_class.train_generator(gen = gen, 
                         steps_per_epoch = steps_per_epoch, 
                         epochs = epochs, 
                         verbose = verbose, 
                         x_val = x_val, 
                         labels_val = labels_val, 
                         callbacks = callbacks)

    # predict
    elif mode == 'predict':
        unet3d_class.load(path = os.path.join(model_save_path, '3dUnet_64.hdf5'))
        # get test img
        pred_img = x_val[0]
        pred_img = np.expand_dims(pred_img, axis = 0) # shape = (1,64,64,64,1)
        # predict
        unet3d_class.predict(pred_img = pred_img)
        pred_mask = unet3d_class.predict_classes()
#         print(pred_mask.shape)

    elif mode == 'evaluate':
        unet3d_class.load(path = os.path.join(model_save_path, '3dUnet_64.hdf5'))
        unet3d_class.evaluate(x_val = x_val, labels_val = labels_val, batch_size = batch_size, sample_weight = sample_weight)
        
        
if __name__ == '__main__':
    main()
    
    