# -*- encoding: utf-8 -*-
''' unet model base on KerasBaseModel in model.py

Author: Ze Ma
Date: November 2019

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2019 All Right Reserved
'''
import os
import time
import numpy as np
import math
from evaluation.metrics_keras import get_label_dice_coefficient_function
from mapper.keras_generator import image_generator_2Dto3D
from keras import backend as K
from keras.utils import np_utils
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from .model3D import Model3D

class Unet3D(Model3D):
    """ Unet model
    """
    def __init__(self, 
             which_gpu, 
             n_labels, 
             input_shape, 
             pool_size, 
             deconvolution, 
             depth, 
             n_base_filters, 
             batch_normalization, 
             activation_name):
        self.which_gpu = which_gpu
        self.n_labels = n_labels
        self.input_shape = input_shape 
        self.pool_size = pool_size 
        self.deconvolution = deconvolution 
        self.depth = depth
        self.n_base_filters = n_base_filters 
        self.batch_normalization = batch_normalization 
        self.activation_name = activation_name
        
        self.model = None
        
        
    def build(self):
        n_labels = self.n_labels
        input_shape = self.input_shape 
        pool_size = self.pool_size 
        deconvolution = self.deconvolution 
        depth = self.depth 
        n_base_filters = self.n_base_filters 
        batch_normalization = self.batch_normalization 
        activation_name = self.activation_name
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = Model3D.convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                              batch_normalization=batch_normalization)
            layer2 = Model3D.convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                              batch_normalization=batch_normalization)
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth-2, -1, -1):
            up_convolution = Model3D.get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                    n_filters=current_layer._keras_shape[1])(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis= -1)
            current_layer = Model3D.convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                   input_layer=concat, batch_normalization=batch_normalization)
            current_layer = Model3D.convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                   input_layer=current_layer, batch_normalization=batch_normalization)

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
        
        self.model = model
        
        
    def compile_model(self, 
                include_label_wise_dice_coefficients, 
                initial_learning_rate, 
                metrics, 
                loss_fun,
                input_optimizer):
        model = self.model
        n_labels = self.n_labels
        
        if not isinstance(metrics, list):
            metrics = [metrics]

        if include_label_wise_dice_coefficients and n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
            if metrics:
                metrics = metrics + label_wise_dice_metrics
            else:
                metrics = label_wise_dice_metrics
        input_optimizer = input_optimizer(lr = initial_learning_rate)
        model.compile(optimizer=input_optimizer, loss=loss_fun, metrics=metrics)
        model.summary()
        
        self.model = model

    
    def callbacks(self, path, monitor, save_best_only):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        model_checkpoint = ModelCheckpoint(path, monitor=monitor, 
                                save_best_only=save_best_only, save_weights_only = True, verbose = 1)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-7, verbose = 1)
        return [model_checkpoint, lr_reducer]
    
    def train(self, 
               model_callbacks, 
               x_train, 
               labels_train, 
               x_test, 
               labels_test, 
               batch_size, 
               epochs, 
               verbose, 
               shuffle, 
               class_weight, 
               sample_weight):
        # select training gpu
        which_gpu = self.which_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
        n_labels = self.n_labels
        model = self.model
        callbacks = model_callbacks
        # 转化为3分类标签
        x_test = np.expand_dims(x_test, axis = -1)
        labels_test = np_utils.to_categorical(labels_test, n_labels)
        x_train = np.expand_dims(x_train, axis = -1)
        labels_train = np_utils.to_categorical(labels_train, n_labels)
        print('Not using data augmentation.')
        model.fit(x_train, labels_train, batch_size = batch_size, epochs = epochs, verbose = verbose, shuffle = shuffle, 
               class_weight = class_weight, sample_weight = sample_weight, validation_data = [x_test, labels_test], callbacks = callbacks)
            
    def train_generator(self, 
                  model_callbacks, 
                  x_train, 
                  labels_train, 
                  x_test, 
                  labels_test, 
                  input_generator,
                  batch_size, 
                  epochs, 
                  verbose, 
                  shuffle):
        # select training gpu
        which_gpu = self.which_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
        n_labels = self.n_labels
        model = self.model
        callbacks = model_callbacks
        x_test = np.expand_dims(x_test, axis = -1)
        labels_test = np_utils.to_categorical(labels_test, n_labels)
        print('Using real-time data augmentation.')
        datagen = input_generator()
        gen = image_generator_2Dto3D(data = x_train, label = labels_train, datagen = datagen, 
                            class_num = n_labels, batch_size = batch_size, shuffle=True)
        steps_per_epoch = math.ceil(len(x_train)/batch_size)  # define how many batches for one epoch
        model.fit_generator(gen, epochs = epochs, steps_per_epoch = steps_per_epoch, 
                     verbose = verbose, validation_data = [x_test, labels_test], callbacks = callbacks)
            

    def load(self, path):
        # select training gpu
        which_gpu = self.which_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
        model = self.model
        model.load_weights(path)
        print('loading model...done.')
        self.model = model
    
    def predict(self, test_img):
        t0 = time.time()
        model = self.model
        print('predicting...')
        pred_mask = model.predict(test_img, batch_size = 1)
        pred_mask = np.squeeze(pred_mask)
        pred_mask = np.argmax(pred_mask, axis = -1)
        print('Done! predict time: ', str(time.time() - t0))
        return pred_mask
        
    def evaluate(self, x_test, labels_test, batch_size, sample_weight):
        model = self.model
        n_labels = self.n_labels
        x_test = np.expand_dims(x_test, axis = -1)
        labels_test = np_utils.to_categorical(labels_test, n_labels)
        scores = model.evaluate(x_test, labels_test, batch_size = batch_size, sample_weight = sample_weight, verbose = 1) # may OOM
        print('evaluation result: ', str(scores))  
        
