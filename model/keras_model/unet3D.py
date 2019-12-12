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
from evaluation.metrics_keras import get_label_dice_coefficient_function
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
        """
        Builds the 3D UNet Keras model.f
        :param metrics: List metrics to be calculated during model training (default is dice coefficient).
        :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
        coefficient for each label as metric.
        :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
        layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
        to train the model.
        :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
        layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
        divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
        increases the amount memory required during training.
        :return: Untrained 3D UNet Model
        """
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
        
        
    def build_model(self):
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

    
    def callbacks(self, model_save_path, monitor, save_best_only):
        if os.path.exists(model_save_path) == False:
            os.mkdir(model_save_path)
        model_checkpoint = ModelCheckpoint(os.path.join(model_save_path, '3dUnet_64.hdf5'), monitor=monitor, 
                                save_best_only=save_best_only, save_weights_only = True, verbose = 1)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-7, verbose = 1)
        return [model_checkpoint, lr_reducer]
    
    def train_model(self, 
               model_callbacks, 
               x_train, 
               labels_train, 
               x_test, 
               labels_test, 
               using_data_augmentation, 
               input_generator,
               batch_size, 
               epochs, 
               verbose, 
               shuffle):
        # select training gpu
        which_gpu = self.which_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
        model = self.model
        callbacks = model_callbacks
        if using_data_augmentation == False:
            x_train = np.expand_dims(x_train, axis = -1)
            x_test = np.expand_dims(x_test, axis = -1)
            # 转化为3分类标签
            labels_train = np_utils.to_categorical(labels_train, 3)
            labels_test = np_utils.to_categorical(labels_test, 3)
            print('Not using data augmentation.')
            model.fit(x_train, labels_train, batch_size = batch_size, epochs = epochs, verbose = verbose, shuffle = shuffle, 
                   validation_data = [x_test, labels_test], callbacks = callbacks)
        else:
            print('Using real-time data augmentation.')
            print('Generating dataset......')
            datagen = input_generator
            
            batches = 0
            x_batch_array = []
            labels_batch_array = []
            for x_batch, labels_batch in datagen.flow(x_train, labels_train, batch_size = batch_size, shuffle = True):
                x_batch_array.append(x_batch)
                labels_batch_array.append(labels_batch)
                batches += 1
                if batches == 50:
                    break
            x_train = np.concatenate(x_batch_array, axis = 0)
            labels_train = np.concatenate(labels_batch_array, axis = 0)

            x_train = np.expand_dims(x_train, axis = -1)
            x_test = np.expand_dims(x_test, axis = -1)
            # 转化为3分类标签
            labels_train = np_utils.to_categorical(labels_train, 3)
            labels_test = np_utils.to_categorical(labels_test, 3)
            
            model.fit(x_train, labels_train, batch_size = batch_size, epochs = epochs, verbose = verbose, shuffle = shuffle, 
                   validation_data = [x_test, labels_test], callbacks = callbacks)

    def load_model(self, model_save_path):
        # select training gpu
        which_gpu = self.which_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
        model = self.model
        print('loading model...done.')
        model.load_weights(os.path.join(model_save_path, '3dUnet_64.hdf5'))
        self.model = model
    
    def evaluate_model(self, x_test, labels_test):
        model = self.model
        x_test = np.expand_dims(x_test, axis = -1)
        labels_test = np_utils.to_categorical(labels_test, 3)
        scores = model.evaluate(x_test, labels_test, batch_size = 8, verbose = 1) # may OOM
        print('test loss: ', scores[0])
        print('test accuracy: ', scores[1])
    
    def predict(self, test_img):
        t0 = time.time()
        model = self.model
        print('predicting...')
        pred_mask = model.predict(test_img, batch_size = 1)
        pred_mask = np.squeeze(pred_mask)
        pred_mask = np.argmax(pred_mask, axis = -1)
        print('Done! predict time: ', str(time.time() - t0))
        return pred_mask
        
        
        
