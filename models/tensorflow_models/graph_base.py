# -*- encoding: utf-8 -*-
''' graphnet model for tensorflow

Author: Ze Ma
Date: Feb 2020

(C) Copyright: Unionstrong (Beijing) Technology Co., Ltd
2016-2020 All Right Reserved
'''

import logging
logging.basicConfig(level = logging.INFO)
from .model import *
from evaluation.metrics_tf import dice_coefficient
import tensorflow as tf
import time
import math
from utils.tf_utils import shuffle_mini_batches


class GraphBase(TfModelBase):
    """ graphnet model based on TfModelBase in model.py
    """
    def __init__(self):
        pass
    
    def build(self):
        pass
    
    def _get_placeholders(self, batch_size, num_point, num_coordinate, nums_info, with_adj, with_sample_weight):
        x = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_coordinate))
        labels_seg = tf.placeholder(tf.int32, shape=(batch_size, num_point))
        labels_cls = tf.placeholder(tf.int32, shape=(batch_size))
        info = tf.placeholder(tf.float32, shape=(batch_size, nums_info))
        training_flag = tf.placeholder(tf.bool, shape=())
        if with_adj == True:
            adjs = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_point))
        else: adjs = False
        if with_sample_weight == None: sample_weight = None
        else: sample_weight = tf.placeholder(tf.float32, shape=(batch_size))

        return x, labels_seg, labels_cls, info, training_flag, adjs, sample_weight
    
 
    def compile(self, labels_seg, labels_cls, input_optimizer, batch, batch_size, base_learning_rate, momentum, sample_weight = None):
        loss_seg_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_seg, labels=labels_seg)
        loss_cls_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_cls, labels=labels_cls)
        if sample_weight != None:
            loss_seg_batch = tf.multiply(sample_weight,loss_seg_batch)
            loss_cls_batch = tf.multiply(sample_weight,loss_cls_batch)
        loss_seg = tf.reduce_mean(loss_seg_batch)
        loss_cls = tf.reduce_mean(loss_cls_batch)
        total_loss = loss_seg + loss_cls
            
        # Get training operator
        learning_rate = super()._get_learning_rate(batch, base_learning_rate, batch_size)
        if input_optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
        elif input_optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=batch)
        dice = dice_coefficient(self.pred_seg, labels_seg)
        
#         self.loss_seg = loss_seg
#         self.loss_cls = loss_cls
#         self.total_loss = total_loss
#         self.train_op = train_op
#         self.dice = dice
        return loss_seg, loss_cls, total_loss, train_op, dice
        
    def train(self,
           sess,
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
           shuffle = True, 
           with_aug = False):
        saver_folder_name = self.saver_folder_name
        temp_best_acc = 0
        
        init = tf.global_variables_initializer()
        sess.run(init)
        logging.info('----------------start training-----------------')
        t0 = time.time()
        for epoch in range(epochs):
            total_correct_cls = 0
            total_correct_seg = 0
            total_dice = 0
            total_seen_graph = 0
            total_seen_points = 0
            loss_sum_seg = 0
            loss_sum_cls = 0
            loss_sum = 0

            num_minibatches = math.ceil(num_train_sample/batch_size)
            # random shuffle the whole train data set
            shuffled_minibatch_list = shuffle_mini_batches(x_train, labels_train_cls, labels_train_seg, info_train, batch_size, adj_train)
            for i in range(len(shuffled_minibatch_list)):
                one_batch = shuffled_minibatch_list[i]
                minibatch_x, minibatch_y_cls, minibatch_y_seg, minibatch_info, minibatch_adj= one_batch
                if with_aug == True:
                    minibatch_x, minibatch_y_cls = next(input_generator.flow(minibatch_x, minibatch_y_cls, batch_size))
                    minibatch_y_seg, _ = next(input_generator.flow(minibatch_y_seg, minibatch_y_cls, batch_size))
                feed_dict = {x: minibatch_x, labels_cls: minibatch_y_cls, labels_seg: minibatch_y_seg, 
                         info: minibatch_info, adjs: minibatch_adj, training_flag: True}

                _, minibatch_loss, minibatch_loss_cls, minibatch_loss_seg, minibatch_pred_cls, minibatch_pred_seg, minibatch_dice =                          sess.run([self.train_op, self.total_loss, self.loss_cls, self.loss_seg, self.pred_cls, self.pred_seg, self.dice], feed_dict = feed_dict)

                pred_cls_output = np.argmax(minibatch_pred_cls, 1)
                correct_cls = np.sum(pred_cls_output == minibatch_y_cls)
                total_correct_cls += correct_cls
                total_seen_graph += batch_size
                pred_seg_output = np.argmax(minibatch_pred_seg, 2)
                correct_seg = np.sum(pred_seg_output == minibatch_y_seg)
                total_correct_seg += correct_seg
                total_seen_points += (batch_size * num_point)
                total_dice += minibatch_dice
                loss_sum_cls += minibatch_loss_cls
                loss_sum_seg += minibatch_loss_seg
                loss_sum += minibatch_loss

            if epoch % 1 == 0:
                logging.info ("total loss after epoch %i: %f" % (epoch, (loss_sum/float(num_minibatches))))
                logging.info ("seg loss after epoch %i: %f" % (epoch, (loss_sum_seg/float(num_minibatches))))
                logging.info ("cls loss after epoch %i: %f" % (epoch, (loss_sum_cls/float(num_minibatches))))
                logging.info ("train_accuracy for seg after epoch %i: %f" % (epoch, (total_correct_seg/float(total_seen_points))))
                logging.info ("train_mean_dice after epoch %i: %f" % (epoch, (total_dice/float(num_minibatches))))
                logging.info ("train_accuracy for cls after epoch %i: %f" % (epoch, (total_correct_cls/float(total_seen_graph))))
                logging.info ("-----train end------")
            # start evaluate validation set
            total_correct_cls = 0
            total_correct_seg = 0
            total_dice = 0
            total_seen_graph = 0
            total_seen_points = 0
            loss_sum_seg = 0
            loss_sum_cls = 0
            loss_sum = 0

            num_minibatches_test = math.ceil(num_val_sample/batch_size) # number of minibatches in the validation set
            # random shuffle the whole train data set
            shuffled_minibatch_list = shuffle_mini_batches(x_val, labels_val_cls, labels_val_seg, info_val, batch_size, adj_val)
            for i in range(len(shuffled_minibatch_list)):
                one_batch = shuffled_minibatch_list[i]
                minibatch_x, minibatch_y_cls, minibatch_y_seg, minibatch_info, minibatch_adj = one_batch
                feed_dict = {x: minibatch_x, labels_cls: minibatch_y_cls, labels_seg: minibatch_y_seg,
                         info: minibatch_info, adjs: minibatch_adj, training_flag: False}

                minibatch_loss, minibatch_loss_cls, minibatch_loss_seg, minibatch_pred_cls, minibatch_pred_seg, minibatch_dice = sess.run(
                    [self.total_loss, self.loss_cls, self.loss_seg, self.pred_cls, self.pred_seg, self.dice], feed_dict = feed_dict)

                pred_cls_output = np.argmax(minibatch_pred_cls, 1)
                correct_cls = np.sum(pred_cls_output == minibatch_y_cls)
                total_correct_cls += correct_cls
                total_seen_graph += batch_size
                pred_seg_output = np.argmax(minibatch_pred_seg, 2)
                correct_seg = np.sum(pred_seg_output == minibatch_y_seg)
                total_correct_seg += correct_seg
                total_seen_points += (batch_size * num_point)
                total_dice += minibatch_dice
                loss_sum_cls += minibatch_loss_cls
                loss_sum_seg += minibatch_loss_seg
                loss_sum += minibatch_loss

            epoch_test_accuracy = total_correct_cls/float(total_seen_graph)
            if epoch % 1 == 0:
                logging.info ("validation total loss after epoch %i: %f" % (epoch, (loss_sum/float(num_minibatches_test))))
                logging.info ("validation seg loss after epoch %i: %f" % (epoch, (loss_sum_seg/float(num_minibatches_test))))
                logging.info ("validation cls loss after epoch %i: %f" % (epoch, (loss_sum_cls/float(num_minibatches_test))))
                logging.info ("val_accuracy for seg after epoch %i: %f" % (epoch, (total_correct_seg/float(total_seen_points))))
                logging.info ("validation_mean_dice after epoch %i: %f" % (epoch, (total_dice/float(num_minibatches_test))))
                logging.info ("val_accuracy for cls after epoch %i: %f" % (epoch, (total_correct_cls/float(total_seen_graph))))
                logging.info('---------------------------------------------------')

            # save the best model only
            if epoch_test_accuracy >= temp_best_acc:
                saver.save(sess = sess, save_path = saver_folder_name + '/GraphNet_weights.ckpt', global_step = epoch)
                temp_best_acc = epoch_test_accuracy
                    
        logging.info('----------------total training time is: ', str(time.time()- t0))
        logging.info("Finished!")
    
    def load(self, sess, saver):
        ckpt = tf.train.get_checkpoint_state(self.saver_folder_name)
        saver.restore(sess, ckpt.model_checkpoint_path)
        self.sess = sess
        logging.info('------------------------model loaded----------------------------')
    
    def predict_classes(self, logits, x, info, adjs, training_flag, x_test, info_test, adj_test):
        sess = self.sess
        
        logging.info('----------------start prediction-------------------')
        feed_dict = {x: x_test, info: info_test, adjs: adj_test, training_flag: False}
        samples_pred = sess.run(logits, feed_dict = feed_dict)
        assert(samples_pred.shape == (x_test.shape[0], 2))
        pred_test = np.argmax(samples_pred, 1)
        assert(pred_test.shape == (x_test.shape[0], ))
            
        return pred_test

    def evaluate(self, logits, x, info, adjs, training_flag, x_val, labels_val_cls, info_val, adj_val):
        sess = self.sess
        # input data
        y_val = labels_val_cls
        
        logging.info('----------------start evaluation-------------------')
        feed_dict = {x: x_val, info: info_val, adjs: adj_val, training_flag: False}
        samples_pred = sess.run(logits, feed_dict = feed_dict)
        assert(samples_pred.shape == (x_val.shape[0], 2))
        pred_val = np.argmax(samples_pred, 1)
        assert(pred_val.shape == (x_val.shape[0], ))
        total_correct = np.sum(pred_val == y_val)

        accuracy = total_correct/ len(y_val)
        logging.info('mean accuracy is: ', str(accuracy))
           