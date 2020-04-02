import numpy as np
import os
import random


def get_file_dir_lists(load_path, drop_list):
    """ read all file names from the path, and drop the bad sample
        Args:
            load_path: str, the file saving path that contains all the samples
            drop_list: list of str, the no-useful samples
        Return:
            img_dir_list: list of str, contains all the filename path for each useful sample
    """
    image_dir = os.listdir(load_path)
    img_dir_list = [(load_path + i) for i in sorted(image_dir)]
            
    # drop data that cant be used
    for name in drop_list:
        img_dir_list = list(filter(lambda x: (name not in x), img_dir_list))
        print(name, 'dropped!')
    
    return img_dir_list


def split_dataset(img_list, split_id, test_id, val_id):
    """ seperate dataset
        Args:
            img_list: list of str, contains all the filename path for each useful sample
            split_id: int, eg.-27, the key index in a str that is used to seperate dataset
            test_id: list of int, eg. [0], [0, 1]. if sample_name[split_id] in test_id, that sample will be in the test set
            val_id: list of int, eg. [0], [0, 1]. if sample_name[split_id] in val_id, that sample will be in the val set
        Return:
            Three lists for train,val and test dataset. Each list contains sample filenames
    """
    img_train_list = []
    img_test_list = []
    img_val_list = []
    for i in range(len(img_list)):
        if int(img_list[i][split_id]) in test_id:   # test set
            img_test_list.append(img_list[i])
        elif int(img_list[i][split_id]) in val_id:   # val set
            img_val_list.append(img_list[i])
        else:    # train set
            img_train_list.append(img_list[i])
            
    return img_train_list, img_val_list, img_test_list


