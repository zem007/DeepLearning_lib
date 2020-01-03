import numpy as np
import os
import random


def get_file_dir_lists(load_path, drop_list):
    image_dir = os.listdir(load_path)
    img_dir_list = [(load_path + i) for i in sorted(image_dir)]
            
    # drop data that cant be used
    for name in drop_list:
        img_dir_list = list(filter(lambda x: (name not in x), img_dir_list))
        print(name, 'dropped!')
    
    return img_dir_list


def split_dataset(img_list, split_id, test_id, val_id):
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


