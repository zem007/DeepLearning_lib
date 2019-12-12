import numpy as np
import os
import random

def get_file_dir_lists(load_path, drop_data_list):
    image_path = load_path[0]
    an_path = load_path[1]
    vessel_path = load_path[2]
    image_dir = os.listdir(image_path)
    an_dir = os.listdir(an_path)
    vessel_dir = os.listdir(vessel_path)
    
    img_dir_list = [(image_path + i) for i in sorted(image_dir)]
    an_dir_list = [(an_path + i) for i in sorted(an_dir)]
    vessel_dir_list = [(vessel_path + i) for i in sorted(vessel_dir)]
    # check the file names are matched
    for i in range(len(img_dir_list)):
        if img_dir_list[i][-30:-12] != an_dir_list[i][-31:-13] or img_dir_list[i][-30:-12] != vessel_dir_list[i][-35:-17]:
            print(i+ 'th image is not matched')
            
    # 舍弃有问题的数据
    for data_name in drop_data_list:
        if data_name in img_dir_list:
            img_dir_list.remove(data_name)
        elif data_name in an_dir_list:
            an_dir_list.remove(data_name)
        elif data_name in vessel_dir_list:
            vessel_dir_list.remove(data_name)
        
    assert(len(img_dir_list) == len(an_dir_list))
    assert(len(img_dir_list) == len(vessel_dir_list))
    
    return img_dir_list, an_dir_list, vessel_dir_list


def split_dataset(img_name_list, an_name_list, vessel_name_list, split_id = 0, val_ratio = 0.1):
    img_train_list = []
    an_train_list = []
    vessel_train_list = []
    img_test_list = []
    an_test_list = []
    vessel_test_list = []
    img_val_list = []
    an_val_list = []
    vessel_val_list = []
    for i in range(len(img_name_list)):
        if int(img_name_list[i][-19]) == split_id:   # test set
            img_test_list.append(img_name_list[i])
            an_test_list.append(an_name_list[i])
            vessel_test_list.append(vessel_name_list[i])
        elif int(img_name_list[i][-19]) <= val_ratio * 10:   # val set
            img_val_list.append(img_name_list[i])
            an_val_list.append(an_name_list[i])
            vessel_val_list.append(vessel_name_list[i])
        else:    # train set
            img_train_list.append(img_name_list[i])
            an_train_list.append(an_name_list[i])
            vessel_train_list.append(vessel_name_list[i])
            
    assert(len(img_name_list) == len(an_train_list) + len(an_test_list) + len(an_val_list))
    return (img_train_list, an_train_list, vessel_train_list), (img_val_list, an_val_list, vessel_val_list), (img_test_list, an_test_list, vessel_test_list)

def img_norm(img_array):
    img_array -= np.mean(img_array)
    img_array /= np.std(img_array) + 1e-6
    return img_array

