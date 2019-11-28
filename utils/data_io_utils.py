import numpy as np
import os
import random

def get_file_dir_lists(load_path, drop_data):
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
    if drop_data:
        img_dir_list.remove('/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000105.T0002CVAImage.mha')
        an_dir_list.remove('/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000105.T0002Aneurysms.mha')
        vessel_dir_list.remove('/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000105.T0002CVAVesselMask.mha')
        img_dir_list.remove('/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000208.T0001CVAImage.mha')
        an_dir_list.remove('/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000208.T0001Aneurysms.mha')
        vessel_dir_list.remove('/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000208.T0001CVAVesselMask.mha')
        img_dir_list.remove('/data/maze/3D_An_Seg_20190628/01.CVAImage/A002.P000288.T0001CVAImage.mha')
        an_dir_list.remove('/data/maze/3D_An_Seg_20190628/03.Aneurysms/A002.P000288.T0001Aneurysms.mha')
        vessel_dir_list.remove('/data/maze/3D_An_Seg_20190628/02.CVAVesselMask/A002.P000288.T0001CVAVesselMask.mha')
        
    assert(len(img_dir_list) == len(an_dir_list))
    assert(len(img_dir_list) == len(vessel_dir_list))
    
    return img_dir_list, an_dir_list, vessel_dir_list


def random_seperate_dataset(ratio = 0.8):    
    rand_int = random.randint(1,100)
    if rand_int <= ratio*100:
        return 'train_set'
    else:
        return 'test_set'
    
def small_image_mask_block(img_array, mask_array, vessel_array, size):
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
        one_small_mask = vessel_array[center_int[0]-size//2: center_int[0]+size//2, center_int[1]-size//2: center_int[1]+size//2, center_int[2]-size//2: center_int[2]+size//2]
        one_small_img = img_array[center_int[0]-size//2: center_int[0]+size//2, center_int[1]-size//2: center_int[1]+size//2, center_int[2]-size//2: center_int[2]+size//2]
        small_img_list.append(one_small_img)
        small_mask_list.append(one_small_mask)
        
    small_img_array = np.array(small_img_list)
    small_mask_array = np.array(small_mask_list)
    assert(small_img_array.shape == (an_nums, size, size, size))
    assert(small_mask_array.shape == (an_nums, size, size, size))
    
    return small_img_array, small_mask_array

def img_norm(img_array):
    img_array -= np.mean(img_array)
    img_array /= np.std(img_array) + 1e-6
    return img_array

