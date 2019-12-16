"""image processing utils
"""
import numpy as np

def cut_around(mask_arr, center, size = 64):
    small_mask = mask_arr[center[0]-size//2: center[0]+size//2, center[1]-size//2: center[1]+size//2, center[2]-size//2: center[2]+size//2]
    return small_mask
    
def get_label_mean(input_tuple, dim):
    x_mean = np.mean(input_tuple[dim])
    return x_mean

def img_norm(img_array, eps):
    img_array -= np.mean(img_array)
    img_array /= (np.std(img_array) + eps)
    return img_array

