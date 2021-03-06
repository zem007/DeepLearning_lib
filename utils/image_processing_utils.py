"""image processing utils
"""
import numpy as np
from common.variables import *

def cut_around_3D(mask_arr, center, size = 64):
    """ cut small mask from original mask by using center point location
        Args:
            mask_arr: np.array of shape (x, y, z)
            center: list, eg. [100, 100, 100], the xyz coordinates
            size: int, the cut size
        Returns: 
            small_mask: np.array of shape (size, size, size)
    """
    center = [int(round(i)) for i in center]
    mask_arr = np.pad(mask_arr, ((size//2, size//2), (size//2, size//2), (size//2, size//2)), mode = 'constant')
    small_mask = mask_arr[center[0]: center[0]+size, center[1]: center[1]+size, center[2]: center[2]+size]
    return small_mask

def get_label_mean(mask_image, axis = 0):
    """ get the center point of each an
        Args: 
            mask_image: the original An image array
            axis: int, average which axis
        Returns:
            mean_dict: dict, key is the an num, and value is the 3d center coordinates. eg, {1: [100,100,100],
                                                                  2: [200,200,200],
                                                                  3: [43, 345,172]} 
    """
    # nums of An. 1 or 2 or 3
    an_nums = int(np.max(mask_image))
    mean_dict = {}
    for i in range(an_nums):
        print('an #: ', str(i+1))
        # all index for one an            
        one_an_index = np.where(mask_image == (i+1))
        one_an_array = np.array(one_an_index)    # tuple to array
        one_an_array = np.transpose(one_an_array)    # transpose to (n, 3)
        mean_dims = np.mean(one_an_array, axis = axis)
        mean_dict.update({(i+1): mean_dims})
    return mean_dict


def normalize_image(img_array, eps = EPS):
    """ standard normalization
    """
    img_array -= np.mean(img_array)
    img_array /= (np.std(img_array) + eps)
    return img_array

