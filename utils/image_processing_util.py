"""image processing utils
"""
import numpy as np

def cut_around(mask_arr, center, size = 64):
    small_mask = mask_arr[center[0]-size//2: center[0]+size//2, center[1]-size//2: center[1]+size//2, center[2]-size//2: center[2]+size//2]
    return small_mask
    
    