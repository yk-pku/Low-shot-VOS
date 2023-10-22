import numpy as np


def get_mask_from_label(mask, target_label):
    target_mask = np.where(mask == target_label, 1, 0)
    
    return target_mask