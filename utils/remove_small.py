from skimage.morphology import remove_small_objects
import numpy as np

def remove_small(mask):
    '''
    mask: np.array
    '''
    min_size = np.sum(mask)//80
    mask_refine = remove_small_objects(mask, min_size=min_size, connectivity=1)
    return mask_refine
