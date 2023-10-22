import numpy as np
import torch

def ram_select_label(mask, is_voc = False): 
    labels = np.unique(mask)
    fg_labels = labels[labels != 0]
    if is_voc:
        fg_labels = fg_labels[fg_labels != 255]
    if len(fg_labels) == 0:
        return None, None
    target_label = np.random.choice(fg_labels)
    target_mask = np.where(mask == target_label, 1, 0)
    
    return target_mask, target_label

def seq_select_labels(mask, is_voc):
    labels = np.unique(mask)
    fg_labels = labels[labels != 0]
    if is_voc:
        fg_labels = fg_labels[fg_labels != 255]
    if len(fg_labels) == 0:
        return None, None
    target_labels = []
    target_masks = []
    for fl in fg_labels:
        tl = fl 
        tm = np.where(mask == tl, 1, 0)
        target_labels.append(tl)
        target_masks.append(tm)
    
    return target_masks, target_labels 

def seq_select_labels_torch(mask, is_voc):
    labels = torch.unique(torch.as_tensor(mask))

    fg_labels = labels[labels != 0]
    if is_voc:
        fg_labels = fg_labels[fg_labels != 255]
    if len(fg_labels) == 0:
        return None, None
    target_labels = []
    target_masks = []
    for fl in fg_labels:
        tl = fl 
        try:
            tm = torch.where(mask == tl, 1, 0)
        except:
            import pdb; pdb.set_trace()
        target_labels.append(tl)
        target_masks.append(tm)
    
    return target_masks, target_labels 