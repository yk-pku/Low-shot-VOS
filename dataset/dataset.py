import os
from typing import Any
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import sys
sys.path.append("..")
from utils.select_label import ram_select_label, seq_select_labels,seq_select_labels_torch
from utils.get_mask_from_label import get_mask_from_label
import json


class YoutubeDataset(Dataset):
    def __init__(self, image_info_root, mask_root, logger = None):
        self.vids = os.listdir(image_info_root)
        self.data = []
        self.masks = []
        for vid in self.vids:
            cur_data = dict()
            pt_name = os.listdir(os.path.join(image_info_root, vid))[0]
            # keys: img_features, original_size, input_size
            img_info = torch.load(os.path.join(image_info_root, vid, pt_name))
            cur_data.update(img_info)
            self.data.append(cur_data)
            self.masks.append(os.path.join(mask_root, vid, pt_name.replace('.pt', '.png')))

        if logger != None:
            logger.info(f'load {len(self.data)} images')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # data['original_size'] = data['original_size']
        # data['input_size'] = data['input_size']
        mask = Image.open(self.masks[index])
        mask = np.array(mask)
        input_mask, target_label = ram_select_label(mask)
        data['ori_mask'] = input_mask
        return data

def get_iou(mask1, mask2):
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

class RankIoUDataset(Dataset):
    def __init__(self, image_info_root, mask_root, pgt1_root, pgt2_root, logger = None):
        self.vids = os.listdir(pgt1_root)
        self.data = []
        self.masks = []
        self.pgt1 = []
        self.pgt2 = []
        for vid in self.vids:
            cur_data = dict()
            pt_name = os.listdir(os.path.join(image_info_root, vid))[0]
            img_info = torch.load(os.path.join(image_info_root, vid, pt_name))
            cur_data.update(img_info)
            self.data.append(cur_data)
            self.masks.append(os.path.join(mask_root, vid, pt_name.replace('.pt', '.png')))
            self.pgt1.append(os.path.join(pgt1_root, vid, pt_name.replace('.pt', '.png')))
            self.pgt2.append(os.path.join(pgt2_root, vid, pt_name.replace('.pt', '.png')))

        if logger != None:
            logger.info(f'load {len(self.data)} images')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data['vid'] = self.vids[idx]
        mask = Image.open(self.masks[idx])
        mask = np.array(mask)
        input_mask, target_label = ram_select_label(mask)
        pgt1 = Image.open(self.pgt1[idx])
        pgt1 = np.array(pgt1)
        input_pgt1 = get_mask_from_label(pgt1, target_label)
        pgt2 = Image.open(self.pgt2[idx])
        pgt2 = np.array(pgt2)
        input_pgt2 = get_mask_from_label(pgt2, target_label)
        
        input_pgt3 = input_pgt1 | input_pgt2

        IoU = []
        IoU.append(get_iou(input_pgt1, input_mask))
        IoU.append(get_iou(input_pgt2, input_mask))
        IoU.append(get_iou(input_pgt3, input_mask))

        candi_mask = [input_mask, input_pgt1, input_pgt2]

        data['sel_mask'] = candi_mask
        data['sel_iou'] = IoU

        return data



class RankIoUInferDataset(Dataset):
    def __init__(self, img_info_root, phase1_root, pre1_root, union_root, logger = None):
        self.vid_names = os.listdir(phase1_root)
        self.img = []
        self.img_info_root = img_info_root
        self.phase1_root = phase1_root
        self.pre1_root = pre1_root
        self.union_root = union_root
        self.vids = dict()
        for vid in self.vid_names:
            self.vids[vid] = sorted(os.listdir(os.path.join(img_info_root, vid)))

        if logger != None:
            logger.info(f'load {len(self.vid_names)} videos')

    def __len__(self):
        return len(self.vid_names)

    def __getitem__(self, idx):
        vid_name = self.vid_names[idx]
        frames = self.vids[vid_name]
        data = []
        for frame in frames:
            temp_data = dict()
            img_info = torch.load(os.path.join(self.img_info_root, vid_name, frame))
            temp_data.update(img_info)
            temp_data['vid'] = vid_name
            temp_data['frame'] = frame
            phase1_mask = Image.open(os.path.join(self.phase1_root, vid_name, frame.replace('.pt', '.png')))
            phase1_mask = np.array(phase1_mask)
            labels = np.unique(phase1_mask)
            labels = labels[labels != 0]
            temp_data['labels'] = labels
            temp_data['phase1_mask'] = phase1_mask
            pre1_mask = Image.open(os.path.join(self.pre1_root, vid_name, frame.replace('.pt', '.png')))
            pre1_mask = np.array(pre1_mask)
            temp_data['pre1_mask'] = pre1_mask
            union_mask = Image.open(os.path.join(self.union_root, vid_name, frame.replace('.pt', '.png')))
            union_mask = np.array(union_mask)
            temp_data['union_mask'] = union_mask
            data.append(temp_data)

        return data

class YoutubeDatasetAllObjects(Dataset):
    def __init__(self, data_json, image_info_root, gt_mask_root, mask_root, logger = None):
        self.image_info_root = image_info_root
        with open(data_json) as f:
            self.vids = json.load(f)['vids']
        
        self.datas = []
        self.masks = []
        self.gt_masks = []
        self.vids_idx = []
        for vid_idx, vid in enumerate(self.vids):
            imgs = os.listdir(os.path.join(image_info_root, vid))
            for img in imgs:
                img_info_path = os.path.join(vid, img)
                self.vids_idx.append(vid_idx)
                self.datas.append(img_info_path)
                self.gt_masks.append(os.path.join(gt_mask_root, vid, img.replace('.pt', '.png')))
                self.masks.append(os.path.join(mask_root, vid, img.replace('.pt', '.png')))
      
        if logger != None:
            logger.info(f'load {len(self.datas)} images')
        logger.info('dataset prepared')
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = torch.load(os.path.join(self.image_info_root,self.datas[index]))
        mask = Image.open(self.masks[index])
        mask = np.array(mask)
        gt_mask = Image.open(self.gt_masks[index])
        gt_mask = np.array(gt_mask)

        input_masks, target_labels = seq_select_labels(mask,is_voc=False)
        data['img_info_path'] = self.datas[index]
        data['ori_masks'] = input_masks
        data['mask_shape'] = mask.shape
        data['vid'] = self.vids_idx[index]
        data['ori_labels'] = target_labels
        data['gt_mask'] = gt_mask
        return data