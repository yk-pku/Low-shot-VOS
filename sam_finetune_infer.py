import os
import argparse
import yaml
from PIL import Image
import numpy as np
import time
import random
from tqdm import tqdm 
from collections import OrderedDict 

from torch.utils.data import DataLoader
from dataset.dataset import YoutubeDatasetAllObjects
from utils.log import get_logger
from segment_anything import sam_model_registry
from models.sam_no_imencoder import SAM_NoImEncoder
import os

import json
from utils.remove_small import remove_small
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument('--data_json', './dataset/sample_vids_all.json')
    parser.add_argument('--gt_mask_root', default = '../YouTubeVOS/Annotations/',
                        help='ground truth')
    parser.add_argument('--mask_root',
                        help='mask from pseudo label or gt, to be refined' )
    parser.add_argument('--gt',
                        help='whether or not use gt_mask_root as mask_root')
    parser.add_argument('--image_info_root', default = './1shot_cache/')
    parser.add_argument('--coords_root', default = './1shot_cache_coords/')
    parser.add_argument('--save_sample',action='store_true')

    # save_path
    parser.add_argument('--samresult_path', default = './YouTubeVOS_JF/')
    parser.add_argument('--name', default = 'samtest')

    # sample method
    parser.add_argument('--point_num',default = 16 ,type = int)
    parser.add_argument('--point_num_after',default = 25 ,type = int)
    parser.add_argument('--model_predict_time',default=1,type = int)
    parser.add_argument('--box', action='store_true')

    # model
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--model-type", type=str, default="vit_h", help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",)
    parser.add_argument('--persam', action='store_true')
    parser.add_argument('--multimask_output',action='store_true')
    parser.add_argument("--sam_checkpoint", type=str, 
                        help="The path to the SAM checkpoint to use for mask generation.")

    # calculate metrics config
    parser.add_argument("--cal_iou",default=True,type=bool,
                        help='refine mask(predicted once by sam) vs gt_mask')
    parser.add_argument("--cal_iou_post_refine",default=True,type=bool,
                        help='refine mask(predicted twice by sam) vs gt_mask')
    parser.add_argument("--cal_iou_union",default=True,type=bool,
                        help='(refine mask predicted by sam & original mask) vs gt_mask')
    # save image config
    parser.add_argument('--save_removesmall_imgs',action = 'store_true')
    parser.add_argument("--save_refine_imgs",action = 'store_true')
    parser.add_argument("--save_predict_time",default=1,type = int)
    parser.add_argument("--noUnion", action='store_true',
                        help='save refine mask without Union')
    args = parser.parse_args()

    if args.gt_mask_root == args.mask_root:#sample from gt
        args.cal_iou_union = False
        args.gt = True
    else:
        args.gt = False
    
    if args.save_refine_imgs:
        args.samresult_path = './YouTubeVOS_Finetune_Mask/'
        args.image_info_root = './1shot_cache_all/'
        args.data_json = './dataset/sample_vids_all.json'
    
    if args.save_removesmall_imgs:
        args.samresult_path = './YouTubeVOS_RemoveSmall_Mask/'
        args.image_info_root = './1shot_cache_all/'
        args.data_json = './dataset/sample_vids_all.json'

    if not os.path.exists(os.path.join(args.samresult_path, args.name)):
        os.makedirs(os.path.join(args.samresult_path, args.name))
    with open('%s%s/config.yml' % (args.samresult_path, args.name), 'w') as f:
        yaml.dump(args, f)

    return args

def my_collate(batch):
        data = []
        for item in batch:
            data.append(item)
        return data

def main(args,logger):
    logger.info('loading sam model')

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    model = SAM_NoImEncoder(sam, args, logger)
    model.to(device = args.device)

    logger.info('prepare dataset')
    dataset = YoutubeDatasetAllObjects(args.data_json, args.image_info_root, gt_mask_root=args.gt_mask_root , mask_root = args.mask_root, logger=logger)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = my_collate)
    palette = Image.open('../YouTubeVOS/Annotations/ffb904207d/00000.png').getpalette()

    logger.info('start testing')
    pbar = tqdm(total=len(dataloader))

    for ii, items in enumerate(dataloader):
        #----save_removesmall_imgs----
        if args.save_removesmall_imgs:
            item = items[0]# only one img
            cur_result = np.zeros(item['mask_shape']).astype(np.uint8)
            if item['ori_masks'] is not None:
                for ri_mask, ri_label in zip(item['ori_masks'],item['ori_labels']):
                    ri_mask_refine = remove_small(ri_mask)
                    cur_result[ri_mask_refine] = ri_label

            img_E = Image.fromarray(cur_result)
            img_E.putpalette(palette)
            vid, img_name = item['img_info_path'].split('/')
            this_out_path = os.path.join(args.samresult_path, args.name, 'Annotations', vid)
            os.makedirs(this_out_path, exist_ok=True)
            img_E.save(os.path.join(this_out_path, img_name.replace('.pt','.png')))

            pbar.update(1)
            continue

        out_masks_bs = model.infer(items) 

        #only one img
        out_masks = out_masks_bs[0]
        item = items[0]
        #---save refine imgs---
        if args.save_refine_imgs:
            cur_result = np.zeros(item['mask_shape']).astype(np.uint8)
            if item['ori_masks'] is not None:
                for out_mask, ri_mask, ri_label in zip(out_masks,item['ori_masks'],item['ori_labels']):
                    out_mask_np = out_mask[args.save_predict_time-1]   
                    if args.noUnion:
                        cur_result[out_mask_np] = ri_label
                    else:
                        cur_result[out_mask_np | ri_mask ] = ri_label

            img_E = Image.fromarray(cur_result)
            img_E.putpalette(palette)
            vid, img_name = item['img_info_path'].split('/')
            this_out_path = os.path.join(args.samresult_path, args.name, 'Annotations', vid)
            os.makedirs(this_out_path, exist_ok=True)
            img_E.save(os.path.join(this_out_path, img_name.replace('.pt','.png')))

            pbar.update(1)
            continue
     
    
if __name__ == "__main__":
    start = time.perf_counter()

    args = parse_args()
    logger = get_logger(os.path.join(args.samresult_path, args.name))
    main(args,logger)

    end = time.perf_counter() 

    logger.info(f'running time:{end - start}s')