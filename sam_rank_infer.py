from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset import RankIoUInferDataset
from utils.parse import arg_parse_iou
from utils.log import get_logger
from utils.get_mask_from_label import get_mask_from_label
from models.sam_iou_rank import SAM_NoImEncoder
import os
import json

def my_collate(batch):
        data = []
        for item in batch:
            data.append(item)
        return data

args = arg_parse_iou()

# get logger
os.makedirs(args.out_path, exist_ok = True)
logger = get_logger(args.out_path)

logger.info(f'configs\n{json.dumps(vars(args), indent=2)}')

# prepare the dataset
dataset = RankIoUInferDataset(args.image_info_root, phase1_root = args.phase1_root,
                         pre1_root = args.pre1_root, union_root = args.union_root, logger = logger)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = my_collate)

# prepare the model
sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
model = SAM_NoImEncoder(sam, args, logger)
logger.info(f'load ckpts from {args.save_ckpt}')
model.load_state_dict(torch.load(args.save_ckpt), strict = False)
model.to(device = device)

# start training
model.eval()
data_len = len(dataloader)

from PIL import Image
# get palette
palette = Image.open('../YouTube/valid/Annotations/ffd7c15f47/00000.png').getpalette()

for i, item in enumerate(dataloader):
    data = item[0]
    vid_name = data[0]['vid']
    logger.info(f'{vid_name}:{i}/{data_len}')
    
    this_out_path = os.path.join(args.out_path, vid_name)
    os.makedirs(this_out_path, exist_ok = True)
    
    for f_data in data:
        img_f = f_data['img_features']
        img_name = f_data['frame']
        phase1_mask = f_data['phase1_mask']
        pre1_mask = f_data['pre1_mask']
        union_mask = f_data['union_mask']
        res_mask = np.zeros((union_mask.shape[0], union_mask.shape[1])).astype(np.uint8)
        labels = f_data['labels']
        if len(labels) == 0:
            img_E = Image.fromarray(union_mask)
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, img_name.replace('.pt', '.png')))
            continue
        for label in labels:
            phase1_mask_l = get_mask_from_label(phase1_mask, label)
            pre1_mask_l = get_mask_from_label(pre1_mask, label)
            union_mask_l = get_mask_from_label(union_mask, label)
            input_masks = [phase1_mask_l, pre1_mask_l, union_mask_l]
            scores = model.infer(f_data, input_masks)
            select_mask = input_masks[torch.argmax(scores)]
            select_idx = np.where(select_mask == 1, True, False)
            res_mask[select_idx] = label
            
        img_E = Image.fromarray(res_mask)
        img_E.putpalette(palette)
        img_E.save(os.path.join(this_out_path, img_name.replace('.pt', '.png')))


