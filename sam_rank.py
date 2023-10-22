from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset import RankIoUDataset
from utils.parse import arg_parse_iou
from utils.log import get_logger
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
dataset = RankIoUDataset(args.image_info_root, mask_root = args.mask_root,
                         pgt1_root = args.pgt1_root, pgt2_root = args.pgt2_root, logger = logger)
dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, collate_fn = my_collate)

# prepare the model
sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
model = SAM_NoImEncoder(sam, args, logger)
save_ckpt = './fine-tuned_sam.pth'
logger.info(f'load ckpts from {save_ckpt}')
model.load_state_dict(torch.load(save_ckpt), strict = False)
model.to(device = device)

# prepare the optimizer, and use cosine scheduler
optimizer = torch.optim.AdamW(
    filter(lambda x: x.requires_grad is not False, model.parameters()),
    lr = args.lr
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = len(dataloader) * args.epoch_num,
    eta_min = args.min_lr
    )

# start training
model.train()
data_len = len(dataloader)
for epoch in range(args.epoch_num):
    epoch_avg_loss = 0
    for ii, item in enumerate(dataloader):
        avg_loss = model(item, epoch)

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        scheduler.step()

        logger.info(
            f'epoch:{epoch}, \
            iter:{ii}/{data_len}, \
            lr:{scheduler.get_last_lr()[0]}, \
            loss:{round(float(avg_loss), 3)}'
            )
        epoch_avg_loss += avg_loss

    epoch_avg_loss = epoch_avg_loss / data_len
    logger.info(f'epoch:{epoch}, epoch_avg_loss:{round(float(epoch_avg_loss), 3)}')

    if (epoch == (args.epoch_num - 1)) or ((epoch + 1) % 50 ==0):
        torch.save(model.state_dict(), os.path.join(args.out_path, f'ckpts_{epoch}.pth'))
