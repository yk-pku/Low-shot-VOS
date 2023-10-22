from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import torch
import numpy as np
import os
import cv2


output_path = './1shot_cache'
os.makedirs(output_path, exist_ok = True)

low_shot_vos_gt_root = '../YouTube/phase2_1shot/yv_gt_480p'
low_shot_vos_img_root = '../YouTube/train_480p/JPEGImages'

print('prepare sam model')

sam_checkpoint = './ckpts/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device = device)
predictor = SamPredictor(sam)

print('start...')

vids = os.listdir(low_shot_vos_gt_root)
vid_num = len(vids)

for i, vid in enumerate(vids):
    if i % 100 == 0:
        print(f'{i} / {vid_num - 1}')
    save_root = os.path.join(output_path, vid)
    os.makedirs(save_root, exist_ok = True)
    imgs = os.listdir(os.path.join(low_shot_vos_img_root, vid))
    for img in imgs:
        img_path = os.path.join(low_shot_vos_img_root, vid, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        img_features = predictor.features.detach().cpu()
        original_size = predictor.original_size
        input_size = predictor.input_size
        save_dict = dict()
        save_dict['img_features'] = img_features
        save_dict['original_size'] = original_size
        save_dict['input_size'] = input_size
        save_path = os.path.join(save_root, img.replace('.jpg', '.pt'))
        torch.save(save_dict, save_path)    


