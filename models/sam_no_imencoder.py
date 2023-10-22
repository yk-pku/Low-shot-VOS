import torch
import torch.nn as nn
import numpy as np
from utils.mask_to_points import mask_to_points_grid, mask_to_box
from utils.aug_mask import point_disturb
from utils.loss import BootstrappedBCE, FocalBCE, DiceLoss
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from typing import Optional, Tuple
import os

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

class SAM_NoImEncoder(nn.Module):
    def __init__(self, sam, args, logger = None):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.img_size = sam.image_encoder.img_size
        self.mask_threshold = sam.mask_threshold

        for name, para in self.prompt_encoder.named_parameters():
            if 'point_embeddings.1.weight' in name:
                if args.point_prompt_fix:
                    para.requires_grad = False
                else:
                    para.requires_grad = True
            else:
                para.requires_grad = False

        if args.decoder_fix:
            for name, para in self.mask_decoder.named_parameters():
                para.requires_grad = False

        self.args = args
        self.logger = logger
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)


    def forward(self, bs_data, epoch):
        bs = len(bs_data)

        loss_all = 0
        iou_all = 0
        for i, data in enumerate(bs_data):
            img_features = data['img_features'].cuda()
            original_size = data['original_size']
            input_size = data['input_size']
            ori_mask = data['ori_mask']

            # select points
            coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

            if self.args.aug_mask:
                if epoch >= self.args.start_aug:
                    input_points, input_labels = mask_to_points_grid(ori_mask, point_num = self.args.point_num)
                    if self.args.point_disturb:
                        input_points = point_disturb(input_points, ori_mask, self.args)
                        input_labels = input_labels[:len(input_points)]
                else:
                    input_points, input_labels = mask_to_points_grid(ori_mask, point_num = self.args.point_num)
            else:
                input_points, input_labels = mask_to_points_grid(ori_mask, point_num = self.args.point_num)

            # prepare point prompts

            point_coords = self.transform.apply_coords(input_points, original_size)

            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device())
            labels_torch = torch.as_tensor(input_labels, dtype=torch.int, device=self.device())
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

            # forward prompt encoder
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(coords_torch, labels_torch),
                boxes=box_torch,
                masks=mask_input_torch,
            )

            # forward decoder
            low_res_masks, iou_predictions, _ = self.mask_decoder(
                image_embeddings = img_features,
                image_pe = self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = self.args.multimask_output,
            )

            # Upscale the masks to the original image resolution
            masks = self.postprocess_masks(low_res_masks, input_size, original_size)
            
            # Compute loss
            gt_mask = torch.tensor(ori_mask, dtype = float).cuda()
            if self.args.bsbce:
                loss = self.bsbceloss(masks[0], gt_mask[None, :], epoch)
            else:
                loss = F.binary_cross_entropy_with_logits(masks[0], gt_mask[None, :])

            loss += self.fbceloss(masks[0], gt_mask[None, :])

            loss += 0.5*self.dloss(masks[0], gt_mask[None, :])

            iou = self.IoU(masks[0], gt_mask[None, :])
            loss_all += loss
            iou_all += iou

        return loss_all / bs, iou_all / bs

    @torch.no_grad()
    def infer(self, bs_data):
        bs = len(bs_data)
        return_mask = []
        for i, data in enumerate(bs_data):# every img
            img_features = data['img_features'].cuda()
            original_size = data['original_size']
            input_size = data['input_size']
            ori_masks = data['ori_masks']
            ori_labels = data['ori_labels']
            gt_mask = data['gt_mask']

            # select points
            mask_one_image = []
            if ori_masks is None:# no object
                return_mask.append(mask_one_image)
                continue
            

            save_path = os.path.join(self.args.coords_root,data['img_info_path'])
            if self.args.gt:
                save_path = save_path.split('.pt')[0] + 'gt_16_25.pt'
            else:
                save_path = save_path.split('.pt')[0] + 'pl_16_25.pt'
            save_dict = dict()
            exist = False
            if os.path.exists(save_path):
                save_dict = torch.load(save_path)
                exist = True

            for obj_idx, ori_mask in enumerate(ori_masks):# every object
                mask_one_object = []
                coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
                for idx, point_num in enumerate(self.sample_point_num):
                    if exist:
                        try:
                            input_points = save_dict[f'input_points_{obj_idx}_{idx}']
                            input_labels = save_dict[f'input_labels_{obj_idx}_{idx}']
                        except:
                            print('error')
                            input_points, input_labels = mask_to_points_grid(ori_mask, point_num = point_num)
                    else:
                        input_points, input_labels = mask_to_points_grid(ori_mask, point_num = point_num)
                        if self.args.save_sample:
                            save_dict[f'input_points_{obj_idx}_{idx}'] = input_points
                            save_dict[f'input_labels_{obj_idx}_{idx}'] = input_labels
                    if self.args.box:
                        gt_mask_ = (gt_mask == ori_labels[obj_idx])  
                        box_ = mask_to_box(gt_mask_)
                        if box_ is not None:
                            box_ = self.transform.apply_boxes(box_,original_size)
                            box_torch = torch.as_tensor(box_, dtype=torch.float,device=self.device())
                            box_torch = box_torch[None,:]
                    
                    point_coords = self.transform.apply_coords(input_points, original_size)
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device())
                    labels_torch = torch.as_tensor(input_labels, dtype=torch.int, device=self.device())
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

                    # forward prompt encoder
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=(coords_torch, labels_torch),
                        boxes=box_torch,
                        masks=mask_input_torch,
                    )

                    # forward decoder
                    low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings = img_features,
                        image_pe = self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings = sparse_embeddings,
                        dense_prompt_embeddings = dense_embeddings,
                        multimask_output = self.args.multimask_output,
                    )
                    # Upscale the masks to the original image resolution
                    masks = self.postprocess_masks(low_res_masks, input_size, original_size) 
                    

                    # prompt_num = 1, multimask = True -> masks [1,3,h,w], low_res_masks [1,3,256,256]
                    # prompt_num = 1, multimask = False -> masks [1,1,h,w], low_res_masks [1,1,256,256]
                    # choose mask from best one
                    max_idx = 0
                    if self.args.persam:
                        weights = torch.cat((1 - self.mask_weights.weights.sum(0).unsqueeze(0), self.mask_weights.weights), dim=0).unsqueeze(0)
                 
                        b, n, h, w = masks.shape
                        masks = masks.flatten(2)
                        masks = masks * weights
                        masks = masks.sum(1).unsqueeze(0)
                        masks = masks.reshape(1, 1, h, w)

                        b, n, h, w = low_res_masks.shape
                        low_res_masks = low_res_masks.flatten(2)
                        low_res_masks = low_res_masks * weights
                        low_res_masks = low_res_masks.sum(1).unsqueeze(0)
                        low_res_masks = low_res_masks.reshape(1,1,h,w)

                        b, n, = iou_predictions.shape
                        iou_predictions = iou_predictions * weights.squeeze(2)
                        iou_predictions = iou_predictions.sum(1).unsqueeze(1)
       

                    masks = masks.detach().cpu().numpy()
                    mask = masks[0][max_idx] > 0.5
                    if idx == 0:
                        mask_input_torch = low_res_masks[:,max_idx,:,:][:,None,:,:]
                    mask_one_object.append(mask) 
                mask_one_image.append(mask_one_object)
            return_mask.append(mask_one_image)

        return return_mask

    def IoU(self, pre, tar, ori = False):
        tar = tar > 0.5
        if not ori:
            pre = F.sigmoid(pre)
            pre = pre > 0.5
        iou = torch.sum(pre & tar) / torch.sum(pre | tar)
        return iou

    def device(self) -> torch.device:
        return next(self.prompt_encoder.parameters()).device

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
