import torch
import torch.nn as nn
import numpy as np
from utils.mask_to_points import mask_to_points_grid
from utils.aug_mask import aug_mask_circular, point_disturb
from utils.loss import BootstrappedBCE, FocalBCE, DiceLoss
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class SAM_NoImEncoder(nn.Module):
    def __init__(self, sam, args, logger = None):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.img_size = sam.image_encoder.img_size
        self.mask_threshold = sam.mask_threshold
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if args.feature256:
            logger.info(f'use feature256')
            self.ranknet = MLP(input_dim = 256, hidden_dim = 128, output_dim = 1, num_layers = 3, sigmoid_output = False)
        else:
            self.ranknet = MLP(input_dim = 32, hidden_dim = 32, output_dim = 1, num_layers = 3, sigmoid_output = False)

        for name, para in self.prompt_encoder.named_parameters():
            para.requires_grad = False

        for name, para in self.mask_decoder.named_parameters():
            para.requires_grad = False

        self.args = args
        self.logger = logger
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)


    def forward(self, bs_data, epoch):
        bs = len(bs_data)

        loss_all = 0.0
        for i, data in enumerate(bs_data):
            img_features = data['img_features'].cuda()
            original_size = data['original_size']
            input_size = data['input_size']
            sel_mask = data['sel_mask']
            sel_iou = data['sel_iou']
        
            pre_logits = []
            for j in range(3):
                # select points
                coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

                input_points, input_labels = mask_to_points_grid(sel_mask[j], point_num = self.args.point_num)

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
                low_res_masks, iou_predictions, upscaled_embedding, src = self.mask_decoder(
                    image_embeddings = img_features,
                    image_pe = self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = self.args.multimask_output,
                )
                upscaled_embedding_pooling = self.avgpool(upscaled_embedding)
                upscaled_embedding_pooling = upscaled_embedding_pooling.view(1, 32)
                src = self.avgpool(src)
                src = src.view(1, 256)
 
                if self.args.feature256:
                    pre_logits.append(src)
                else:
                    pre_logits.append(upscaled_embedding_pooling)

            pre_logits = torch.cat(pre_logits, dim = 0)

            rank_scores = self.ranknet(pre_logits)


            # Compute loss
            if self.args.feature256:
                loss = max(0, self.indicator(sel_iou[0], sel_iou[1])*(rank_scores[0] - rank_scores[1]) + 0.005) 
                loss += max(0, self.indicator(sel_iou[0], sel_iou[2])*(rank_scores[0] - rank_scores[2]) + 0.005) 
                loss += max(0, self.indicator(sel_iou[1], sel_iou[2])*(rank_scores[1] - rank_scores[2]) + 0.005)
            else:
                loss = max(0, self.indicator(sel_iou[0], sel_iou[1])*(rank_scores[0] - rank_scores[1]) + 0.01)
                loss += max(0, self.indicator(sel_iou[0], sel_iou[2])*(rank_scores[0] - rank_scores[2]) + 0.01)
                loss += max(0, self.indicator(sel_iou[1], sel_iou[2])*(rank_scores[1] - rank_scores[2]) + 0.01)

            loss_all += loss

        return loss_all / bs
    
    def indicator(x, y):
        if x > y:
            return -1
        else:
            return 1
        
    @torch.no_grad()
    def infer(self, data, input_masks):
        img_features = data['img_features'].cuda()
        original_size = data['original_size']

        res_scores = []
        for i, input_mask in enumerate(input_masks):
            # select points
            coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
            input_points, input_labels = mask_to_points_grid(input_mask, point_num = self.args.point_num)

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
            low_res_masks, iou_predictions, upscaled_embedding, src = self.mask_decoder(
                image_embeddings = img_features,
                image_pe = self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = self.args.multimask_output,
            )

            upscaled_embedding_pooling = self.avgpool(upscaled_embedding)
            upscaled_embedding_pooling = upscaled_embedding_pooling.view(1, 32)
            src = self.avgpool(src)
            src = src.view(1, 256)
            if self.args.feature256:
                score = self.ranknet(src)
            else:
                score = self.ranknet(upscaled_embedding_pooling)

            score = score.view(-1)
            res_scores.append(score)

        return torch.cat(res_scores)

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
