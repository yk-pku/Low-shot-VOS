import torch 
import torch.nn as nn
import torch.nn.functional as F

class BootstrappedBCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.start_warm = args.start_warm
        self.end_warm = args.end_warm
        self.top_p = args.top_p

    # input: [1, h, w], target: [1, h, w]
    def forward(self, input, target, cur_epoch):
        if cur_epoch < self.start_warm:
            return F.binary_cross_entropy_with_logits(input, target)
        
        raw_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if cur_epoch > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - cur_epoch)/(self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted = False)

        return loss.mean()
        
class FocalBCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
    
    def forward(self, input, target):
        prob = input.sigmoid()
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction = 'none')
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, input, target):
        prob = input.sigmoid()
        numerator = 2 * (prob * target).sum()
        denominator = prob.sum() + target.sum()
        loss = 1 - (numerator + 1) / (denominator + 1)

        return loss
