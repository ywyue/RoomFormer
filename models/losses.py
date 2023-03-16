import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures.instances import Instances
from detectron2.utils.events import get_event_storage

from util.poly_ops import get_all_order_corners
from diff_ras.polygon import SoftPolygon
from util.bf_utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY

def custom_L1_loss(src_polys, target_polys, target_len):
    """L1 loss for coordinates regression
    We only calculate the loss between valid corners since we filter out invalid corners in final results
    Args:
        src_polys: Tensor of dim [num_target_polys, num_queries_per_poly*2] with the matched predicted polygons coordinates
        target_polys: Tensor of dim [num_target_polys, num_queries_per_poly*2] with the target polygons coordinates
        target_len: list of size num_target_polys, each element indicates 2 * num_corners of this poly
    """
    total_loss = 0.
    for i in range(target_polys.shape[0]):
        tgt_poly_single = target_polys[i, :target_len[i]]
        all_polys = get_all_order_corners(tgt_poly_single)
        total_loss += torch.cdist(src_polys[i, :target_len[i]].unsqueeze(0), all_polys , p=1).min()
    total_loss = total_loss/target_len.sum()
    return total_loss
    
class ClippingStrategy(nn.Module):
    def __init__(self, cfg, is_boundary=False):
        super().__init__()

        self.register_buffer("laplacian", torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3))

        self.is_boundary = is_boundary
        self.side_lengths = np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)

    # not used.
    def _extract_target_boundary(self, masks, shape):
        boundary_targets = F.conv2d(masks.unsqueeze(1), self.laplacian, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        # odd? only if the width doesn't match?
        if boundary_targets.shape[-2:] != shape:
            boundary_targets = F.interpolate(
                boundary_targets, shape, mode='nearest')

        return boundary_targets

    def forward(self, instances, clip_boxes=None, lid=0):                
        device = self.laplacian.device

        gt_masks = []

        if clip_boxes is not None:
            clip_boxes = torch.split(clip_boxes, [len(inst) for inst in instances], dim=0)
            
        for idx, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue

            if clip_boxes is not None:
                # todo, need to support rectangular boxes.
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    clip_boxes[idx].detach(), self.side_lengths[lid][0])
            else:
                gt_masks_per_image = instances_per_image.gt_masks.rasterize_no_crop(self.side_length).to(device)

            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        return torch.cat(gt_masks).squeeze(1)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def dice_loss_no_reduction(input, target):
    smooth = 1.

    iflat = input.flatten(-2,-1) # [200, 4096]
    tflat = target.flatten(-2,-1) # [200, 4096]
    intersection = (iflat * tflat).sum(1) # [200]
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum(1) + tflat.sum(1) + smooth))


@POLY_LOSS_REGISTRY.register()
class MaskRasterizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.register_buffer("rasterize_at", torch.from_numpy(np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)))
        # self.register_buffer("rasterize_at", torch.from_numpy(np.array([128, 128, 128, 128, 128, 128, 128, 128]).reshape(-1, 2)))
        # self.register_buffer("rasterize_at", torch.from_numpy(np.array([256, 256, 256, 256, 256, 256, 256, 256]).reshape(-1, 2)))
        self.inv_smoothness_schedule = (0.1,)
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = ()
        self.inv_smoothness_idx = 0
        self.iter = 0

        # whether to invoke our own rasterizer in "hard" mode.
        self.use_rasterized_gt = True
        
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.clip_to_proposal = False
        self.predict_in_box_space = True
        
        if self.clip_to_proposal or not self.use_rasterized_gt:
            self.clipper = ClippingStrategy(cfg=None)
            self.gt_rasterizer = None
        else:
            self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")

        self.offset = 0.5
        self.loss_fn = dice_loss
        self.name = "mask"

    def _create_targets(self, instances, clip_boxes=None, lid=0):
        if self.clip_to_proposal or not self.use_rasterized_gt:
            targets = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)            
        else:            
            targets = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
            
        return targets

    def forward(self, preds, targets, target_len, lid=0):

        resolution = self.rasterize_at[lid]

        target_masks = []
        pred_masks = []
        for i in range(targets.shape[0]):

            tgt_poly_single = targets[i, :target_len[i]].view(-1, 2).unsqueeze(0)
            pred_poly_single = preds[i, :target_len[i]].view(-1, 2).unsqueeze(0)

            tgt_mask = self.gt_rasterizer(tgt_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            tgt_mask = (tgt_mask + 1)/2

            pred_mask = self.pred_rasterizer(pred_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            target_masks.append(tgt_mask)
            pred_masks.append(pred_mask)

        pred_masks = torch.stack(pred_masks)
        target_masks = torch.stack(target_masks)
            
        return self.loss_fn(pred_masks, target_masks)


class MaskRasterizationCost(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.register_buffer("rasterize_at", torch.from_numpy(np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)))
        # self.register_buffer("rasterize_at", torch.from_numpy(np.array([128, 128, 128, 128, 128, 128, 128, 128]).reshape(-1, 2)))
        self.inv_smoothness_schedule = (0.1,)
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = ()
        self.inv_smoothness_idx = 0
        self.iter = 0

        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")

        # whether to invoke our own rasterizer in "hard" mode.
        self.use_rasterized_gt = True
        
        self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")
        
        self.offset = 0.5
        self.loss_fn = dice_loss_no_reduction
        self.name = "mask"

    def mask_iou(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        ) -> torch.Tensor:
        """
        Inputs:
        mask1: NxHxW torch.float32. Consists of [0, 1]
        mask2: NxHxW torch.float32. Consists of [0, 1]
        Outputs:
        ret: NxM torch.float32. Consists of [0 - 1]
        """

        N, H, W = mask1.shape
        M, H, W = mask2.shape

        mask1 = mask1.view(N, H*W)
        mask2 = mask2.view(M, H*W)

        intersection = torch.matmul(mask1, mask2.t())

        area1 = mask1.sum(dim=1).view(1, -1)
        area2 = mask2.sum(dim=1).view(1, -1)

        union = (area1.t() + area2) - intersection

        ret = torch.where(
            union == 0,
            torch.tensor(0., device=mask1.device),
            intersection / union,
        )

        return ret

    def forward(self, preds, targets, target_len, lid=0):

        resolution = self.rasterize_at[lid]

        target_masks = []
        pred_masks = []

        cost_mask = torch.zeros([preds.shape[0], targets.shape[0]], device=preds.device)

        for i in range(targets.shape[0]):

            tgt_poly_single = targets[i, :target_len[i]].view(-1, 2).unsqueeze(0)
            pred_poly_all = preds[:,:target_len[i]].view(preds.shape[0], -1, 2)

            tgt_mask = self.gt_rasterizer(tgt_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            pred_masks = self.pred_rasterizer(pred_poly_all * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)

            tgt_mask = (tgt_mask + 1)/2
            tgt_masks = tgt_mask.repeat(preds.shape[0], 1, 1)

            cost_mask[:, i] = self.loss_fn(tgt_masks, pred_masks) 
 
            
        return cost_mask