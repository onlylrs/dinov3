# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional

from scipy.optimize import linear_sum_assignment


class DetectionCriterion(nn.Module):
    """DETR detection loss with Hungarian matching"""
    
    def __init__(
        self,
        num_classes: int,
        matcher_cost_class: float = 1.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1,
        losses: List[str] = None,
    ):
        """
        Args:
            num_classes: Number of object classes
            matcher_cost_class: Class coefficient in Hungarian matching cost
            matcher_cost_bbox: L1 box coefficient in Hungarian matching cost  
            matcher_cost_giou: GIoU box coefficient in Hungarian matching cost
            weight_dict: Dict containing loss weights
            eos_coef: Background class coefficient
            losses: List of losses to compute
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou
        )
        
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        self.weight_dict = weight_dict
        
        if losses is None:
            losses = ['labels', 'boxes']
        self.losses = losses
        
        # Class weights for focal loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef  # Background class weight
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (focal loss)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [batch_size, num_queries, num_classes]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding box regression loss (L1 + GIoU)"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dict containing:
                - pred_logits: [batch_size, num_queries, num_classes]
                - pred_boxes: [batch_size, num_queries, 4]
            targets: List of dicts containing:
                - labels: [num_target_boxes]
                - boxes: [num_target_boxes, 4] in format (cx, cy, w, h) normalized
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for DETR"""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dict with:
                - "pred_logits": [batch_size, num_queries, num_classes]
                - "pred_boxes": [batch_size, num_queries, 4]
            targets: List of targets (len(targets) = batch_size)
        
        Returns:
            List of (index_i, index_j) where:
                - index_i: indices of selected predictions (in order)
                - index_j: indices of corresponding selected targets (in order)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it with the class probability
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the GIoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def box_cxcywh_to_xyxy(x):
    """Convert bbox from (cx, cy, w, h) to (x0, y0, x1, y1) format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert bbox from (x0, y0, x1, y1) to (cx, cy, w, h) format"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # Degenerate boxes gives inf / nan results, so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N, M, 2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def box_area(boxes):
    """Compute the area of a set of bounding boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])