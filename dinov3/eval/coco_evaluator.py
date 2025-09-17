# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict


class COCOEvaluator:
    """COCO-style evaluator for mAP@50 and mAP@0.5:0.95"""
    
    def __init__(self, iou_types: List[str] = ['bbox']):
        """
        Args:
            iou_types: List of IoU types to evaluate ('bbox', 'segm', 'keypoints')
        """
        self.iou_types = iou_types
        self.reset()
    
    def reset(self):
        """Reset evaluation state"""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        """
        Update with new predictions and targets
        
        Args:
            predictions: List of prediction dicts with keys:
                - 'boxes': [N, 4] in xyxy format
                - 'scores': [N]
                - 'labels': [N]
            targets: List of target dicts with keys:
                - 'boxes': [N, 4] in xyxy format  
                - 'labels': [N]
                - 'image_id': image id
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute mAP metrics"""
        if len(self.predictions) == 0:
            return {'mAP@50': 0.0, 'mAP@0.5:0.95': 0.0}
        
        # Convert to COCO evaluation format
        eval_results = {}
        
        # Group predictions and targets by image_id
        pred_by_img = defaultdict(list)
        tgt_by_img = defaultdict(list)
        
        for pred, tgt in zip(self.predictions, self.targets):
            img_id = tgt['image_id'].item() if torch.is_tensor(tgt['image_id']) else tgt['image_id']
            pred_by_img[img_id].append(pred)
            tgt_by_img[img_id].append(tgt)
        
        # Compute mAP@0.5:0.95 and mAP@0.5
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # 0.5:0.05:0.95
        
        all_aps_50_95 = []
        all_aps_50 = []
        
        # Get all unique classes
        all_classes = set()
        for pred_list in pred_by_img.values():
            for pred in pred_list:
                if len(pred['labels']) > 0:
                    all_classes.update(pred['labels'].tolist())
        
        for tgt_list in tgt_by_img.values():
            for tgt in tgt_list:
                if len(tgt['labels']) > 0:
                    all_classes.update(tgt['labels'].tolist())
        
        if not all_classes:
            return {'mAP@50': 0.0, 'mAP@0.5:0.95': 0.0}
        
        # Compute AP for each class
        class_aps_50_95 = {}
        class_aps_50 = {}
        
        for cls in all_classes:
            # Collect all predictions and targets for this class
            cls_predictions = []
            cls_targets = []
            
            for img_id in pred_by_img.keys():
                pred_list = pred_by_img[img_id]
                tgt_list = tgt_by_img.get(img_id, [])
                
                # Get predictions for this class
                for pred in pred_list:
                    if len(pred['labels']) > 0:
                        mask = pred['labels'] == cls
                        if mask.any():
                            cls_predictions.append({
                                'boxes': pred['boxes'][mask],
                                'scores': pred['scores'][mask],
                                'image_id': img_id
                            })
                
                # Get targets for this class  
                for tgt in tgt_list:
                    if len(tgt['labels']) > 0:
                        mask = tgt['labels'] == cls
                        if mask.any():
                            cls_targets.append({
                                'boxes': tgt['boxes'][mask],
                                'image_id': img_id
                            })
            
            if not cls_predictions or not cls_targets:
                class_aps_50_95[cls] = 0.0
                class_aps_50[cls] = 0.0
                continue
            
            # Compute AP for this class at different IoU thresholds
            aps_per_iou = []
            for iou_thresh in iou_thresholds:
                ap = self._compute_ap(cls_predictions, cls_targets, iou_thresh)
                aps_per_iou.append(ap)
            
            class_aps_50_95[cls] = np.mean(aps_per_iou)
            class_aps_50[cls] = aps_per_iou[0]  # IoU@0.5 is the first threshold
        
        # Compute mean over all classes
        map_50_95 = np.mean(list(class_aps_50_95.values()))
        map_50 = np.mean(list(class_aps_50.values()))
        
        return {
            'mAP@50': map_50,
            'mAP@0.5:0.95': map_50_95,
            'class_aps_50': class_aps_50,
            'class_aps_50_95': class_aps_50_95
        }
    
    def _compute_ap(self, predictions: List[Dict], targets: List[Dict], iou_threshold: float) -> float:
        """Compute Average Precision for a single class at given IoU threshold"""
        
        # Collect all predictions with scores
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_img_ids = []
        
        for pred in predictions:
            boxes = pred['boxes']
            scores = pred['scores'] 
            img_id = pred['image_id']
            
            all_pred_boxes.append(boxes)
            all_pred_scores.append(scores)
            all_pred_img_ids.extend([img_id] * len(boxes))
        
        if not all_pred_boxes:
            return 0.0
            
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        
        # Sort by confidence score
        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]
        all_pred_img_ids = [all_pred_img_ids[i] for i in sorted_indices]
        
        # Collect all ground truth boxes
        gt_boxes_by_img = {}
        total_gt = 0
        
        for tgt in targets:
            img_id = tgt['image_id']
            boxes = tgt['boxes']
            gt_boxes_by_img[img_id] = boxes
            total_gt += len(boxes)
        
        if total_gt == 0:
            return 0.0
        
        # Match predictions to ground truth
        tp = torch.zeros(len(all_pred_boxes))
        fp = torch.zeros(len(all_pred_boxes))
        
        # Keep track of matched GT boxes to avoid double matching
        matched_gt = {}
        for img_id in gt_boxes_by_img:
            matched_gt[img_id] = torch.zeros(len(gt_boxes_by_img[img_id]), dtype=torch.bool)
        
        for i, (pred_box, img_id) in enumerate(zip(all_pred_boxes, all_pred_img_ids)):
            if img_id not in gt_boxes_by_img:
                fp[i] = 1
                continue
                
            gt_boxes = gt_boxes_by_img[img_id]
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            # Compute IoU with all GT boxes in this image
            ious = self._compute_iou(pred_box.unsqueeze(0), gt_boxes)
            
            # Find best matching GT box
            max_iou, max_idx = torch.max(ious, dim=1)
            max_iou = max_iou.item()
            max_idx = max_idx.item()
            
            if max_iou >= iou_threshold and not matched_gt[img_id][max_idx]:
                tp[i] = 1
                matched_gt[img_id][max_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP using 11-point interpolation
        ap = self._compute_ap_interp(recalls.numpy(), precisions.numpy())
        return ap
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-8)
        return iou
    
    def _compute_ap_interp(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute AP using 11-point interpolation"""
        # Add sentinel values
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])
        
        # Make precision monotonic
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # Find points where recall changes
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        
        # Compute AP using trapezoidal rule
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap