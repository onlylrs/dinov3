# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
import torchvision.transforms as transforms

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder


class CocoDetectionTargetDecoder(TargetDecoder):
    def __init__(self):
        pass
    
    def decode_target(self, target):
        return target


class CocoDetectionDataset(ExtendedVisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        transforms_fn=None,
    ):
        """
        COCO Detection Dataset
        
        Args:
            root: Root directory of images
            ann_file: Path to annotation file (train.json, val.json, test.json)
            transform: Image transforms
            target_transform: Target transforms
            transforms_fn: Custom transforms function
        """
        super().__init__(
            root=root,
            extra=ann_file,
            transform=transform,
            target_transform=target_transform,
        )
        
        self.root = Path(root)
        self.ann_file = ann_file
        self.transforms_fn = transforms_fn
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.img_to_anns]
        
        # Category mapping
        self.categories = {cat['id']: cat for cat in self.coco_data.get('categories', [])}
        self.num_classes = len(self.categories) if self.categories else 91
        
        # Setup decoders
        self.image_decoder = ImageDataDecoder()
        self.target_decoder = CocoDetectionTargetDecoder()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.root / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Convert annotations to target format
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([int(img_info['height']), int(img_info['width'])]),
            'size': torch.as_tensor([int(img_info['height']), int(img_info['width'])]),
        }
        
        # Apply transforms
        if self.transforms_fn:
            image, target = self.transforms_fn(image, target)
        elif self.transform:
            image = self.transform(image)
        
        return image, target
    
    def get_num_classes(self) -> int:
        return self.num_classes


def make_coco_transforms(image_set: str, max_size: int = 1333):
    """Create transforms for COCO detection"""
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    if image_set == 'train':
        return transforms.Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize(scales, max_size=max_size),
                transforms.Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, max_size),
                    RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    
    if image_set == 'val' or image_set == 'test':
        return transforms.Compose([
            RandomResize([800], max_size=max_size),
            normalize,
        ])
    
    raise ValueError(f'unknown image_set {image_set}')


class CocoTransforms:
    """Wrapper for applying transforms to both image and target"""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, target):
        for t in self.transforms:
            if hasattr(t, '__call__') and hasattr(t, 'transforms'):
                # This is a torchvision Compose
                image = t(image)
            else:
                # This is our custom transform
                image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            w = image.width if hasattr(image, 'width') else image.shape[-1]
            if 'boxes' in target:
                boxes = target['boxes']
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                target['boxes'] = boxes
        return image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = torch.tensor(self.sizes)[torch.randint(len(self.sizes), (1,))].item()
        return resize(image, target, size, self.max_size)


def resize(image, target, size, max_size=None):
    """Resize image and adjust bounding boxes accordingly"""
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
            
        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = transforms.functional.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


class RandomSelect:
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w = torch.randint(self.min_size, min(image.width, self.max_size) + 1, size=(1,)).item()
        h = torch.randint(self.min_size, min(image.height, self.max_size) + 1, size=(1,)).item()
        region = transforms.RandomCrop.get_params(image, [h, w])
        return crop(image, target, region)


def crop(image, target, region):
    cropped_image = transforms.functional.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target