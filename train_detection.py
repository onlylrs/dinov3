#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Detection Fine-tuning Script

Usage:
    python train_detection.py --config configs/detection_finetune.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import math
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DINOv3 imports
import sys
sys.path.append('/home/jh/rs_work/dinov3')

from dinov3.data.datasets.coco_detection import CocoDetectionDataset, make_coco_transforms, CocoTransforms
from dinov3.eval.coco_evaluator import COCOEvaluator
from dinov3.train.detection_losses import DetectionCriterion
from dinov3.hub.detectors import _make_dinov3_detector
from dinov3.hub.backbones import Weights as BackboneWeights
from dinov3.logging import setup_logging

logger = logging.getLogger("dinov3")


def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Detection Fine-tuning', add_help=False)
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--output_dir', help='Override output directory')
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]):
    """Create DINOv3 detector model from config."""
    try:
        from dinov3.hub.detectors import _make_dinov3_detector
        
        model_config = config['model']
        dataset_config = config['dataset']
        
        # Special configuration for vit7b16 - use 8 layers instead of default 4
        backbone_name = model_config['backbone_name']
        if backbone_name == 'dinov3_vit7b16':
            # vit7b16: embed_dim=4096, depth=40
            # Detection head expects 32768 input features = 8 * 4096
            # So we need 8 layers: [4, 9, 14, 19, 24, 29, 34, 39]
            layers_to_use = [4, 9, 14, 19, 24, 29, 34, 39]  # 8 layers for 32768 features
        else:
            layers_to_use = None  # Use default 4 layers
        
        # Get backbone weights
        backbone_weights = None
        if model_config.get('pretrained_backbone', True):
            backbone_weights = model_config.get('backbone_weights')
            if backbone_weights and not backbone_weights.startswith('/'):
                # If it's a named weight, pass as string; otherwise use path
                pass
        
        # Get detector weights
        detector_weights = None
        if model_config.get('pretrained_detector', False):
            detector_weights = model_config.get('detector_weights')
        
        detector = _make_dinov3_detector(
            backbone_name=backbone_name,
            pretrained=model_config.get('pretrained_backbone', True),
            detector_weights=detector_weights,
            backbone_weights=backbone_weights,
            num_classes=dataset_config['num_classes'],
            check_hash=False,
            layers_to_use=layers_to_use
        )
        
        return detector
        
    except Exception as e:
        print(f"Error creating model: {e}")
        raise


def create_datasets(config: Dict[str, Any]):
    """Create train/val/test datasets"""
    dataset_config = config['dataset']
    
    # Create transforms
    train_transforms = CocoTransforms(make_coco_transforms('train', max_size=dataset_config.get('max_size', 1333)))
    val_transforms = CocoTransforms(make_coco_transforms('val', max_size=dataset_config.get('max_size', 1333)))
    
    # Create datasets
    train_dataset = CocoDetectionDataset(
        root=dataset_config['root'],
        ann_file=dataset_config['train_ann'],
        transforms_fn=train_transforms
    )
    
    val_dataset = CocoDetectionDataset(
        root=dataset_config['root'],
        ann_file=dataset_config['val_ann'],
        transforms_fn=val_transforms
    )
    
    test_dataset = None
    if 'test_ann' in dataset_config and dataset_config['test_ann']:
        test_dataset = CocoDetectionDataset(
            root=dataset_config['root'],
            ann_file=dataset_config['test_ann'],
            transforms_fn=val_transforms
        )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(datasets, config: Dict[str, Any]):
    """Create data loaders"""
    train_dataset, val_dataset, test_dataset = datasets
    batch_size = config['training']['batch_size']
    num_workers = config['device'].get('num_workers', 4)
    pin_memory = config['device'].get('pin_memory', True)
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader


def create_criterion(config: Dict[str, Any]) -> nn.Module:
    """Create detection criterion"""
    training_config = config['training']
    
    criterion = DetectionCriterion(
        num_classes=config['dataset']['num_classes'],
        matcher_cost_class=training_config['matcher_costs']['cost_class'],
        matcher_cost_bbox=training_config['matcher_costs']['cost_bbox'],
        matcher_cost_giou=training_config['matcher_costs']['cost_giou'],
        weight_dict=training_config['loss_weights'],
        eos_coef=training_config['eos_coef'],
        losses=['labels', 'boxes']
    )
    
    return criterion


def create_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]):
    """Create optimizer and learning rate scheduler"""
    training_config = config['training']
    
    # Only optimize parameters that require gradients (detection head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Cosine annealing scheduler with warmup
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()
    criterion.train()
    
    log_interval = config['logging']['log_interval']
    total_loss = 0
    num_samples = 0
    
    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model.detector(images)
        loss_dict = criterion(outputs, targets)
        
        # Combine losses
        losses = sum(loss_dict[k] * config['training']['loss_weights'].get(k.replace('_0', ''), 1.0) 
                    for k in loss_dict.keys() if 'loss' in k)
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Statistics
        total_loss += losses.item()
        num_samples += len(images)
        
        # Logging
        if i % log_interval == 0:
            logger.info(
                f"Epoch [{epoch}][{i}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    avg_loss = total_loss / len(data_loader)
    logger.info(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    """Evaluate model"""
    model.eval()
    evaluator = COCOEvaluator()
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Convert outputs to evaluator format
        predictions = []
        for i, output in enumerate(outputs):
            pred = {
                'boxes': output['boxes'],
                'scores': output['scores'],
                'labels': output['labels']
            }
            predictions.append(pred)
        
        evaluator.update(predictions, targets)
    
    # Compute metrics
    metrics = evaluator.compute()
    
    # Log results
    logger.info(f"Evaluation Results:")
    for metric_name, value in metrics.items():
        if not metric_name.startswith('class_'):
            logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, filename):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, filename)
    logger.info(f"Saved checkpoint: {filename}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    return checkpoint['epoch'], checkpoint.get('metrics', {})


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = args.output_dir or config['logging']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output=os.path.join(output_dir, "train.log"))
    logger.info(f"Config: {config}")
    
    # Device setup
    device = torch.device(f"cuda:{config['device']['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create datasets and data loaders
    datasets = create_datasets(config)
    data_loaders = create_data_loaders(datasets, config)
    train_loader, val_loader, test_loader = data_loaders
    
    logger.info(f"Dataset sizes - Train: {len(datasets[0])}, Val: {len(datasets[1])}")
    
    # Create criterion, optimizer, and scheduler
    criterion = create_criterion(config)
    criterion.to(device)
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    best_metric_name = config['evaluation']['save_best_metric']
    
    if args.resume:
        start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_metric = metrics.get(best_metric_name, 0.0)
        start_epoch += 1
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        if val_loader:
            metrics = evaluate(model, val_loader, device, config)
        if test_loader:
            logger.info("Evaluating on test set...")
            test_metrics = evaluate(model, test_loader, device, config)
            logger.info(f"Test metrics: {test_metrics}")
        return
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    eval_interval = config['evaluation']['eval_interval']
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, config)
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            metrics = evaluate(model, val_loader, device, config)
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, checkpoint_path)
            
            # Save best checkpoint
            current_metric = metrics.get(best_metric_name, 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, best_checkpoint_path)
                logger.info(f"New best {best_metric_name}: {best_metric:.4f}")
    
    logger.info(f"Training completed. Best {best_metric_name}: {best_metric:.4f}")


if __name__ == '__main__':
    main()