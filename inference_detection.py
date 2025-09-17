#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Detection Inference Script

Usage:
    python inference_detection.py --checkpoint path/to/checkpoint.pth --image path/to/image.jpg
"""

import argparse
import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from train_detection import create_model, load_config


def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Detection Inference', add_help=False)
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    return parser


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def preprocess_image(image_path, max_size=1333):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    
    # Normalize
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # Simple resize for inference
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image, tensor


def visualize_predictions(image, predictions, output_path=None, conf_threshold=0.5):
    """Visualize detection predictions"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = predictions['boxes']
    scores = predictions['scores'] 
    labels = predictions['labels']
    
    # Filter by confidence threshold
    keep = scores > conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Draw bounding boxes
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add text label
        ax.text(x1, y1 - 5, f'Class {label.item()}: {score.item():.3f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10, color='black')
    
    ax.set_title(f'Detection Results (conf > {conf_threshold})')
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Results saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, config = load_model_from_checkpoint(args.checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Load and preprocess image
    print("Loading image...")
    image, tensor = preprocess_image(args.image)
    tensor = tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model([tensor.squeeze(0)])
    
    # Process outputs
    predictions = outputs[0]  # First (and only) image in batch
    
    print(f"Found {len(predictions['boxes'])} detections")
    print(f"Max confidence: {predictions['scores'].max().item():.3f}")
    
    # Convert to CPU for visualization
    predictions = {k: v.cpu() for k, v in predictions.items()}
    
    # Visualize results
    output_path = args.output or args.image.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    visualize_predictions(image, predictions, output_path, args.conf_threshold)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()