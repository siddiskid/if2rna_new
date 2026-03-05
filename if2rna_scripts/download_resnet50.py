#!/usr/bin/env python3
"""
Download pretrained ResNet50 model for offline use on HPC

Run this on a machine WITH internet access (your local machine),
then transfer the model file to the HPC.

Usage:
    python download_resnet50.py --output_dir models/resnet50
"""

import argparse
import torch
import torchvision.models as models
from pathlib import Path


def download_resnet50(output_dir):
    """Download pretrained ResNet50 and save weights"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ResNet50 pretrained on ImageNet...")
    model = models.resnet50(pretrained=True)
    
    # Save full model weights
    model_path = output_dir / "resnet50_imagenet.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"\n✓ Model downloaded successfully!")
    print(f"  Saved to: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1e6:.1f} MB")
    
    print(f"\nTo transfer to HPC:")
    print(f"  scp {model_path} username@sockeye.arc.ubc.ca:/path/to/if2rna_new/models/resnet50/")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Download ResNet50 model')
    parser.add_argument('--output_dir', type=str, default='models/resnet50',
                       help='Output directory for model weights')
    
    args = parser.parse_args()
    
    download_resnet50(args.output_dir)


if __name__ == '__main__':
    main()
