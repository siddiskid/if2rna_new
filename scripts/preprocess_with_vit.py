#!/usr/bin/env python3
"""
SEQUOIA Preprocessing using UNI (Universal Histopathology Foundation Model)

UNI is a histopathology foundation model from Mahmood Lab trained on 100M images.
This uses UNI which outputs 1024 dims, matching the expected input for SEQUOIA models.

Usage:
    python preprocess_with_vit.py --ref_file data/metadata/reference.csv --feat_type uni
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_file', type=str, required=True)
    parser.add_argument('--wsi_path', type=str, default='data/raw/images')
    parser.add_argument('--patch_data_path', type=str, default='data/processed/patches')
    parser.add_argument('--feature_path', type=str, default='data/processed/features')
    parser.add_argument('--sequoia_dir', type=str, default='sequoia-pub')
    parser.add_argument('--max_patches', type=int, default=4000)
    parser.add_argument('--feat_type', type=str, default='uni', choices=['uni', 'resnet'],
                       help='Feature extractor: uni (recommended) or resnet')
    args = parser.parse_args()
    
    sequoia_dir = Path(args.sequoia_dir)
    patch_script = sequoia_dir / "pre_processing" / "patch_gen_hdf5.py"
    features_script = sequoia_dir / "pre_processing" / "compute_features_hdf5.py"
    
    # Create output directories
    Path(args.patch_data_path).mkdir(parents=True, exist_ok=True)
    Path(args.feature_path).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 1: Extract Patches")
    print("="*70)
    
    cmd1 = [
        "python", str(patch_script),
        "--ref_file", args.ref_file,
        "--wsi_path", args.wsi_path,
        "--patch_path", args.patch_data_path,
        "--mask_path", "data/processed/masks",
        "--patch_size", "256"
    ]
    
    print(f"\nRunning: {' '.join(cmd1)}\n")
    subprocess.run(cmd1, check=True)
    
    print("\n" + "="*70)
    if args.feat_type == 'uni':
        print("STEP 2: Extract Features with UNI")
        print("="*70)
        print("\n✓ Using UNI (Universal histopathology foundation model)")
        print("  Model: MahmoodLab/uni from Hugging Face")
        print("  Output: 1024-dimensional features\n")
        print("Note: You need to:")
        print("  1. Accept license at https://huggingface.co/MahmoodLab/uni")
        print("  2. Login via: huggingface-cli login\n")
    else:
        print("STEP 2: Extract Features with ResNet50")
        print("="*70)
        print("\n✓ Using ResNet50 (ImageNet pretrained)")
        print("  Output: 2048-dimensional features\n")
    
    cmd2 = [
        "python", str(features_script),
        "--feat_type", args.feat_type,
        "--ref_file", args.ref_file,
        "--patch_data_path", args.patch_data_path,
        "--feature_path", args.feature_path,
        "--max_patch_number", str(args.max_patches)
    ]
    
    print(f"\nRunning: {' '.join(cmd2)}\n")
    subprocess.run(cmd2, check=True)
    

if __name__ == "__main__":
    main()
