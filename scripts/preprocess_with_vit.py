#!/usr/bin/env python3
"""
SEQUOIA Preprocessing using ImageNet ViT-Large (stopgap until UNI access)

This uses standard ImageNet-pretrained ViT-Large which outputs 1024 dims,
matching the expected input for pretrained SEQUOIA models.

Usage:
    python preprocess_with_vit.py --ref_file data/metadata/reference.csv
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
    print("STEP 2: Extract Features with ImageNet ViT-Large")
    print("="*70)
    print("\n⚠️  Using ImageNet ViT-Large (1024 dims) as UNI substitute")
    print("   This is NOT histopathology-specific but allows testing")
    print("   the pretrained SEQUOIA pipeline.\n")
    
    # Modify the compute_features script to use standard ViT-Large
    cmd2 = [
        "python", str(features_script),
        "--feat_type", "vit_imagenet",  # Custom flag
        "--ref_file", args.ref_file,
        "--patch_data_path", args.patch_data_path,
        "--feature_path", args.feature_path,
        "--max_patch_number", str(args.max_patches)
    ]
    
    print(f"\nRunning: {' '.join(cmd2)}\n")
    print("Note: You'll need to modify compute_features_hdf5.py")
    print("      to support 'vit_imagenet' feat_type")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Use CTransPath (better for histopathology)")
    print("="*70)
    print("\n1. Download CTransPath model:")
    print("   https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view")
    print("\n2. Save to: models/ctranspath/")
    print("\n3. Modify compute_features_hdf5.py to load CTransPath")
    

if __name__ == "__main__":
    main()
