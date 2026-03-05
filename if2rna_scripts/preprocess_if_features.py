#!/usr/bin/env python3
"""
Extract ResNet50/UNI features from IF image patches for IF2RNA

Adapted from SEQUOIA's compute_features_hdf5.py
Reads patches from HDF5 files and extracts feature vectors

Usage:
    python preprocess_if_features.py --ref_file data/metadata/if_reference.csv --feat_type resnet
    python preprocess_if_features.py --ref_file data/metadata/if_reference.csv --feat_type uni --uni_model_dir models/uni
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import sys

# Add sequoia to path
sys.path.insert(0, 'sequoia-pub')
from src.resnet import resnet50


def load_feature_extractor(feat_type, model_dir=None, device='cuda'):
    """
    Load feature extraction model (ResNet50 or UNI)
    """
    if feat_type == 'resnet':
        print("  Loading ResNet50...")
        transforms_val = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        
        # Load model
        model = resnet50(pretrained=False)
        
        # Load pretrained weights if provided
        if model_dir:
            model_path = Path(model_dir) / "resnet50_imagenet.pth"
            if model_path.exists():
                print(f"    Loading weights from: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                print(f"    Warning: Model file not found at {model_path}")
                print(f"    Using randomly initialized weights (not recommended!)")
        else:
            print(f"    Warning: No model_dir provided, using randomly initialized weights")
        
        model = model.to(device)
        model.eval()
        return model, transforms_val
    
    elif feat_type == 'uni':
        print("  Loading UNI model...")
        import timm
        
        transforms_val = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        
        if model_dir:
            model_path = Path(model_dir) / "pytorch_model.bin"
            if not model_path.exists():
                raise FileNotFoundError(f"UNI model not found at {model_path}")
            print(f"    Loading weights from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        else:
            print("    Warning: No model_dir provided for UNI")
        
        model.to(device)
        model.eval()
        return model, transforms_val
    
    else:
        raise ValueError(f"Unknown feature type: {feat_type}")


def extract_features_from_patches(patch_h5_path, model, transforms_val, feat_type, device):
    """
    Extract features from all patches in an HDF5 file
    
    Returns:
        numpy array of features (num_patches x feature_dim)
    """
    features_list = []
    
    try:
        with h5py.File(patch_h5_path, 'r') as f:
            patch_keys = [k for k in f.keys() if k.startswith('patch_')]
            
            for key in patch_keys:
                patch = f[key][:]
                
                if feat_type == 'resnet':
                    # ResNet expects (C, H, W)
                    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).to(device)
                    patch_tensor = transforms_val(patch_tensor)
                    
                    with torch.no_grad():
                        features = model.forward_extract(patch_tensor[None, :])
                        features_list.append(features[0].detach().cpu().numpy())
                
                elif feat_type == 'uni':
                    # UNI expects PIL Image
                    patch_img = Image.fromarray(patch).convert("RGB")
                    patch_tensor = transforms_val(patch_img).to(device)
                    
                    with torch.no_grad():
                        features = model(patch_tensor[None, :])
                        features_list.append(features[0].detach().cpu().numpy())
        
        if len(features_list) == 0:
            return None
        
        return np.array(features_list)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def process_sample(row, patch_dir, output_dir, model, transforms_val, feat_type, device):
    """
    Process a single sample: extract features from patches
    """
    wsi_name = row['wsi_file_name']
    organ = row['organ_type']
    
    # Find patch HDF5 file
    patch_h5_path = Path(patch_dir) / organ / wsi_name / f"{wsi_name}.h5"
    
    if not patch_h5_path.exists():
        return None, f"Patches not found: {wsi_name}"
    
    # Create output directory
    output_path = Path(output_dir) / organ / wsi_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    output_h5 = output_path / f"{wsi_name}.h5"
    if output_h5.exists():
        with h5py.File(output_h5, 'r') as f:
            if f'{feat_type}_features' in f.keys():
                return wsi_name, "Already processed"
    
    # Extract features
    features = extract_features_from_patches(
        patch_h5_path, model, transforms_val, feat_type, device
    )
    
    if features is None or len(features) == 0:
        return None, f"No features extracted: {wsi_name}"
    
    # Save features to HDF5
    try:
        with h5py.File(output_h5, 'a') as f:
            if f'{feat_type}_features' in f.keys():
                del f[f'{feat_type}_features']
            f.create_dataset(f'{feat_type}_features', data=features, compression='gzip')
        
        # Write completion marker
        with open(output_path / f"complete_{feat_type}.txt", 'w') as f:
            f.write(f"Features: {features.shape}\n")
        
        return wsi_name, features.shape[0]
        
    except Exception as e:
        return None, f"Error saving features for {wsi_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Extract features from IF image patches')
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--patch_dir', type=str, default='data/if_patches',
                       help='Directory containing patch HDF5 files')
    parser.add_argument('--output_dir', type=str, default='data/if_features',
                       help='Output directory for features')
    parser.add_argument('--feat_type', type=str, default='resnet',
                       choices=['resnet', 'uni'],
                       help='Feature extractor type')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Directory containing model weights (required for offline use)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start index for parallelization')
    parser.add_argument('--end', type=int, default=None,
                       help='End index for parallelization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading feature extractor...")
    model, transforms_val = load_feature_extractor(
        args.feat_type, args.model_dir, device
    )
    print("  ✓ Model loaded")
    
    # Read reference CSV
    print(f"\nReading reference file: {args.ref_file}")
    df = pd.read_csv(args.ref_file)
    print(f"  Total samples: {len(df)}")
    
    # Apply start/end indexing
    if args.start is not None and args.end is not None:
        df = df.iloc[args.start:args.end]
    elif args.start is not None:
        df = df.iloc[args.start:]
    elif args.end is not None:
        df = df.iloc[:args.end]
    
    print(f"  Processing: {len(df)} samples")
    
    print("\n" + "="*70)
    print(f"Extracting {args.feat_type.upper()} features from patches")
    print("="*70)
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result, message = process_sample(
            row, args.patch_dir, args.output_dir,
            model, transforms_val, args.feat_type, device
        )
        
        if result is None:
            error_count += 1
            if idx < 5:  # Only print first few errors
                print(f"\n  ✗ {message}")
        elif message == "Already processed":
            skip_count += 1
        else:
            success_count += 1
    
    print("\n" + "="*70)
    print("Feature extraction complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
