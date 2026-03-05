#!/usr/bin/env python3
"""
Validate that preprocessing is complete and features are ready for training

Usage:
    python if2rna_scripts/validate_preprocessing.py \
        --ref_file data/metadata/if_reference.csv \
        --feature_dir data/if_features
"""

import argparse
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Validate preprocessing completion')
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--feature_dir', type=str, default='data/if_features',
                       help='Feature directory')
    
    args = parser.parse_args()
    
    print("Loading reference file...")
    df = pd.read_csv(args.ref_file)
    print(f"Total samples: {len(df)}")
    
    missing_samples = []
    incomplete_samples = []
    valid_samples = []
    
    print("\nChecking features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        wsi_name = row['wsi_file_name']
        organ = row['organ_type']
        h5_path = Path(args.feature_dir) / organ / wsi_name / f"{wsi_name}.h5"
        
        if not h5_path.exists():
            missing_samples.append(wsi_name)
            continue
        
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'cluster_features' not in f.keys():
                    incomplete_samples.append(wsi_name)
                else:
                    features = f['cluster_features'][:]
                    valid_samples.append((wsi_name, features.shape))
        except Exception as e:
            incomplete_samples.append(wsi_name)
    
    print(f"\n{'='*70}")
    print("Validation Results:")
    print(f"{'='*70}")
    print(f"✓ Valid samples:      {len(valid_samples)}")
    print(f"✗ Missing files:      {len(missing_samples)}")
    print(f"✗ Incomplete samples: {len(incomplete_samples)}")
    print(f"{'='*70}")
    
    if len(valid_samples) > 0:
        print(f"\nFeature shapes (first 5):")
        for wsi, shape in valid_samples[:5]:
            print(f"  {wsi}: {shape}")
    
    if len(missing_samples) > 0:
        print(f"\nMissing samples (first 10):")
        for wsi in missing_samples[:10]:
            print(f"  {wsi}")
    
    if len(incomplete_samples) > 0:
        print(f"\nIncomplete samples (first 10):")
        for wsi in incomplete_samples[:10]:
            print(f"  {wsi}")
    
    if len(valid_samples) == len(df):
        print(f"\n{'='*70}")
        print("✓ All samples ready for training!")
        print(f"{'='*70}")
        print("\nTo start training:")
        print(f"  python if2rna_scripts/train_if2rna.py \\")
        print(f"      --ref_file {args.ref_file} \\")
        print(f"      --feature_dir {args.feature_dir} \\")
        print(f"      --save_dir results/if2rna_models \\")
        print(f"      --exp_name baseline_resnet \\")
        print(f"      --train")
    else:
        print(f"\n⚠ Only {len(valid_samples)}/{len(df)} samples ready")
        print("  Complete preprocessing before training")


if __name__ == '__main__':
    main()
