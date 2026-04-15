#!/usr/bin/env python3
"""
Run complete IF2RNA preprocessing pipeline

This script runs all preprocessing steps:
1. Extract patches from IF images
2. Extract ResNet50/UNI features from patches
3. Cluster features with K-means

Usage:
    # Full pipeline with ResNet (requires downloaded model)
    python run_if_preprocessing.py \
        --ref_file data/metadata/if_reference.csv \
        --feat_type resnet \
        --model_dir models/resnet50
    
    # Full pipeline with UNI (requires UNI model)
    python run_if_preprocessing.py \
        --ref_file data/metadata/if_reference.csv \
        --feat_type uni \
        --model_dir models/uni
    
    # Run specific steps only
    python run_if_preprocessing.py \
        --ref_file data/metadata/if_reference.csv \
        --feat_type resnet \
        --model_dir models/resnet50 \
        --steps patches features
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, step_name):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"[{step_name}]")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error in {step_name}")
        sys.exit(1)
    
    print(f"\n✓ {step_name} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run IF2RNA preprocessing pipeline')
    
    # Data paths
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--image_dir', type=str, default='data/if_images',
                       help='Directory with IF images')
    parser.add_argument('--patch_dir', type=str, default='data/if_patches',
                       help='Output directory for patches')
    parser.add_argument('--feature_dir', type=str, default='data/if_features',
                       help='Output directory for features')
    
    # Processing parameters
    parser.add_argument('--feat_type', type=str, default='resnet',
                       choices=['resnet', 'uni'],
                       help='Feature extractor type')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Directory with model weights (required for offline HPC use)')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size')
    parser.add_argument('--max_patches', type=int, default=100,
                       help='Max patches per image')
    parser.add_argument('--num_clusters', type=int, default=100,
                       help='Number of K-means clusters')
    parser.add_argument('--no_duplicate_patches', action='store_true',
                       help='Do not duplicate patches when fewer than max_patches are available')
    
    # Pipeline control
    parser.add_argument('--steps', nargs='+',
                       default=['patches', 'features', 'kmeans'],
                       choices=['patches', 'features', 'kmeans'],
                       help='Which steps to run')
    
    # Parallelization
    parser.add_argument('--start', type=int, default=None,
                       help='Start index for parallelization')
    parser.add_argument('--end', type=int, default=None,
                       help='End index for parallelization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate model directory for offline use
    if args.feat_type == 'resnet' and not args.model_dir:
        print("Warning: No --model_dir specified for ResNet50.")
        print("  On HPC without internet, this will fail!")
        print("  Download model first: python if2rna_scripts/download_resnet50.py")
        print("  Then specify: --model_dir models/resnet50")
    
    if args.feat_type == 'uni' and not args.model_dir:
        print("Error: --model_dir required when using feat_type=uni")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("IF2RNA Preprocessing Pipeline")
    print("="*70)
    print(f"Reference file: {args.ref_file}")
    print(f"Feature type: {args.feat_type}")
    print(f"Steps: {', '.join(args.steps)}")
    print("="*70)
    
    # Step 1: Patch extraction
    if 'patches' in args.steps:
        cmd = [
            'python', 'if2rna_scripts/preprocess_if_patches.py',
            '--ref_file', args.ref_file,
            '--image_dir', args.image_dir,
            '--output_dir', args.patch_dir,
            '--patch_size', str(args.patch_size),
            '--max_patches', str(args.max_patches),
            '--seed', str(args.seed)
        ]

        if args.no_duplicate_patches:
            cmd.append('--no_duplicate_patches')
        
        if args.start is not None:
            cmd.extend(['--start', str(args.start)])
        if args.end is not None:
            cmd.extend(['--end', str(args.end)])
        
        run_command(cmd, "Step 1: Patch Extraction")
    
    # Step 2: Feature extraction
    if 'features' in args.steps:
        cmd = [
            'python', 'if2rna_scripts/preprocess_if_features.py',
            '--ref_file', args.ref_file,
            '--patch_dir', args.patch_dir,
            '--output_dir', args.feature_dir,
            '--feat_type', args.feat_type,
            '--seed', str(args.seed)
        ]
        
        if args.model_dir:
            cmd.extend(['--model_dir', args.model_dir])
        
        if args.start is not None:
            cmd.extend(['--start', str(args.start)])
        if args.end is not None:
            cmd.extend(['--end', str(args.end)])
        
        run_command(cmd, f"Step 2: {args.feat_type.upper()} Feature Extraction")
    
    # Step 3: K-means clustering
    if 'kmeans' in args.steps:
        cmd = [
            'python', 'if2rna_scripts/preprocess_if_kmeans.py',
            '--ref_file', args.ref_file,
            '--feature_dir', args.feature_dir,
            '--feat_type', args.feat_type,
            '--num_clusters', str(args.num_clusters),
            '--seed', str(args.seed)
        ]
        
        if args.start is not None:
            cmd.extend(['--start', str(args.start)])
        if args.end is not None:
            cmd.extend(['--end', str(args.end)])
        
        run_command(cmd, "Step 3: K-means Clustering")
    
    print("\n" + "="*70)
    print("✓ Preprocessing pipeline complete!")
    print("="*70)
    print(f"\nFeatures ready for training:")
    print(f"  Feature directory: {args.feature_dir}")
    print(f"  Each sample has 'cluster_features' dataset (100 x feature_dim)")
    print(f"\nNext step: Train IF2RNA model")
    print(f"  python if2rna_scripts/train_if2rna.py --ref_file {args.ref_file} --feature_dir {args.feature_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
