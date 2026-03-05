#!/usr/bin/env python3
"""
Apply K-means clustering to IF image features for IF2RNA

Adapted from SEQUOIA's kmean_features.py
Clusters extracted features into K representative prototypes per image

Usage:
    python preprocess_if_kmeans.py --ref_file data/metadata/if_reference.csv --feat_type resnet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch


def cluster_features(feature_h5_path, wsi_name, feat_type, num_clusters=100, seed=42):
    """
    Cluster features for a single sample
    
    Args:
        feature_h5_path: Path to HDF5 file with features
        wsi_name: Sample name
        feat_type: Feature type (resnet or uni)
        num_clusters: Number of clusters (default: 100)
        seed: Random seed
    
    Returns:
        cluster_features: (num_clusters x feature_dim) array of cluster centroids
    """
    try:
        with h5py.File(feature_h5_path, 'r') as f:
            if f'{feat_type}_features' not in f.keys():
                return None, f"Features not found: {feat_type}_features"
            
            features = f[f'{feat_type}_features'][:]
        
        # Check if we have enough patches
        if features.shape[0] < num_clusters:
            # If fewer patches than clusters, pad with duplicates
            num_repeats = int(np.ceil(num_clusters / features.shape[0]))
            features = np.tile(features, (num_repeats, 1))[:num_clusters]
        
        # Run K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
        kmeans.fit(features)
        
        # Get cluster centers (mean features for each cluster)
        cluster_centers = kmeans.cluster_centers_
        
        return cluster_centers, None
        
    except Exception as e:
        return None, f"Error clustering {wsi_name}: {e}"


def process_sample(row, feature_dir, feat_type, num_clusters, seed):
    """
    Process a single sample: cluster features
    """
    wsi_name = row['wsi_file_name']
    organ = row['organ_type']
    
    # Find feature HDF5 file
    feature_h5_path = Path(feature_dir) / organ / wsi_name / f"{wsi_name}.h5"
    
    if not feature_h5_path.exists():
        return None, f"Features not found: {wsi_name}"
    
    # Check if already clustered
    try:
        with h5py.File(feature_h5_path, 'r') as f:
            if 'cluster_features' in f.keys():
                return wsi_name, "Already clustered"
    except:
        pass
    
    # Cluster features
    cluster_centers, error = cluster_features(
        feature_h5_path, wsi_name, feat_type, num_clusters, seed
    )
    
    if cluster_centers is None:
        return None, error
    
    # Save cluster features to same HDF5 file
    try:
        with h5py.File(feature_h5_path, 'a') as f:
            if 'cluster_features' in f.keys():
                del f['cluster_features']
            f.create_dataset('cluster_features', data=cluster_centers, compression='gzip')
        
        return wsi_name, num_clusters
        
    except Exception as e:
        return None, f"Error saving clusters for {wsi_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Cluster IF image features with K-means')
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--feature_dir', type=str, default='data/if_features',
                       help='Directory containing feature HDF5 files')
    parser.add_argument('--feat_type', type=str, default='resnet',
                       choices=['resnet', 'uni'],
                       help='Feature type to cluster')
    parser.add_argument('--num_clusters', type=int, default=100,
                       help='Number of clusters per image (default: 100)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start index for parallelization')
    parser.add_argument('--end', type=int, default=None,
                       help='End index for parallelization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    print(f"Clustering {args.feat_type.upper()} features (K={args.num_clusters})")
    print("="*70)
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result, message = process_sample(
            row, args.feature_dir, args.feat_type, args.num_clusters, args.seed
        )
        
        if result is None:
            error_count += 1
            if idx < 5:  # Only print first few errors
                print(f"\n  ✗ {message}")
        elif message == "Already clustered":
            skip_count += 1
        else:
            success_count += 1
    
    print("\n" + "="*70)
    print("K-means clustering complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {args.feature_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
