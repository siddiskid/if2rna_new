#!/usr/bin/env python3
"""
Train IF2RNA model using SEQUOIA's ViS architecture

Adapted from sequoia-pub/src/main.py for IF imaging data

Usage:
    python if2rna_scripts/train_if2rna.py \
        --ref_file data/metadata/if_reference.csv \
        --feature_dir data/if_features \
        --save_dir results/if2rna_models \
        --exp_name baseline_resnet \
        --train
"""

import os
import argparse
from tqdm import tqdm
import pickle
import h5py
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

import sys
sys.path.insert(0, 'sequoia-pub/src')
sys.path.insert(0, 'sequoia-pub')
from src.vit import train, evaluate
from src.tformer_lin import ViS


class IFRNADataset(Dataset):
    """Dataset for IF images and RNA expression"""
    
    def __init__(self, df, feature_dir):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        
        # Get number of genes from first sample
        row = self.df.iloc[0]
        rna_cols = [x for x in row.keys() if 'rna_' in x]
        self.num_genes = len(rna_cols)
        
        # Get feature dimension from first sample
        sample_id = row['wsi_file_name']
        organ = row['organ_type']
        h5_path = os.path.join(self.feature_dir, organ, sample_id, f"{sample_id}.h5")
        with h5py.File(h5_path, 'r') as f:
            features = f['cluster_features'][:]
            self.feature_dim = features.shape[1]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['wsi_file_name']
        organ = row['organ_type']
        
        # Load features
        h5_path = os.path.join(self.feature_dir, organ, sample_id, f"{sample_id}.h5")
        try:
            with h5py.File(h5_path, 'r') as f:
                features = torch.tensor(f['cluster_features'][:], dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
            features = None
        
        # Load RNA expression
        rna_cols = [x for x in row.keys() if 'rna_' in x]
        rna_data = torch.tensor(row[rna_cols].values.astype(np.float32), dtype=torch.float32)
        
        return features, rna_data, sample_id, row['patient_id']


def custom_collate_fn(batch):
    """Remove samples with missing features"""
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def patient_kfold(df, n_splits=5, random_state=42, valid_size=0.1):
    """Perform cross-validation with patient-level split"""
    indices = np.arange(len(df))
    patients_unique = df['patient_id'].unique()
    
    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    
    train_idx = []
    valid_idx = []
    test_idx = []
    
    for k, (ind_train, ind_test) in enumerate(skf.split(patients_unique)):
        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]
        
        # Get test indices
        test_mask = df['patient_id'].isin(patients_test)
        test_idx.append(indices[test_mask])
        
        # Split train into train/val if needed
        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=random_state)
            valid_mask = df['patient_id'].isin(patients_valid)
            valid_idx.append(indices[valid_mask])
        
        train_mask = df['patient_id'].isin(patients_train)
        train_idx.append(indices[train_mask])
    
    return train_idx, valid_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description='Train IF2RNA model')
    
    # Data paths
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV with IF images and RNA data')
    parser.add_argument('--feature_dir', type=str, default='data/if_features',
                       help='Directory with preprocessed features')
    parser.add_argument('--save_dir', type=str, default='results/if2rna_models',
                       help='Directory to save trained models')
    parser.add_argument('--exp_name', type=str, default='if2rna_baseline',
                       help='Experiment name')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='vis',
                       choices=['vis', 'vit'],
                       help='Model architecture (vis=linearized transformer)')
    parser.add_argument('--depth', type=int, default=6,
                       help='Transformer depth')
    parser.add_argument('--num_heads', type=int, default=16,
                       help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--train', action='store_true',
                       help='Train the model (if not set, only evaluates)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_on', type=str, default='loss',
                       choices=['loss', 'loss+corr'],
                       help='Metric for saving best model')
    parser.add_argument('--stop_on', type=str, default='loss',
                       choices=['loss', 'loss+corr'],
                       help='Metric for early stopping')
    
    # Optional
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming/finetuning')
    parser.add_argument('--sample_percent', type=float, default=None,
                       help='Downsample data fraction (for testing)')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Reproducibility for DataLoader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Load reference data
    print(f"\nLoading reference file: {args.ref_file}")
    df = pd.read_csv(args.ref_file)
    
    if args.sample_percent is not None:
        df = df.sample(frac=args.sample_percent, random_state=args.seed).reset_index(drop=True)
        print(f"Downsampled to {len(df)} samples ({args.sample_percent*100}%)")
    
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    # Get number of genes
    rna_cols = [c for c in df.columns if c.startswith('rna_')]
    num_genes = len(rna_cols)
    print(f"Number of genes: {num_genes}")
    
    # Cross-validation splits
    train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k, random_state=args.seed)
    
    test_results_splits = {}
    
    for fold in range(args.k):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{args.k}")
        print(f"{'='*70}")
        
        # Split data
        train_df = df.iloc[train_idxs[fold]]
        val_df = df.iloc[val_idxs[fold]]
        test_df = df.iloc[test_idxs[fold]]
        
        print(f"Train: {len(train_df)} samples, {train_df['patient_id'].nunique()} patients")
        print(f"Val:   {len(val_df)} samples, {val_df['patient_id'].nunique()} patients")
        print(f"Test:  {len(test_df)} samples, {test_df['patient_id'].nunique()} patients")
        
        # Save split patient IDs
        np.save(os.path.join(save_dir, f'train_{fold}.npy'), train_df['patient_id'].unique())
        np.save(os.path.join(save_dir, f'val_{fold}.npy'), val_df['patient_id'].unique())
        np.save(os.path.join(save_dir, f'test_{fold}.npy'), test_df['patient_id'].unique())
        
        # Create datasets
        train_dataset = IFRNADataset(train_df, args.feature_dir)
        val_dataset = IFRNADataset(val_df, args.feature_dir)
        test_dataset = IFRNADataset(test_df, args.feature_dir)
        
        feature_dim = train_dataset.feature_dim
        print(f"Feature dimension: {feature_dim}")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Initialize model
        print(f"\nInitializing {args.model_type.upper()} model...")
        if args.model_type == 'vis':
            model = ViS(
                num_outputs=num_genes,
                input_dim=feature_dim,
                depth=args.depth,
                nheads=args.num_heads,
                dimensions_f=64,
                dimensions_c=64,
                dimensions_s=64,
                device=device
            )
        else:  # vit
            from src.vit import ViT
            model = ViT(
                num_outputs=num_genes,
                dim=feature_dim,
                depth=args.depth,
                heads=args.num_heads,
                mlp_dim=2048,
                dim_head=64,
                device=device
            )
        
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            if fold > 0:
                checkpoint_path = checkpoint_path.replace('.pt', f'_{fold}.pt')
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        model.to(device)
        
        # Training
        if args.train:
            print(f"\nTraining for {args.num_epochs} epochs...")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                amsgrad=False,
                weight_decay=0.0
            )
            
            dataloaders = {'train': train_dataloader, 'val': val_dataloader}
            
            model = train(
                model,
                dataloaders,
                optimizer,
                num_epochs=args.num_epochs,
                run=None,
                split=fold,
                save_on=args.save_on,
                stop_on=args.stop_on,
                delta=0.5,
                save_dir=save_dir
            )
        
        # Evaluation on test set
        print(f"\nEvaluating on test set...")
        preds, real, sample_ids, patient_ids = evaluate(
            model, 
            test_dataloader, 
            run=None, 
            suff=f'_{fold}'
        )
        
        # Random baseline
        print("Evaluating random baseline...")
        if args.model_type == 'vis':
            random_model = ViS(
                num_outputs=num_genes,
                input_dim=feature_dim,
                depth=args.depth,
                nheads=args.num_heads,
                dimensions_f=64,
                dimensions_c=64,
                dimensions_s=64,
                device=device
            )
        else:
            from src.vit import ViT
            random_model = ViT(
                num_outputs=num_genes,
                dim=feature_dim,
                depth=args.depth,
                heads=args.num_heads,
                mlp_dim=2048,
                dim_head=64,
                device=device
            )
        random_model.to(device)
        random_preds, _, _, _ = evaluate(
            random_model,
            test_dataloader,
            run=None,
            suff=f'_{fold}_rand'
        )
        
        # Store results
        test_results = {
            'real': real,
            'preds': preds,
            'random': random_preds,
            'sample_ids': sample_ids,
            'patient_ids': patient_ids
        }
        test_results_splits[f'split_{fold}'] = test_results
    
    # Save all results
    test_results_splits['genes'] = [c.replace('rna_', '') for c in rna_cols]
    results_path = os.path.join(save_dir, 'test_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
