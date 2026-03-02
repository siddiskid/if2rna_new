#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import argparse

def fix_and_evaluate_fold(predictions_file, gene_list_file, reference_file, output_dir):
    """
    Fix prediction column names and calculate correlations with ground truth.
    """
    fold_name = Path(predictions_file).stem
    print(f"\n{'='*80}")
    print(f"Processing {fold_name}")
    print(f"{'='*80}")
    
    print("Loading gene list...")
    gene_list = pd.read_csv(gene_list_file)
    genes = gene_list['gene'].tolist()
    print(f"  {len(genes)} genes")
    
    print("Loading predictions...")
    predictions = pd.read_csv(predictions_file)
    print(f"  Shape: {predictions.shape}")
    
    metadata_cols = ['wsi_file_name', 'patient_id', 'tcga_project']
    pred_cols = [col for col in predictions.columns if col.startswith('pred_gene_')]
    
    column_mapping = {}
    for i, pred_col in enumerate(pred_cols):
        if i < len(genes):
            column_mapping[pred_col] = f"rna_{genes[i]}"
    
    print(f"Renaming {len(column_mapping)} columns...")
    predictions = predictions.rename(columns=column_mapping)
    
    fixed_pred_file = Path(output_dir) / f"{fold_name}_fixed.csv"
    predictions.to_csv(fixed_pred_file, index=False)
    print(f"  Saved fixed predictions to {fixed_pred_file}")
    
    print("Loading reference data...")
    reference = pd.read_csv(reference_file)
    print(f"  Shape: {reference.shape}")
    
    merged = predictions.merge(
        reference,
        on='wsi_file_name',
        how='inner',
        suffixes=('_pred', '_true')
    )
    print(f"Merged data shape: {merged.shape}")
    print(f"  {len(merged)} samples matched")
    
    gene_cols = [col for col in predictions.columns if col.startswith('rna_')]
    common_genes = [gene for gene in gene_cols if gene in reference.columns]
    print(f"  {len(common_genes)} common genes found")
    
    if len(common_genes) == 0:
        print("ERROR: No common genes found!")
        return None
    
    print("\nCalculating per-gene correlations...")
    correlations = []
    for gene in common_genes:
        pred_col = f"{gene}_pred"
        true_col = f"{gene}_true"
        
        if pred_col in merged.columns and true_col in merged.columns:
            pred_values = merged[pred_col].dropna()
            true_values = merged[true_col].dropna()
            
            valid_indices = pred_values.index.intersection(true_values.index)
            if len(valid_indices) > 1:
                r, p = pearsonr(pred_values.loc[valid_indices], true_values.loc[valid_indices])
                correlations.append({
                    'gene': gene,
                    'correlation': r,
                    'p_value': p,
                    'n_samples': len(valid_indices)
                })
    
    correlations_df = pd.DataFrame(correlations)
    correlations_df = correlations_df.sort_values('correlation', ascending=False)
    
    corr_file = Path(output_dir) / f"{fold_name}_correlations.csv"
    correlations_df.to_csv(corr_file, index=False)
    print(f"  Saved correlations to {corr_file}")
    
    print("\nCorrelation Statistics:")
    print(f"  Mean correlation: {correlations_df['correlation'].mean():.4f}")
    print(f"  Median correlation: {correlations_df['correlation'].median():.4f}")
    print(f"  Std correlation: {correlations_df['correlation'].std():.4f}")
    print(f"  Max correlation: {correlations_df['correlation'].max():.4f}")
    print(f"  Min correlation: {correlations_df['correlation'].min():.4f}")
    
    significant = correlations_df[correlations_df['p_value'] < 0.05]
    print(f"  Significant (p<0.05): {len(significant)} / {len(correlations_df)} ({100*len(significant)/len(correlations_df):.1f}%)")
    
    print("\nTop 10 correlated genes:")
    print(correlations_df.head(10)[['gene', 'correlation', 'p_value']].to_string(index=False))
    
    return correlations_df

def main():
    parser = argparse.ArgumentParser(description='Fix prediction columns and evaluate correlations')
    parser.add_argument('--predictions_dir', default='results', help='Directory with prediction files')
    parser.add_argument('--gene_list', default='models/gene_list.csv', help='Gene list file')
    parser.add_argument('--reference', default='data/metadata/reference.csv', help='Reference file with ground truth')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='Fold numbers to process')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_correlations = {}
    for fold in args.folds:
        predictions_file = Path(args.predictions_dir) / f"predictions_brca-{fold}.csv"
        
        if not predictions_file.exists():
            print(f"WARNING: {predictions_file} not found, skipping")
            continue
        
        correlations = fix_and_evaluate_fold(
            predictions_file,
            args.gene_list,
            args.reference,
            args.output_dir
        )
        
        if correlations is not None:
            all_correlations[f"brca-{fold}"] = correlations
    
    if len(all_correlations) > 0:
        print(f"\n\n{'='*80}")
        print("ENSEMBLE ANALYSIS")
        print(f"{'='*80}")
        
        print(f"Successfully processed {len(all_correlations)} folds")
        
        print("\nPer-fold mean correlations:")
        for fold_name, corr_df in all_correlations.items():
            mean_corr = corr_df['correlation'].mean()
            print(f"  {fold_name}: {mean_corr:.4f}")
        
        all_mean = np.mean([corr_df['correlation'].mean() for corr_df in all_correlations.values()])
        print(f"\nOverall mean correlation across all folds: {all_mean:.4f}")

if __name__ == '__main__':
    main()
