"""
Evaluate trained IF2RNA model on test set.

Analyzes test_results.pkl to compute:
- Per-gene Pearson correlations
- Per-organ performance
- Top/bottom performing genes
- Visualization of predictions vs actual expression
"""

import os
import warnings
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
os.environ['FONTCONFIG_PATH'] = '/tmp'
warnings.filterwarnings('ignore', category=UserWarning)

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse


def compute_gene_correlations(predictions, actuals, gene_names):
    """Compute Pearson correlation for each gene."""
    correlations = []
    p_values = []
    
    for i in range(predictions.shape[1]):
        pred = predictions[:, i]
        actual = actuals[:, i]
        
        # Remove NaN values
        mask = ~(np.isnan(pred) | np.isnan(actual))
        if mask.sum() > 1:
            r, p = stats.pearsonr(pred[mask], actual[mask])
            correlations.append(r)
            p_values.append(p)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    return np.array(correlations), np.array(p_values)


def analyze_by_organ(results, reference_df):
    """Analyze performance by organ type."""
    organ_results = {}
    
    # Iterate through splits
    for split_key, split_data in results.items():
        if not split_key.startswith('split_'):
            continue
            
        # Get test sample IDs
        sample_ids = split_data.get('sample_ids', [])
        predictions = split_data.get('preds', np.array([]))
        actuals = split_data.get('real', np.array([]))  # Changed from 'y' to 'real'
        
        # Match to organ
        for idx, sample_id in enumerate(sample_ids):
            # Extract organ from sample_id (first part before underscore)
            parts = sample_id.split('_')
            # Handle special case: mu/hu_organ_... format
            if len(parts) > 1 and parts[0] in ['mu', 'hu']:
                organ = parts[1]
            else:
                organ = parts[0] if parts else 'unknown'
            
            if organ not in organ_results:
                organ_results[organ] = {'predictions': [], 'actuals': []}
            
            if idx < len(predictions):
                organ_results[organ]['predictions'].append(predictions[idx])
                organ_results[organ]['actuals'].append(actuals[idx])
    
    # Compute statistics per organ
    organ_stats = {}
    for organ, data in organ_results.items():
        if len(data['predictions']) > 0:
            pred = np.vstack(data['predictions'])
            actual = np.vstack(data['actuals'])
            
            mae = np.mean(np.abs(pred - actual))
            mse = np.mean((pred - actual) ** 2)
            
            organ_stats[organ] = {
                'n_samples': len(data['predictions']),
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse)
            }
    
    return organ_stats


def plot_top_genes(predictions, actuals, correlations, gene_names, output_dir, top_n=16):
    """Plot predicted vs actual for top correlated genes."""
    # Get top genes by correlation
    valid_mask = ~np.isnan(correlations)
    valid_corr = correlations[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    top_indices = valid_indices[np.argsort(valid_corr)[-top_n:]][::-1]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, gene_idx in enumerate(top_indices):
        ax = axes[idx]
        
        pred = predictions[:, gene_idx]
        actual = actuals[:, gene_idx]
        
        # Remove NaN
        mask = ~(np.isnan(pred) | np.isnan(actual))
        pred_clean = pred[mask]
        actual_clean = actual[mask]
        
        ax.scatter(actual_clean, pred_clean, alpha=0.5, s=20)
        
        # Add identity line
        min_val = min(actual_clean.min(), pred_clean.min())
        max_val = max(actual_clean.max(), pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Actual Expression (log1p)')
        ax.set_ylabel('Predicted Expression (log1p)')
        ax.set_title(f'{gene_names[gene_idx]}\nr = {correlations[gene_idx]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_genes_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved top genes plot to {output_dir / 'top_genes_predictions.png'}")


def plot_correlation_distribution(correlations, output_dir):
    """Plot distribution of gene correlations."""
    valid_corr = correlations[~np.isnan(correlations)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(valid_corr, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(valid_corr), color='r', linestyle='--', 
                    label=f'Median: {np.median(valid_corr):.3f}')
    axes[0].axvline(np.mean(valid_corr), color='g', linestyle='--', 
                    label=f'Mean: {np.mean(valid_corr):.3f}')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Number of Genes')
    axes[0].set_title('Distribution of Gene Correlations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CDF
    sorted_corr = np.sort(valid_corr)
    axes[1].plot(sorted_corr, np.arange(len(sorted_corr)) / len(sorted_corr))
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5, label='r = 0')
    axes[1].axhline(0.5, color='g', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Pearson Correlation')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distribution to {output_dir / 'correlation_distribution.png'}")


def plot_organ_performance(organ_stats, output_dir):
    """Plot performance metrics by organ."""
    if not organ_stats:
        print("No organ statistics to plot")
        return
    
    organs = list(organ_stats.keys())
    n_samples = [organ_stats[o]['n_samples'] for o in organs]
    maes = [organ_stats[o]['mae'] for o in organs]
    rmses = [organ_stats[o]['rmse'] for o in organs]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sample counts
    axes[0].bar(organs, n_samples, edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Number of Test Samples')
    axes[0].set_title('Test Samples by Organ')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE
    axes[1].bar(organs, maes, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('MAE by Organ')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    axes[2].bar(organs, rmses, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_ylabel('Root Mean Squared Error')
    axes[2].set_title('RMSE by Organ')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'organ_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved organ performance plot to {output_dir / 'organ_performance.png'}")


def plot_fold_performance(results, output_dir):
    """Plot performance across folds."""
    fold_keys = [k for k in results.keys() if k.startswith('split_')]
    n_folds = len(fold_keys)
    fold_maes = []
    fold_corr_medians = []
    
    for fold_key in sorted(fold_keys):
        fold_data = results[fold_key]
        pred = fold_data.get('preds', np.array([]))
        actual = fold_data.get('real', np.array([]))  # Changed from 'y' to 'real'
        
        if len(pred) == 0 or len(actual) == 0:
            continue
        
        # Compute MAE
        mae = np.mean(np.abs(pred - actual))
        fold_maes.append(mae)
        
        # Compute median correlation
        gene_corrs = []
        for i in range(pred.shape[1]):
            mask = ~(np.isnan(pred[:, i]) | np.isnan(actual[:, i]))
            if mask.sum() > 1:
                r, _ = stats.pearsonr(pred[mask, i], actual[mask, i])
                gene_corrs.append(r)
        fold_corr_medians.append(np.median(gene_corrs) if gene_corrs else 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MAE by fold
    axes[0].bar(range(n_folds), fold_maes, edgecolor='black', alpha=0.7)
    axes[0].axhline(np.mean(fold_maes), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(fold_maes):.3f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Test MAE')
    axes[0].set_title('Test MAE by Fold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Median correlation by fold
    axes[1].bar(range(n_folds), fold_corr_medians, edgecolor='black', alpha=0.7, color='green')
    axes[1].axhline(np.mean(fold_corr_medians), color='r', linestyle='--',
                    label=f'Mean: {np.mean(fold_corr_medians):.3f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Median Gene Correlation')
    axes[1].set_title('Median Correlation by Fold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fold_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved fold performance plot to {output_dir / 'fold_performance.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate IF2RNA model')
    parser.add_argument('--results_file', type=str, 
                        default='results/if2rna_models/baseline_resnet_log/test_results.pkl',
                        help='Path to test_results.pkl')
    parser.add_argument('--reference_file', type=str,
                        default='data/metadata/if_reference.csv',
                        help='Path to reference CSV')
    parser.add_argument('--output_dir', type=str,
                        default='results/if2rna_models/baseline_resnet_log/evaluation',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IF2RNA Model Evaluation")
    print("=" * 70)
    
    # Load results
    print(f"\nLoading results from {args.results_file}...")
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\nResults structure:")
    print(f"  Type: {type(results)}")
    print(f"  Keys: {list(results.keys())}")
    
    # Load reference file for organ info
    print(f"\nLoading reference from {args.reference_file}...")
    reference_df = pd.read_csv(args.reference_file, low_memory=False)
    
    # Get gene names from results
    gene_names = results.get('genes', [])
    
    # Count folds
    fold_keys = [k for k in results.keys() if k.startswith('split_')]
    n_folds = len(fold_keys)
    
    print(f"\nNumber of genes: {len(gene_names)}")
    print(f"Number of folds: {n_folds}")
    
    # Debug: check first fold structure
    if fold_keys:
        first_key = fold_keys[0]
        print(f"\nFirst fold ({first_key}) structure:")
        print(f"  Keys: {list(results[first_key].keys())}")
        for k, v in results[first_key].items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: list of length {len(v)}")
            else:
                print(f"  {k}: {type(v)}")
    
    # Aggregate predictions and actuals across all folds
    all_predictions = []
    all_actuals = []
    
    for fold_key in sorted(fold_keys):
        fold_data = results[fold_key]
        fold_pred = fold_data.get('preds', np.array([]))
        fold_actual = fold_data.get('real', np.array([]))  # Changed from 'y' to 'real'
        
        if len(fold_pred) > 0 and len(fold_actual) > 0:
            all_predictions.append(fold_pred)
            all_actuals.append(fold_actual)
            print(f"  {fold_key}: {fold_pred.shape[0]} test samples, shape {fold_pred.shape}")
        else:
            print(f"  {fold_key}: No data (preds: {fold_pred.shape if hasattr(fold_pred, 'shape') else 'empty'}, real: {fold_actual.shape if hasattr(fold_actual, 'shape') else 'empty'})")
    
    if len(all_predictions) == 0:
        print("\nERROR: No test data found in results file!")
        print(f"Available keys in results: {list(results.keys())}")
        return
    
    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)
    
    print(f"\nTotal test samples: {all_predictions.shape[0]}")
    print(f"Total genes: {all_predictions.shape[1]}")
    
    # Compute overall metrics
    print("\n" + "=" * 70)
    print("Overall Performance")
    print("=" * 70)
    
    mae = np.mean(np.abs(all_predictions - all_actuals))
    mse = np.mean((all_predictions - all_actuals) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Compute gene correlations
    print("\n" + "=" * 70)
    print("Computing Per-Gene Correlations...")
    print("=" * 70)
    
    correlations, p_values = compute_gene_correlations(all_predictions, all_actuals, gene_names)
    
    valid_corr = correlations[~np.isnan(correlations)]
    print(f"\nGenes with valid correlations: {len(valid_corr)} / {len(correlations)}")
    print(f"Median correlation: {np.median(valid_corr):.4f}")
    print(f"Mean correlation: {np.mean(valid_corr):.4f}")
    print(f"Std correlation: {np.std(valid_corr):.4f}")
    print(f"Min correlation: {np.min(valid_corr):.4f}")
    print(f"Max correlation: {np.max(valid_corr):.4f}")
    
    # Correlation percentiles
    print("\nCorrelation Percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  {p}th percentile: {np.percentile(valid_corr, p):.4f}")
    
    # Top and bottom genes
    print("\n" + "=" * 70)
    print("Top 10 Genes (by correlation):")
    print("=" * 70)
    
    valid_mask = ~np.isnan(correlations)
    valid_indices = np.where(valid_mask)[0]
    valid_corr_array = correlations[valid_mask]
    
    top_10_idx = valid_indices[np.argsort(valid_corr_array)[-10:]][::-1]
    
    for i, idx in enumerate(top_10_idx, 1):
        print(f"{i:2d}. {gene_names[idx]:30s} r = {correlations[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("Bottom 10 Genes (by correlation):")
    print("=" * 70)
    
    bottom_10_idx = valid_indices[np.argsort(valid_corr_array)[:10]]
    
    for i, idx in enumerate(bottom_10_idx, 1):
        print(f"{i:2d}. {gene_names[idx]:30s} r = {correlations[idx]:.4f}")
    
    # Save gene correlations to CSV
    gene_corr_df = pd.DataFrame({
        'gene': gene_names,
        'correlation': correlations,
        'p_value': p_values
    })
    gene_corr_df = gene_corr_df.sort_values('correlation', ascending=False)
    gene_corr_df.to_csv(output_dir / 'gene_correlations.csv', index=False)
    print(f"\nSaved gene correlations to {output_dir / 'gene_correlations.csv'}")
    
    # Analyze by organ
    print("\n" + "=" * 70)
    print("Performance by Organ:")
    print("=" * 70)
    
    organ_stats = analyze_by_organ(results, reference_df)
    
    for organ, stats in sorted(organ_stats.items()):
        print(f"\n{organ.upper()}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  MAE: {stats['mae']:.4f}")
        print(f"  RMSE: {stats['rmse']:.4f}")
    
    # Save organ statistics
    organ_df = pd.DataFrame(organ_stats).T
    organ_df.to_csv(output_dir / 'organ_statistics.csv')
    print(f"\nSaved organ statistics to {output_dir / 'organ_statistics.csv'}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)
    
    plot_correlation_distribution(correlations, output_dir)
    plot_top_genes(all_predictions, all_actuals, correlations, gene_names, output_dir)
    plot_organ_performance(organ_stats, output_dir)
    plot_fold_performance(results, output_dir)
    
    # Save summary statistics
    summary = {
        'total_test_samples': all_predictions.shape[0],
        'total_genes': all_predictions.shape[1],
        'overall_mae': float(mae),
        'overall_mse': float(mse),
        'overall_rmse': float(rmse),
        'median_correlation': float(np.median(valid_corr)),
        'mean_correlation': float(np.mean(valid_corr)),
        'std_correlation': float(np.std(valid_corr)),
        'min_correlation': float(np.min(valid_corr)),
        'max_correlation': float(np.max(valid_corr)),
        'n_valid_genes': int(len(valid_corr))
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"\nSaved summary statistics to {output_dir / 'summary_statistics.csv'}")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
