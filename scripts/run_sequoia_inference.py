#!/usr/bin/env python3
"""
Run SEQUOIA model inference on preprocessed slides

This script:
1. Loads pre-trained SEQUOIA model from HuggingFace
2. Runs inference on preprocessed slides
3. Outputs predicted gene expression values
4. Compares with ground truth RNA-seq (if available)

Usage:
    # Run inference with downloaded model
    python run_sequoia_inference.py --model_dir models/sequoia/brca-0 --ref_file data/metadata/reference.csv
    
    # Run inference loading from HuggingFace directly
    python run_sequoia_inference.py --model_id gevaertlab/sequoia-brca-0 --ref_file data/metadata/reference.csv
    
    # Use multiple folds (ensemble)
    python run_sequoia_inference.py --model_dir models/sequoia --cancer BRCA --folds 0 1 2 3 4
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import h5py
import json


class SEQUOIAInference:
    """Run SEQUOIA model inference"""
    
    def __init__(self, feature_dir: str = "data/processed/features",
                 output_dir: str = "results"):
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_model(self, model_path: str = None, model_id: str = None):
        """Load SEQUOIA model from local path or HuggingFace"""
        print("\n[1/4] Loading SEQUOIA model...")
        
        try:
            # Add SEQUOIA src to path
            import sys
            sys.path.insert(0, 'sequoia-pub')
            from src.tformer_lin import ViS
        except ImportError as e:
            print(f"   ✗ Could not import SEQUOIA modules: {e}")
            print("   Make sure SEQUOIA is set up: bash scripts/setup_sequoia.sh")
            sys.exit(1)
        
        try:
            if model_id:
                # Load from HuggingFace
                print(f"   Loading from HuggingFace: {model_id}")
                from huggingface_hub import login
                # User should be logged in via: huggingface-cli login
                model = ViS.from_pretrained(model_id)
            elif model_path:
                # Load from local path
                print(f"   Loading from local path: {model_path}")
                model = ViS.from_pretrained(model_path)
            else:
                print("   ✗ Must provide either model_path or model_id")
                return None
            
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✓ Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"   ✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_features(self, wsi_name: str, project: str, 
                     feat_type: str = "resnet") -> Optional[dict]:
        """Load preprocessed features for a slide"""
        # Remove .svs extension if present
        wsi_name = wsi_name.replace('.svs', '').replace('.tiff', '')
        
        # Find feature file
        feat_file = self.feature_dir / project / wsi_name / f"{wsi_name}.h5"
        
        if not feat_file.exists():
            print(f"   ⚠ Feature file not found: {feat_file}")
            return None
        
        try:
            with h5py.File(feat_file, 'r') as f:
                # Check what features are available
                available_features = list(f.keys())
                
                # Load features based on type
                if "cluster_features" in f:
                    cluster_features = f["cluster_features"][:]
                else:
                    print(f"   ⚠ cluster_features not found in {feat_file.name}")
                    print(f"      Available: {available_features}")
                    return None
                
                return {
                    "cluster_features": cluster_features,
                    "wsi_name": wsi_name
                }
                
        except Exception as e:
            print(f"   ✗ Error loading features from {feat_file}: {e}")
            return None
    
    def run_inference_single(self, model, features: dict) -> Optional[np.ndarray]:
        """Run inference on a single slide"""
        try:
            cluster_features = torch.from_numpy(features["cluster_features"]).float()
            cluster_features = cluster_features.to(self.device)
            
            # Add batch dimension
            cluster_features = cluster_features.unsqueeze(0)
            
            with torch.no_grad():
                predictions = model(cluster_features)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy().squeeze()
            
            return predictions
            
        except Exception as e:
            print(f"   ✗ Inference error for {features['wsi_name']}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_gene_list(self, gene_list_path: str = "models/gene_list.csv") -> List[str]:
        """Load gene list for SEQUOIA output"""
        gene_list_path = Path(gene_list_path)
        
        if not gene_list_path.exists():
            print(f"   ⚠ Gene list not found: {gene_list_path}")
            return None
        
        try:
            # Gene list from SEQUOIA is a CSV with just gene names
            df = pd.read_csv(gene_list_path, header=None)
            genes = df[0].tolist()
            print(f"   ✓ Loaded {len(genes)} genes")
            return genes
        except Exception as e:
            print(f"   ✗ Error loading gene list: {e}")
            return None
    
    def run_inference(self, model, ref_file: str, gene_list: List[str],
                     feat_type: str = "resnet") -> pd.DataFrame:
        """Run inference on all slides in reference file"""
        print("\n[2/4] Running inference...")
        
        ref_df = pd.read_csv(ref_file)
        print(f"   Processing {len(ref_df)} slides")
        
        results = []
        
        for idx, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
            wsi_name = row['wsi_file_name']
            patient_id = row['patient_id']
            project = row.get('tcga_project', 'UNKNOWN')
            
            # Load features
            features = self.load_features(wsi_name, project, feat_type)
            if features is None:
                continue
            
            # Run inference
            predictions = self.run_inference_single(model, features)
            if predictions is None:
                continue
            
            # Create result row
            result = {
                "wsi_file_name": wsi_name,
                "patient_id": patient_id,
                "tcga_project": project
            }
            
            # Add predicted gene expression
            if gene_list and len(predictions) == len(gene_list):
                for gene, pred_value in zip(gene_list, predictions):
                    result[f"pred_{gene}"] = pred_value
            else:
                # No gene list or mismatch - just use indices
                for i, pred_value in enumerate(predictions):
                    result[f"pred_gene_{i}"] = pred_value
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"   ✓ Processed {len(results_df)} slides successfully")
        
        return results_df
    
    def compare_with_ground_truth(self, predictions_df: pd.DataFrame,
                                  ref_file: str) -> pd.DataFrame:
        """Compare predictions with ground truth RNA-seq"""
        print("\n[3/4] Comparing with ground truth...")
        
        ref_df = pd.read_csv(ref_file)
        
        # Find common genes
        pred_genes = [c.replace('pred_', '') for c in predictions_df.columns 
                     if c.startswith('pred_') and not c.startswith('pred_gene_')]
        rna_genes = [c.replace('rna_', '') for c in ref_df.columns 
                    if c.startswith('rna_')]
        
        common_genes = set(pred_genes) & set(rna_genes)
        
        if not common_genes:
            print("   ⚠ No common genes found between predictions and ground truth")
            return None
        
        print(f"   Found {len(common_genes)} common genes")
        
        # Merge dataframes
        merged = predictions_df.merge(ref_df, on='wsi_file_name', how='inner')
        
        correlations = []
        
        for gene in common_genes:
            pred_col = f"pred_{gene}"
            true_col = f"rna_{gene}"
            
            if pred_col in merged.columns and true_col in merged.columns:
                # Calculate correlation
                pred_values = merged[pred_col].values
                true_values = merged[true_col].values
                
                # Remove NaN values
                mask = ~(np.isnan(pred_values) | np.isnan(true_values))
                if mask.sum() > 0:
                    corr = np.corrcoef(pred_values[mask], true_values[mask])[0, 1]
                    rmse = np.sqrt(np.mean((pred_values[mask] - true_values[mask])**2))
                    
                    correlations.append({
                        "gene": gene,
                        "correlation": corr,
                        "rmse": rmse,
                        "n_samples": mask.sum()
                    })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        print(f"\n   Correlation statistics:")
        print(f"      Mean correlation: {corr_df['correlation'].mean():.4f}")
        print(f"      Median correlation: {corr_df['correlation'].median():.4f}")
        print(f"      Mean RMSE: {corr_df['rmse'].mean():.4f}")
        
        return corr_df
    
    def save_results(self, predictions_df: pd.DataFrame, 
                    correlations_df: Optional[pd.DataFrame],
                    model_name: str):
        """Save inference results"""
        print("\n[4/4] Saving results...")
        
        # Save predictions
        pred_path = self.output_dir / f"predictions_{model_name}.csv"
        predictions_df.to_csv(pred_path, index=False)
        print(f"   ✓ Predictions saved: {pred_path}")
        
        # Save correlations if available
        if correlations_df is not None:
            corr_path = self.output_dir / f"correlations_{model_name}.csv"
            correlations_df.to_csv(corr_path, index=False)
            print(f"   ✓ Correlations saved: {corr_path}")
            
            # Save summary
            summary = {
                "model": model_name,
                "n_slides": len(predictions_df),
                "n_genes": len(correlations_df),
                "mean_correlation": float(correlations_df['correlation'].mean()),
                "median_correlation": float(correlations_df['correlation'].median()),
                "mean_rmse": float(correlations_df['rmse'].mean())
            }
            
            summary_path = self.output_dir / f"summary_{model_name}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   ✓ Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SEQUOIA model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Path to local model directory")
    parser.add_argument("--model_id", type=str, default=None,
                       help="HuggingFace model ID (e.g., gevaertlab/sequoia-brca-0)")
    parser.add_argument("--cancer", type=str, default=None,
                       help="Cancer type (e.g., BRCA) - used with --folds")
    parser.add_argument("--folds", nargs="+", type=int, default=[0],
                       help="Model folds to use (for ensemble)")
    parser.add_argument("--ref_file", type=str, required=True,
                       help="Path to reference.csv")
    parser.add_argument("--feature_dir", type=str, default="data/processed/features",
                       help="Directory containing preprocessed features")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--gene_list", type=str, default="models/gene_list.csv",
                       help="Path to gene list file")
    parser.add_argument("--feat_type", type=str, default="resnet",
                       choices=["resnet", "uni"],
                       help="Feature type used during preprocessing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_dir and not args.model_id and not args.cancer:
        print("Error: Must provide either --model_dir, --model_id, or --cancer")
        sys.exit(1)
    
    inference = SEQUOIAInference(args.feature_dir, args.output_dir)
    
    # Load gene list
    print("\nLoading gene list...")
    gene_list = inference.load_gene_list(args.gene_list)
    
    # Determine which models to run
    if args.cancer:
        # Use multiple folds
        models_to_run = [
            (f"models/sequoia/{args.cancer.lower()}-{fold}", f"{args.cancer.lower()}-{fold}")
            for fold in args.folds
        ]
    elif args.model_id:
        models_to_run = [(args.model_id, args.model_id.split('/')[-1])]
    else:
        models_to_run = [(args.model_dir, Path(args.model_dir).name)]
    
    # Run inference for each model
    all_predictions = []
    
    for model_path, model_name in models_to_run:
        print("\n" + "="*70)
        print(f"Processing model: {model_name}")
        print("="*70)
        
        # Load model
        if args.model_id and model_path == args.model_id:
            model = inference.load_model(model_id=model_path)
        else:
            model = inference.load_model(model_path=model_path)
        
        if model is None:
            print(f"   ✗ Failed to load model: {model_name}")
            continue
        
        # Run inference
        predictions_df = inference.run_inference(model, args.ref_file, 
                                                gene_list, args.feat_type)
        
        if len(predictions_df) == 0:
            print(f"   ✗ No predictions generated for {model_name}")
            continue
        
        # Compare with ground truth
        correlations_df = inference.compare_with_ground_truth(predictions_df, args.ref_file)
        
        # Save results
        inference.save_results(predictions_df, correlations_df, model_name)
        
        all_predictions.append((model_name, predictions_df))
    
    print("\n" + "="*70)
    print("✓ Inference Complete!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Processed {len(all_predictions)} model(s)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
