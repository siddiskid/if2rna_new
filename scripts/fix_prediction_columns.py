#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path

def fix_prediction_columns(predictions_file, gene_list_file, output_file):
    """
    Replace pred_gene_X columns with actual gene names from gene_list.csv
    and add 'rna_' prefix to match reference.csv format.
    """
    print(f"Loading gene list from {gene_list_file}")
    gene_list = pd.read_csv(gene_list_file)
    genes = gene_list['gene'].tolist()
    print(f"  Found {len(genes)} genes")
    
    print(f"Loading predictions from {predictions_file}")
    predictions = pd.read_csv(predictions_file)
    print(f"  Shape: {predictions.shape}")
    
    metadata_cols = ['wsi_file_name', 'patient_id', 'tcga_project']
    pred_cols = [col for col in predictions.columns if col.startswith('pred_gene_')]
    
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Prediction columns: {len(pred_cols)}")
    
    if len(pred_cols) != len(genes):
        print(f"WARNING: Number of prediction columns ({len(pred_cols)}) != number of genes ({len(genes)})")
    
    column_mapping = {}
    for i, pred_col in enumerate(pred_cols):
        if i < len(genes):
            column_mapping[pred_col] = f"rna_{genes[i]}"
    
    print(f"Renaming {len(column_mapping)} prediction columns")
    predictions_renamed = predictions.rename(columns=column_mapping)
    
    print(f"Saving to {output_file}")
    predictions_renamed.to_csv(output_file, index=False)
    print(f"Done! Saved {predictions_renamed.shape[0]} rows x {predictions_renamed.shape[1]} columns")
    
    print("\nFirst few gene columns:")
    gene_cols = [col for col in predictions_renamed.columns if col.startswith('rna_')][:5]
    print(f"  {gene_cols}")

def main():
    parser = argparse.ArgumentParser(description='Fix prediction column names')
    parser.add_argument('--predictions', required=True, help='Input predictions CSV file')
    parser.add_argument('--gene_list', required=True, help='Gene list CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    fix_prediction_columns(args.predictions, args.gene_list, args.output)

if __name__ == '__main__':
    main()
