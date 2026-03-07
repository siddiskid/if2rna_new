#!/usr/bin/env python3
"""
Create reference CSV with GENE SYMBOLS (not Ensembl IDs)

TCGA RNA-seq files have both gene_id (Ensembl) and gene_name (symbol).
SEQUOIA expects gene symbols, so we use gene_name column.

Usage:
    python scripts/create_reference_csv_gene_symbols.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def create_reference_with_symbols():
    """Create reference CSV using gene symbols from TCGA RNA files"""
    
    slides_dir = "data/hne_data/raw/images"
    rna_dir = "data/hne_data/raw/rna"
    output_file = "data/metadata/tcga_reference.csv"
    
    print("="*70)
    print("Creating Reference CSV with Gene Symbols")
    print("="*70)
    
    # Get all slides and RNA files (sorted to match by index)
    slides = sorted(glob.glob(f"{slides_dir}/*.svs"))
    rna_files = sorted(glob.glob(f"{rna_dir}/*.tsv"))
    
    print(f"\nFound {len(slides)} slides")
    print(f"Found {len(rna_files)} RNA files")
    
    if len(slides) != len(rna_files):
        print(f"\n⚠ WARNING: Mismatch in counts!")
        print(f"  This script assumes 1:1 matching by sorted order.")
    
    # Process each slide-RNA pair
    rows = []
    
    for i, (slide_path, rna_path) in enumerate(zip(slides, rna_files)):
        slide_file = Path(slide_path).name
        # Remove .svs for reference CSV (SEQUOIA's patch script adds it back)
        slide_id = slide_file.replace('.svs', '')
        
        # Extract patient ID from slide name (TCGA-XX-XXXX format)
        parts = slide_id.split('-')
        patient_id = '-'.join(parts[:3]) if len(parts) >= 3 else slide_id
        
        # Extract project (e.g., BRCA from TCGA-5L)
        project = f"TCGA-{parts[1]}" if len(parts) > 1 else "UNKNOWN"
        
        print(f"\n[{i+1}/{len(slides)}] {slide_id}")
        print(f"  Patient: {patient_id}")
        print(f"  Project: {project}")
        print(f"  RNA file: {Path(rna_path).name}")
        
        # Read RNA file
        # TCGA format has comment lines starting with #
        rna_df = pd.read_csv(rna_path, sep='\t', comment='#')
        
        print(f"  RNA data shape: {rna_df.shape}")
        print(f"  Columns: {list(rna_df.columns)}")
        
        # Skip metadata rows (N_unmapped, N_multimapping, etc.)
        rna_df = rna_df[~rna_df['gene_id'].str.startswith('N_')]
        
        print(f"  After filtering: {len(rna_df)} genes")
        
        # Use gene_name (symbol) instead of gene_id (Ensembl)
        if 'gene_name' not in rna_df.columns:
            print(f"  ✗ ERROR: No 'gene_name' column in RNA file!")
            print(f"  Available columns: {list(rna_df.columns)}")
            continue
        
        # Use tpm_unstranded for expression values (normalized, preferred)
        # Fall back to unstranded counts if TPM not available
        if 'tpm_unstranded' in rna_df.columns:
            expr_col = 'tpm_unstranded'
        elif 'unstranded' in rna_df.columns:
            expr_col = 'unstranded'
        else:
            print(f"  ✗ ERROR: No expression column found!")
            continue
        
        print(f"  Using expression column: {expr_col}")
        
        # Create row for reference CSV
        row = {
            'wsi_file_name': slide_id,  # No .svs extension
            'patient_id': patient_id,
            'tcga_project': project
        }
        
        # Add gene expression with gene_name as key
        for _, gene_row in rna_df.iterrows():
            gene_symbol = gene_row['gene_name']
            expr_value = gene_row[expr_col]
            row[f'rna_{gene_symbol}'] = expr_value
        
        rows.append(row)
        
        # Show sample of genes
        sample_genes = rna_df['gene_name'].head(5).tolist()
        print(f"  Sample genes: {', '.join(sample_genes)}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.fillna(0)  # Fill missing values with 0
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✓ Reference CSV Created")
    print("="*70)
    print(f"  Output: {output_file}")
    print(f"  Slides: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Count RNA columns
    rna_cols = [c for c in df.columns if c.startswith('rna_')]
    print(f"  RNA columns: {len(rna_cols)}")
    
    # Show sample RNA columns (gene symbols)
    print(f"  Sample RNA columns: {rna_cols[:10]}")
    
    # Check overlap with SEQUOIA gene list
    sequoia_genes_file = "sequoia-pub/examples/gene_list.csv"
    if Path(sequoia_genes_file).exists():
        sequoia_df = pd.read_csv(sequoia_genes_file)
        sequoia_genes = set(sequoia_df['gene'].tolist())
        ref_genes = set([c.replace('rna_', '') for c in rna_cols])
        
        overlap = sequoia_genes & ref_genes
        print(f"\n  SEQUOIA genes: {len(sequoia_genes)}")
        print(f"  Reference genes: {len(ref_genes)}")
        print(f"  ✓ Overlap: {len(overlap)} genes")
        
        if len(overlap) == 0:
            print(f"\n  ⚠ WARNING: No gene overlap!")
            print(f"    SEQUOIA genes: {list(sequoia_genes)[:5]}")
            print(f"    Reference genes: {list(ref_genes)[:5]}")
    
    print("="*70)

if __name__ == "__main__":
    create_reference_with_symbols()
