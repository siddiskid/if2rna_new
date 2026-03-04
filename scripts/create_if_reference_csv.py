#!/usr/bin/env python3
"""
Create reference.csv for IF2RNA from NanoString data

This script:
1. Reads normalized gene expression from NanoString Excel files
2. Matches ROIs to extracted IF images
3. Creates reference.csv in SEQUOIA format

Output format:
    wsi_file_name,patient_id,organ_type,rna_GENE1,rna_GENE2,...

Usage:
    python create_if_reference_csv.py
    python create_if_reference_csv.py --use_segments  # Include PanCK+/- segments
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_nanostring_expression(excel_path, use_normalized=True):
    """
    Read gene expression data from NanoString Excel file
    
    Args:
        excel_path: Path to Export file
        use_normalized: If True, use Export4 (Q3 normalized), else Export1 (raw)
    
    Returns:
        segment_df: DataFrame with segment metadata
        expression_df: DataFrame with gene expression (genes x segments)
    """
    # Read segment properties
    segment_df = pd.read_excel(excel_path, sheet_name='SegmentProperties')
    
    # Read expression matrix
    if use_normalized and 'TargetCountMatrix' in pd.ExcelFile(excel_path).sheet_names:
        expression_df = pd.read_excel(excel_path, sheet_name='TargetCountMatrix')
    elif 'BioProbeCountMatrix' in pd.ExcelFile(excel_path).sheet_names:
        expression_df = pd.read_excel(excel_path, sheet_name='BioProbeCountMatrix')
    else:
        raise ValueError(f"Could not find expression matrix in {excel_path}")
    
    return segment_df, expression_df


def match_rois_to_images(segment_df, image_dir, organ):
    """
    Match segment metadata to extracted IF images
    
    Args:
        segment_df: DataFrame with segment metadata
        image_dir: Directory containing extracted images
        organ: Organ name
    
    Returns:
        DataFrame with matched ROIs and image paths
    """
    matched_data = []
    
    organ_image_dir = Path(image_dir) / organ
    if not organ_image_dir.exists():
        print(f"  ⚠ Image directory not found: {organ_image_dir}")
        return pd.DataFrame()
    
    for idx, row in segment_df.iterrows():
        slide_name = row['SlideName']
        scan_label = row['ScanLabel']
        roi_label = str(row['ROILabel']).zfill(3)
        segment_label = row['SegmentLabel']
        
        # Construct expected image filename
        if pd.notna(segment_label) and segment_label not in ['Full ROI', '']:
            # Segment-specific image
            image_name = f"{slide_name}_{scan_label}_ROI_{roi_label}_{segment_label}.png"
        else:
            # Full ROI composite
            image_name = f"{slide_name}_{scan_label}_ROI_{roi_label}.png"
        
        image_path = organ_image_dir / slide_name / image_name
        
        if image_path.exists():
            matched_data.append({
                'wsi_file_name': image_name.replace('.png', ''),
                'image_path': str(image_path),
                'slide_name': slide_name,
                'scan_label': scan_label,
                'roi_label': roi_label,
                'segment_label': segment_label,
                'segment_display_name': row['SegmentDisplayName'],
                'roi_index': idx
            })
    
    return pd.DataFrame(matched_data)


def create_reference_csv(if_data_dir, image_dir, output_file, organs, 
                        use_normalized=True, use_segments=False,
                        min_expression=0.0):
    """
    Create reference.csv for IF2RNA training
    
    Args:
        if_data_dir: Directory containing NanoString data
        image_dir: Directory containing extracted images
        output_file: Output CSV file path
        organs: List of organs to process
        use_normalized: Use normalized expression data
        use_segments: Include segment-level data (PanCK+/-)
        min_expression: Minimum expression threshold (filter low genes)
    """
    all_reference_data = []
    
    print("="*70)
    print("Creating IF2RNA reference.csv")
    print("="*70)
    
    for organ in organs:
        print(f"\n[{organ}]")
        
        organ_dir = Path(if_data_dir) / organ
        if not organ_dir.exists():
            print(f"  ✗ Organ directory not found")
            continue
        
        # Read expression data
        excel_file = 'Export4_NormalizationQ3.xlsx' if use_normalized else 'Export1_InitialDataset.xlsx'
        excel_path = organ_dir / "workflow_and_count_files" / "count" / excel_file
        
        if not excel_path.exists():
            print(f"  ✗ Expression file not found: {excel_file}")
            continue
        
        print(f"  Reading expression data from {excel_file}...")
        segment_df, expression_df = read_nanostring_expression(excel_path, use_normalized)
        
        print(f"    {len(segment_df)} segments")
        print(f"    {len(expression_df)} genes")
        
        # Match ROIs to images
        print(f"  Matching ROIs to images...")
        matched_df = match_rois_to_images(segment_df, image_dir, organ)
        
        if len(matched_df) == 0:
            print(f"  ✗ No images matched")
            continue
        
        print(f"    ✓ Matched {len(matched_df)} ROIs to images")
        
        # Filter segments if needed
        if not use_segments:
            # Only keep full ROI composites (no PanCK+/-)
            matched_df = matched_df[
                matched_df['segment_label'].isna() | 
                (matched_df['segment_label'] == 'Full ROI')
            ].copy()
            print(f"    Using full ROI composites: {len(matched_df)} ROIs")
        
        # Extract patient IDs from slide names
        matched_df['patient_id'] = matched_df['slide_name'].str.replace('_', '-')
        matched_df['organ_type'] = organ.replace(' ', '_')
        
        # Get expression columns (exclude metadata columns)
        metadata_cols = ['TargetName', 'GeneName', 'Accession', 'CodeClass', 'SystematicName']
        gene_cols = [col for col in expression_df.columns if col not in metadata_cols]
        
        # Build reference data for each matched ROI
        for idx, row in matched_df.iterrows():
            roi_idx = row['roi_index']
            
            if roi_idx >= len(gene_cols):
                print(f"  ⚠ ROI index {roi_idx} out of bounds")
                continue
            
            # Get expression values for this ROI (column in expression matrix)
            expression_col = gene_cols[roi_idx]
            expression_values = expression_df[expression_col].values
            
            # Build reference row
            ref_row = {
                'wsi_file_name': row['wsi_file_name'],
                'patient_id': row['patient_id'],
                'organ_type': row['organ_type'],
                'slide_name': row['slide_name'],
                'scan_label': row['scan_label'],
                'roi_label': row['roi_label'],
                'segment_label': row['segment_label'] if pd.notna(row['segment_label']) else 'Full',
            }
            
            # Add gene expression values
            for gene_idx, gene_row in expression_df.iterrows():
                gene_name = gene_row['TargetName'] if 'TargetName' in expression_df.columns else gene_row['GeneName']
                gene_value = expression_values[gene_idx]
                
                # Handle NaN/inf values
                if pd.isna(gene_value) or np.isinf(gene_value):
                    gene_value = 0.0
                
                ref_row[f'rna_{gene_name}'] = gene_value
            
            all_reference_data.append(ref_row)
    
    # Create final DataFrame
    reference_df = pd.DataFrame(all_reference_data)
    
    if len(reference_df) == 0:
        print("\n✗ No reference data created!")
        return
    
    # Filter low-expression genes if requested
    gene_cols = [col for col in reference_df.columns if col.startswith('rna_')]
    if min_expression > 0:
        print(f"\nFiltering genes with mean expression < {min_expression}...")
        gene_means = reference_df[gene_cols].mean()
        genes_to_keep = gene_means[gene_means >= min_expression].index.tolist()
        print(f"  Keeping {len(genes_to_keep)} / {len(gene_cols)} genes")
        
        non_gene_cols = [col for col in reference_df.columns if not col.startswith('rna_')]
        reference_df = reference_df[non_gene_cols + genes_to_keep]
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Reference CSV created!")
    print(f"  Output: {output_path}")
    print(f"  Samples: {len(reference_df)}")
    print(f"  Genes: {len([col for col in reference_df.columns if col.startswith('rna_')])}")
    print(f"  Organs: {reference_df['organ_type'].nunique()}")
    print(f"{'='*70}")
    
    # Print summary by organ
    print("\nSummary by organ:")
    for organ in reference_df['organ_type'].unique():
        count = len(reference_df[reference_df['organ_type'] == organ])
        print(f"  {organ}: {count} samples")
    
    return reference_df


def main():
    parser = argparse.ArgumentParser(description='Create IF2RNA reference.csv')
    parser.add_argument('--if_data_dir', type=str,
                       default='data/if_data',
                       help='Directory containing NanoString data')
    parser.add_argument('--image_dir', type=str,
                       default='data/if_images',
                       help='Directory containing extracted images')
    parser.add_argument('--output_file', type=str,
                       default='data/metadata/if_reference.csv',
                       help='Output reference CSV file')
    parser.add_argument('--organs', nargs='+',
                       default=['Colon', 'Kidney', 'Liver', 'Lymph Node'],
                       help='Organs to process')
    parser.add_argument('--raw', action='store_true',
                       help='Use raw counts instead of normalized')
    parser.add_argument('--use_segments', action='store_true',
                       help='Include segment-level data (PanCK+/-)')
    parser.add_argument('--min_expression', type=float, default=0.0,
                       help='Minimum mean expression to keep gene')
    
    args = parser.parse_args()
    
    create_reference_csv(
        if_data_dir=args.if_data_dir,
        image_dir=args.image_dir,
        output_file=args.output_file,
        organs=args.organs,
        use_normalized=not args.raw,
        use_segments=args.use_segments,
        min_expression=args.min_expression
    )


if __name__ == '__main__':
    main()
