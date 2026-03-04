#!/usr/bin/env python3
"""
Validate IF2RNA data preparation

This script checks:
1. Image extraction completeness
2. Reference CSV format and integrity
3. Image-RNA matches

Usage:
    python validate_if_data.py
"""

import argparse
import pandas as pd
from pathlib import Path


def validate_images(image_dir):
    """Validate extracted images"""
    print("\n[1/3] Validating extracted images...")
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"  ✗ Image directory not found: {image_dir}")
        return False
    
    organs = [d for d in image_dir.iterdir() if d.is_dir()]
    if not organs:
        print(f"  ✗ No organ directories found")
        return False
    
    print(f"  ✓ Found {len(organs)} organ directories")
    
    total_images = 0
    for organ_dir in sorted(organs):
        slides = [d for d in organ_dir.iterdir() if d.is_dir()]
        images = list(organ_dir.rglob("*.png"))
        print(f"    {organ_dir.name}: {len(slides)} slides, {len(images)} images")
        total_images += len(images)
    
    if total_images == 0:
        print(f"  ✗ No images found!")
        return False
    
    print(f"  ✓ Total images: {total_images}")
    return True


def validate_reference_csv(csv_path):
    """Validate reference CSV format"""
    print("\n[2/3] Validating reference CSV...")
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  ✗ Reference CSV not found: {csv_path}")
        return False, None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ CSV loaded successfully")
        print(f"    Samples: {len(df)}")
        print(f"    Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ['wsi_file_name', 'patient_id', 'organ_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing required columns: {missing_cols}")
            return False, None
        
        print(f"  ✓ Required columns present")
        
        # Check gene columns
        gene_cols = [col for col in df.columns if col.startswith('rna_')]
        if not gene_cols:
            print(f"  ✗ No gene expression columns found")
            return False, None
        
        print(f"  ✓ Gene columns: {len(gene_cols)}")
        
        # Check for NaN/inf values
        gene_data = df[gene_cols]
        nan_count = gene_data.isna().sum().sum()
        inf_count = (gene_data == float('inf')).sum().sum() + (gene_data == float('-inf')).sum().sum()
        
        if nan_count > 0:
            print(f"  ⚠ Warning: {nan_count} NaN values in gene expression")
        if inf_count > 0:
            print(f"  ⚠ Warning: {inf_count} infinite values in gene expression")
        
        # Print summary statistics
        print(f"\n  Expression statistics:")
        print(f"    Mean expression: {gene_data.mean().mean():.2f}")
        print(f"    Median expression: {gene_data.median().median():.2f}")
        print(f"    Min expression: {gene_data.min().min():.2f}")
        print(f"    Max expression: {gene_data.max().max():.2f}")
        
        # Print sample distribution
        print(f"\n  Sample distribution:")
        for organ in sorted(df['organ_type'].unique()):
            count = len(df[df['organ_type'] == organ])
            print(f"    {organ}: {count}")
        
        return True, df
        
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}")
        return False, None


def validate_image_matches(df, image_dir):
    """Validate that all reference entries have corresponding images"""
    print("\n[3/3] Validating image-RNA matches...")
    
    image_dir = Path(image_dir)
    missing_images = []
    
    for idx, row in df.iterrows():
        wsi_name = row['wsi_file_name']
        organ = row['organ_type'].replace('_', ' ')  # Lymph_Node -> Lymph Node
        
        # Try to find the image
        possible_paths = [
            image_dir / organ / row['slide_name'] / f"{wsi_name}.png",
            image_dir / organ.replace(' ', '_') / row['slide_name'] / f"{wsi_name}.png",
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                found = True
                break
        
        if not found:
            missing_images.append((wsi_name, organ))
            if len(missing_images) <= 5:  # Only print first 5
                print(f"  ✗ Missing: {wsi_name}")
    
    if missing_images:
        print(f"  ✗ {len(missing_images)} images referenced in CSV but not found")
        return False
    
    print(f"  ✓ All {len(df)} samples have corresponding images")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate IF2RNA data')
    parser.add_argument('--image_dir', type=str,
                       default='data/if_images',
                       help='Directory containing extracted images')
    parser.add_argument('--reference_csv', type=str,
                       default='data/metadata/if_reference.csv',
                       help='Reference CSV file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("IF2RNA Data Validation")
    print("="*70)
    
    # Validate images
    images_ok = validate_images(args.image_dir)
    
    # Validate reference CSV
    csv_ok, df = validate_reference_csv(args.reference_csv)
    
    # Validate matches
    if images_ok and csv_ok and df is not None:
        matches_ok = validate_image_matches(df, args.image_dir)
    else:
        matches_ok = False
    
    print("\n" + "="*70)
    if images_ok and csv_ok and matches_ok:
        print("✓ All validation checks passed!")
        print("  Ready for SEQUOIA preprocessing")
    else:
        print("✗ Some validation checks failed")
        print("  Please fix issues before proceeding")
    print("="*70)


if __name__ == '__main__':
    main()
