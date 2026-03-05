#!/usr/bin/env python3
"""
Extract IF images from NanoString ROI report zip files

This script extracts IF images from the roi_report zip files for all organs
and organizes them into a structure suitable for SEQUOIA preprocessing.

Output structure:
    data/if_images/{organ}/{slide_name}/ROI_XXX.png

Usage:
    python extract_if_images.py
    python extract_if_images.py --organs Colon Kidney
"""

import argparse
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
import re


def parse_roi_image_name(filename):
    """
    Parse ROI image filename to extract metadata
    
    Examples:
        025247T1 - 001.png -> (025247T1, 001, None)
        025247T1 - 001 - PanCK+.png -> (025247T1, 001, PanCK+)
        MIPCC-DSP-A-20220222C-01 - 011 - CD68.png -> (MIPCC-DSP-A-20220222C-01, 011, CD68)
    """
    # Remove .png extension first
    name_without_ext = filename.replace('.png', '')
    
    # Split by ' - ' (space-hyphen-space)
    parts = name_without_ext.split(' - ')
    
    if len(parts) < 2:
        return None, None, None
    
    scan_label = parts[0]
    roi_num = parts[1]
    segment = parts[2] if len(parts) >= 3 else None
    
    return scan_label, roi_num, segment


def extract_images_from_zip(zip_path, output_dir, slide_name, extract_segments=True):
    """
    Extract IF images from a single zip file
    
    Args:
        zip_path: Path to zip file
        output_dir: Output directory
        slide_name: Slide/sample name (e.g., hu_colon_001)
        extract_segments: If True, extract segment images (PanCK+/-)
    """
    output_dir = Path(output_dir)
    slide_dir = output_dir / slide_name
    slide_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.filelist:
            if not file_info.filename.endswith('.png'):
                continue
            
            filename = Path(file_info.filename).name
            scan_label, roi_num, segment = parse_roi_image_name(filename)
            
            if not scan_label or not roi_num:
                continue
            
            # Skip segment/overlay images if not requested
            if segment and not extract_segments:
                continue
            
            # Skip "Segments" overlay images (we want the composite)
            if segment == "Segments":
                continue
            
            # Create output filename
            if segment:
                # For segment images: hu_colon_001_025247T1_ROI_001_PanCK+.png
                output_name = f"{slide_name}_{scan_label}_ROI_{roi_num}_{segment}.png"
            else:
                # For composite images: hu_colon_001_025247T1_ROI_001.png
                output_name = f"{slide_name}_{scan_label}_ROI_{roi_num}.png"
            
            output_path = slide_dir / output_name
            
            # Extract file
            with zip_ref.open(file_info) as source, open(output_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            
            extracted_count += 1
    
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description='Extract IF images from NanoString zip files')
    parser.add_argument('--if_data_dir', type=str, 
                       default='data/if_data',
                       help='Directory containing IF data')
    parser.add_argument('--output_dir', type=str,
                       default='data/if_images',
                       help='Output directory for extracted images')
    parser.add_argument('--organs', nargs='+',
                       default=['Colon', 'Kidney', 'Liver', 'Lymph Node'],
                       help='Organs to process')
    parser.add_argument('--extract_segments', action='store_true',
                       help='Also extract segment images (PanCK+/-)')
    
    args = parser.parse_args()
    
    if_data_dir = Path(args.if_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Extracting IF images from NanoString ROI reports")
    print("="*70)
    
    total_images = 0
    
    for organ in args.organs:
        organ_dir = if_data_dir / organ
        if not organ_dir.exists():
            print(f"\n⚠ Organ directory not found: {organ_dir}")
            continue
        
        print(f"\n[{organ}]")
        
        roi_report_dir = organ_dir / "workflow_and_count_files" / "workflow" / "roi_report"
        if not roi_report_dir.exists():
            print(f"  ✗ ROI report directory not found")
            continue
        
        # Find all zip files
        zip_files = list(roi_report_dir.glob("*.zip"))
        if not zip_files:
            print(f"  ✗ No zip files found")
            continue
        
        print(f"  Found {len(zip_files)} zip file(s)")
        
        # Create organ-specific output directory
        organ_output_dir = output_dir / organ
        organ_output_dir.mkdir(parents=True, exist_ok=True)
        
        for zip_file in tqdm(zip_files, desc=f"  Extracting {organ}"):
            slide_name = zip_file.stem  # e.g., hu_colon_001
            
            count = extract_images_from_zip(
                zip_file, 
                organ_output_dir, 
                slide_name,
                extract_segments=args.extract_segments
            )
            total_images += count
        
        print(f"  ✓ Extracted {sum(1 for _ in organ_output_dir.rglob('*.png'))} images")
    
    print(f"\n{'='*70}")
    print(f"✓ Extraction complete!")
    print(f"  Total images extracted: {total_images}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nDirectory structure:")
    for organ_dir in sorted(output_dir.glob("*")):
        if organ_dir.is_dir():
            slide_count = len(list(organ_dir.glob("*")))
            image_count = len(list(organ_dir.rglob("*.png")))
            print(f"  {organ_dir.name}: {slide_count} slides, {image_count} images")


if __name__ == '__main__':
    main()
