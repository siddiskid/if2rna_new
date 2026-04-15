#!/usr/bin/env python3
"""
Extract patches from IF ROI images for IF2RNA

Adapted from SEQUOIA's patch_gen_hdf5.py for IF images
Differences:
- Works with regular image files (not whole slide .svs)
- No tissue segmentation needed (ROIs are already cropped)
- Simpler patching strategy

Usage:
    python preprocess_if_patches.py --ref_file data/metadata/if_reference.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
from PIL import Image
import cv2


def extract_patches_from_image(
    image_path,
    patch_size=256,
    max_patches=100,
    overlap=0,
    background_threshold=0.95,
    duplicate_if_needed=True,
):
    """
    Extract patches from a single IF ROI image
    
    Args:
        image_path: Path to image file
        patch_size: Size of square patches
        max_patches: Maximum number of patches to extract
        overlap: Overlap between patches (0-0.5)
        background_threshold: Skip patches with >X completely black pixels (0-1)
    
    Returns:
        List of patches as numpy arrays
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        if h < patch_size or w < patch_size:
            # Image too small, resize and return as single patch
            img_resized = img.resize((patch_size, patch_size))
            return [np.array(img_resized)]
        
        # Calculate stride
        stride = int(patch_size * (1 - overlap))
        
        patches = []
        positions = []
        
        # Extract patches in a grid
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img_array[y:y+patch_size, x:x+patch_size]
                
                # For IF images: only skip if patch is COMPLETELY black (no signal at all)
                # IF images are naturally dark with bright fluorescent signals
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                completely_black_ratio = np.sum(gray < 5) / gray.size
                
                # Keep patch if it has ANY signal (less than 95% completely black)
                if completely_black_ratio < background_threshold:
                    patches.append(patch)
                    positions.append((x, y))
        
        # If we have too many patches, sample randomly
        if len(patches) > max_patches:
            indices = np.random.choice(len(patches), max_patches, replace=False)
            patches = [patches[i] for i in sorted(indices)]
        
        # Optionally duplicate patches to fixed count. For ROSIE conversions,
        # this should usually be disabled to avoid creating many repeated tokens.
        if duplicate_if_needed and len(patches) < max_patches and len(patches) > 0:
            while len(patches) < max_patches:
                patches.append(patches[np.random.randint(len(patches))])
        
        return patches
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def process_sample(row, image_dir, output_dir, patch_size, max_patches, overlap, duplicate_if_needed):
    """
    Process a single sample from reference CSV
    """
    wsi_name = row['wsi_file_name']
    organ = row['organ_type']
    slide_name = row['slide_name']
    
    # Prefer explicit image_path if provided in the reference.
    image_path = None
    if 'image_path' in row and pd.notna(row['image_path']):
        candidate = Path(str(row['image_path']))
        if candidate.exists():
            image_path = candidate

    # Fallback to legacy image directory layout.
    if image_path is None:
        organ_display = organ.replace('_', ' ')  # Lymph_Node -> Lymph Node
        possible_paths = [
            Path(image_dir) / organ / slide_name / f"{wsi_name}.png",
            Path(image_dir) / organ_display / slide_name / f"{wsi_name}.png",
        ]

        for path in possible_paths:
            if path.exists():
                image_path = path
                break
    
    if image_path is None:
        return None, f"Image not found: {wsi_name}"
    
    # Create output directory
    output_path = Path(output_dir) / organ / wsi_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    h5_file = output_path / f"{wsi_name}.h5"
    if h5_file.exists():
        return wsi_name, "Already processed"
    
    # Extract patches
    patches = extract_patches_from_image(
        image_path, 
        patch_size=patch_size, 
        max_patches=max_patches,
        overlap=overlap,
        duplicate_if_needed=duplicate_if_needed,
    )
    
    if len(patches) == 0:
        return None, f"No patches extracted: {wsi_name}"
    
    # Save to HDF5
    try:
        with h5py.File(h5_file, 'w') as f:
            for i, patch in enumerate(patches):
                f.create_dataset(f"patch_{i}", data=patch, compression='gzip')
        
        # Write completion marker
        with open(output_path / "complete.txt", 'w') as f:
            f.write(f"Number of patches: {len(patches)}\n")
        
        return wsi_name, len(patches)
        
    except Exception as e:
        return None, f"Error saving {wsi_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Extract patches from IF images')
    parser.add_argument('--ref_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--image_dir', type=str, default='data/if_images',
                       help='Directory containing IF images')
    parser.add_argument('--output_dir', type=str, default='data/if_patches',
                       help='Output directory for patches')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size (default: 256)')
    parser.add_argument('--max_patches', type=int, default=100,
                       help='Maximum patches per image (default: 100)')
    parser.add_argument('--overlap', type=float, default=0.0,
                       help='Overlap between patches 0-0.5 (default: 0)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start index for parallelization')
    parser.add_argument('--end', type=int, default=None,
                       help='End index for parallelization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_duplicate_patches', action='store_true',
                       help='Do not duplicate patches when fewer than --max_patches are found')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print("Extracting patches from IF images")
    print("="*70)
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result, message = process_sample(
            row, args.image_dir, args.output_dir, 
            args.patch_size, args.max_patches, args.overlap,
            duplicate_if_needed=not args.no_duplicate_patches,
        )
        
        if result is None:
            error_count += 1
            if idx < 5:  # Only print first few errors
                print(f"\n  ✗ {message}")
        elif message == "Already processed":
            skip_count += 1
        else:
            success_count += 1
    
    print("\n" + "="*70)
    print("Patch extraction complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
