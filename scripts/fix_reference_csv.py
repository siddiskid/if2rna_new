#!/usr/bin/env python3
"""
Quick fix: Remove .svs extension from wsi_file_name column in reference CSV

SEQUOIA's patch_gen_hdf5.py adds .svs extension automatically, so the 
reference CSV should NOT include it.
"""

import pandas as pd
import sys
from pathlib import Path

def fix_reference_csv(input_file: str, output_file: str = None):
    """Remove .svs extension from wsi_file_name column"""
    
    if output_file is None:
        output_file = input_file
    
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original wsi_file_name samples:")
    print(df['wsi_file_name'].head())
    
    # Remove .svs extension if present
    df['wsi_file_name'] = df['wsi_file_name'].str.replace('.svs$', '', regex=True)
    
    print(f"\nFixed wsi_file_name samples:")
    print(df['wsi_file_name'].head())
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved fixed reference to: {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_reference_csv.py <reference.csv> [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_reference_csv(input_file, output_file)
