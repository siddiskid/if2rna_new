#!/usr/bin/env python3
"""
Create SEQUOIA reference.csv from TCGA downloaded data

This script:
1. Reads TCGA RNA-seq data files
2. Matches them with H&E slide files
3. Creates a reference.csv in SEQUOIA format with columns:
   - wsi_file_name
   - patient_id
   - tcga_project
   - rna_{GENENAME} (one column per gene)

Usage:
    python create_reference_csv.py --data_dir data/raw
    python create_reference_csv.py --data_dir data/raw --genes gene_subset.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Optional
import re


class ReferenceBuilder:
    """Build SEQUOIA reference.csv from TCGA data"""
    
    def __init__(self, data_dir: str = "data/raw", slides_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        # Allow custom slides directory (e.g., data/raw/images instead of data/raw/slides)
        if slides_dir:
            self.slides_dir = Path(slides_dir)
        else:
            self.slides_dir = self.data_dir / "slides"
        self.rna_dir = self.data_dir / "rna"
        self.metadata_dir = self.data_dir.parent / "metadata"
        
    def get_slide_files(self) -> List[Path]:
        """Get all slide files"""
        print("\n[1/5] Finding slide files...")
        
        slides = list(self.slides_dir.glob("*.svs"))
        slides += list(self.slides_dir.glob("*.tiff"))
        
        print(f"   Found {len(slides)} slide files")
        return slides
    
    def get_rna_files(self) -> List[Path]:
        """Get all RNA-seq files"""
        print("\n[2/5] Finding RNA-seq files...")
        
        rna_files = list(self.rna_dir.glob("*.tsv"))
        rna_files += list(self.rna_dir.glob("*.gz"))
        rna_files += list(self.rna_dir.glob("*.txt"))
        
        print(f"   Found {len(rna_files)} RNA files")
        return rna_files
    
    def parse_tcga_slide_name(self, slide_name: str) -> dict:
        """Parse TCGA slide filename to extract patient and sample info"""
        # TCGA format: TCGA-XX-XXXX-XXX-XX-XXX.UUID.svs
        # Example: TCGA-AC-A62V-01Z-00-DX1.2D8994FD-58B8-43C1-B99D-AA964E7DFD60.svs
        
        parts = slide_name.split('.')
        tcga_part = parts[0]
        
        # Extract patient ID (first 3 parts: TCGA-XX-XXXX)
        tcga_parts = tcga_part.split('-')
        patient_id = '-'.join(tcga_parts[:3])
        
        # Extract project (e.g., BRCA from TCGA-XX)
        project = tcga_parts[1] if len(tcga_parts) > 1 else "UNKNOWN"
        
        return {
            "patient_id": patient_id,
            "tcga_project": f"TCGA-{project}",
            "sample_id": tcga_part
        }
    
    def read_rna_file(self, rna_path: Path, gene_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Read TCGA RNA-seq file and extract gene expression"""
        print(f"   Reading: {rna_path.name}")
        
        # Determine if gzipped
        if rna_path.suffix == '.gz':
            import gzip
            df = pd.read_csv(rna_path, sep='\t', compression='gzip', comment='#')
        else:
            df = pd.read_csv(rna_path, sep='\t', comment='#')
        
        # Skip metadata rows (N_unmapped, N_multimapping, etc.)
        df = df[~df['gene_id'].str.startswith('N_')]
        
        # Use tpm_unstranded for expression values (normalized)
        # If not available, use unstranded counts
        if 'tpm_unstranded' in df.columns:
            expr_col = 'tpm_unstranded'
        elif 'unstranded' in df.columns:
            expr_col = 'unstranded'
        else:
            print(f"   ⚠ Could not find expression column, using first numeric column")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            expr_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[-1]
        
        # Create gene expression dictionary
        gene_expr = {}
        
        for _, row in df.iterrows():
            gene_name = row['gene_name'] if 'gene_name' in df.columns else row['gene_id']
            
            # Filter by gene subset if provided
            if gene_subset and gene_name not in gene_subset:
                continue
            
            # Use log2(TPM + 1) transformation
            expr_value = row[expr_col]
            if pd.notna(expr_value):
                # SEQUOIA uses log2(TPM + 1)
                gene_expr[f"rna_{gene_name}"] = np.log2(float(expr_value) + 1)
        
        return gene_expr
    
    def load_manifest(self) -> Optional[pd.DataFrame]:
        """Load manifest file if it exists"""
        print("\n[3/5] Checking for manifest files...")
        
        manifests = list(self.metadata_dir.glob("*_manifest.csv"))
        
        if not manifests:
            print("   ⚠ No manifest files found - will match by filename only")
            return None
        
        # Combine all manifests
        all_manifests = []
        for manifest_path in manifests:
            df = pd.read_csv(manifest_path)
            all_manifests.append(df)
        
        combined = pd.concat(all_manifests, ignore_index=True)
        print(f"   Found {len(combined)} entries in manifest files")
        
        return combined
    
    def match_slides_to_rna(self, slides: List[Path], rna_files: List[Path],
                            manifest: Optional[pd.DataFrame] = None) -> List[dict]:
        """Match slide files to RNA files"""
        print("\n[4/5] Matching slides to RNA data...")
        
        matches = []
        
        # If we have a manifest, use it for matching
        if manifest is not None:
            for slide_path in slides:
                slide_name = slide_path.name
                
                # Find in manifest
                manifest_row = manifest[manifest['slide_file_name'] == slide_name]
                
                if len(manifest_row) == 0:
                    print(f"   ⚠ No manifest entry for {slide_name}")
                    continue
                
                manifest_row = manifest_row.iloc[0]
                
                # Find corresponding RNA file
                rna_file_name = manifest_row['rna_file_name']
                rna_path = self.rna_dir / rna_file_name
                
                if not rna_path.exists():
                    print(f"   ⚠ RNA file not found: {rna_file_name}")
                    continue
                
                matches.append({
                    "slide_path": slide_path,
                    "rna_path": rna_path,
                    "case_id": manifest_row['case_id']
                })
        else:
            # No manifest - try to match by patient ID
            # This is a simplified approach for testing
            print("   ⚠ Using simplified matching (no manifest)")
            
            if len(slides) == len(rna_files):
                for slide_path, rna_path in zip(slides, rna_files):
                    slide_info = self.parse_tcga_slide_name(slide_path.name)
                    matches.append({
                        "slide_path": slide_path,
                        "rna_path": rna_path,
                        "case_id": slide_info['patient_id']
                    })
            else:
                print(f"   ✗ Cannot match: {len(slides)} slides vs {len(rna_files)} RNA files")
                return []
        
        print(f"   Matched {len(matches)} slide-RNA pairs")
        return matches
    
    def build_reference(self, matches: List[dict], 
                       gene_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Build reference.csv dataframe"""
        print("\n[5/5] Building reference.csv...")
        
        if not matches:
            print("   ✗ No matches to process")
            return None
        
        reference_data = []
        
        for match in matches:
            slide_path = match['slide_path']
            rna_path = match['rna_path']
            
            # Parse slide info
            slide_info = self.parse_tcga_slide_name(slide_path.name)
            
            # Read RNA data
            gene_expr = self.read_rna_file(rna_path, gene_subset)
            
            # Combine into row
            # Use .stem to remove .svs extension (SEQUOIA scripts add it back)
            row = {
                "wsi_file_name": slide_path.stem,
                "patient_id": match['case_id'],
                "tcga_project": slide_info['tcga_project']
            }
            row.update(gene_expr)
            
            reference_data.append(row)
        
        df = pd.DataFrame(reference_data)
        
        # Fill missing values with 0
        df = df.fillna(0)
        
        print(f"   Created reference with {len(df)} rows and {len(df.columns)} columns")
        print(f"   Genes: {len([c for c in df.columns if c.startswith('rna_')])}")
        
        return df
    
    def save_reference(self, df: pd.DataFrame, output_path: Path):
        """Save reference.csv"""
        df.to_csv(output_path, index=False)
        print(f"\n   ✓ Saved reference file: {output_path}")
        
        # Also save gene list separately
        gene_cols = [c for c in df.columns if c.startswith('rna_')]
        gene_names = [c.replace('rna_', '') for c in gene_cols]
        
        gene_list_path = output_path.parent / "gene_list_used.txt"
        with open(gene_list_path, 'w') as f:
            for gene in sorted(gene_names):
                f.write(f"{gene}\n")
        
        print(f"   ✓ Saved gene list: {gene_list_path}")
    
    def run(self, gene_subset_file: Optional[str] = None, 
           output_name: str = "reference.csv"):
        """Run complete reference building process"""
        print("\n" + "="*60)
        print("SEQUOIA Reference Builder")
        print("="*60)
        
        # Load gene subset if provided
        gene_subset = None
        if gene_subset_file:
            print(f"\nLoading gene subset from: {gene_subset_file}")
            with open(gene_subset_file, 'r') as f:
                gene_subset = set(line.strip() for line in f if line.strip())
            print(f"   Loaded {len(gene_subset)} genes")
        
        # Get files
        slides = self.get_slide_files()
        rna_files = self.get_rna_files()
        
        if not slides:
            print("\n✗ No slide files found!")
            return
        
        if not rna_files:
            print("\n✗ No RNA files found!")
            return
        
        # Load manifest
        manifest = self.load_manifest()
        
        # Match slides to RNA
        matches = self.match_slides_to_rna(slides, rna_files, manifest)
        
        if not matches:
            print("\n✗ Could not match slides to RNA data!")
            return
        
        # Build reference
        df = self.build_reference(matches, gene_subset)
        
        if df is None or len(df) == 0:
            print("\n✗ Failed to build reference!")
            return
        
        # Save reference
        output_path = self.metadata_dir / output_name
        self.save_reference(df, output_path)
        
        print("\n" + "="*60)
        print("Reference Building Complete!")
        print("="*60)
        print(f"Reference file: {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Total genes: {len([c for c in df.columns if c.startswith('rna_')])}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create SEQUOIA reference.csv from TCGA data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create reference from all genes
  python create_reference_csv.py --data_dir data/raw
  
  # Create reference with specific gene subset
  python create_reference_csv.py --genes models/gene_list.csv --data_dir data/raw
  
  # Specify custom slides directory
  python create_reference_csv.py --data_dir data/raw --slides_dir data/raw/images
  
  # Specify output name
  python create_reference_csv.py --output reference_brca.csv
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory containing rna/ subdirectory"
    )
    parser.add_argument(
        "--slides_dir",
        type=str,
        default=None,
        help="Directory containing slide files (default: data_dir/slides)"
    )
    parser.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Optional: File with gene subset (one gene per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reference.csv",
        help="Output filename (saved in data/metadata/)"
    )
    
    args = parser.parse_args()
    
    builder = ReferenceBuilder(args.data_dir, slides_dir=args.slides_dir)
    builder.run(gene_subset_file=args.genes, output_name=args.output)


if __name__ == "__main__":
    main()
