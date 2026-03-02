#!/usr/bin/env python3
"""
Download paired H&E and RNA-seq data from TCGA via GDC API

This script downloads:
1. Diagnostic H&E whole-slide images (.svs files)
2. RNA-seq gene expression data (HTSeq counts)
3. Metadata linking images to RNA data

Usage:
    python download_tcga_data.py --cancer_types BRCA LUAD --num_samples 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
import pandas as pd
from tqdm import tqdm


class TCGADownloader:
    """Download paired H&E and RNA data from TCGA via GDC API"""
    
    GDC_API_URL = "https://api.gdc.cancer.gov"
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.rna_dir = self.output_dir / "rna"
        self.metadata_dir = self.output_dir.parent / "metadata"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.rna_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def query_gdc(self, endpoint: str, filters: dict, fields: List[str], 
                  size: int = 100) -> dict:
        """Query GDC API"""
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": size
        }
        
        response = requests.get(
            f"{self.GDC_API_URL}/{endpoint}",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_slide_files(self, project_id: str, limit: int = 50) -> List[Dict]:
        """Get H&E slide files for a project"""
        print(f"\n[1/4] Querying H&E slides for {project_id}...")
        
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
                {"op": "in", "content": {"field": "files.experimental_strategy", "value": ["Diagnostic Slide"]}},
                {"op": "in", "content": {"field": "files.data_format", "value": ["SVS"]}}
            ]
        }
        
        fields = [
            "file_id",
            "file_name",
            "file_size",
            "cases.submitter_id",
            "cases.samples.submitter_id",
            "cases.samples.sample_type"
        ]
        
        result = self.query_gdc("files", filters, fields, size=limit)
        files = result["data"]["hits"]
        
        print(f"   Found {len(files)} H&E slides")
        return files
    
    def get_rna_files(self, case_ids: List[str], project_id: str) -> List[Dict]:
        """Get RNA-seq files for specific cases"""
        print(f"\n[2/4] Querying RNA-seq data for {len(case_ids)} cases...")
        
        # Try HTSeq - Counts first
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
                {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["HTSeq - Counts"]}},
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}}
            ]
        }
        
        fields = [
            "file_id",
            "file_name",
            "cases.submitter_id",
            "cases.samples.submitter_id"
        ]
        
        result = self.query_gdc("files", filters, fields, size=1000)
        files = result["data"]["hits"]
        
        # If no HTSeq files found, try STAR - Counts
        if len(files) == 0:
            print("   No HTSeq files found, trying STAR - Counts...")
            filters["content"][2] = {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}}
            result = self.query_gdc("files", filters, fields, size=1000)
            files = result["data"]["hits"]
        
        # If still no files, try any Gene Expression Quantification
        if len(files) == 0:
            print("   Trying any Gene Expression Quantification workflow...")
            filters = {
                "op": "and",
                "content": [
                    {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
                    {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
                    {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}}
                ]
            }
            result = self.query_gdc("files", filters, fields, size=1000)
            files = result["data"]["hits"]
        
        print(f"   Found {len(files)} RNA-seq files")
        return files
    
    def download_file(self, file_id: str, output_path: Path, 
                     desc: str = "Downloading") -> bool:
        """Download a file from GDC"""
        if output_path.exists():
            print(f"   Skipping (already exists): {output_path.name}")
            return True
        
        url = f"{self.GDC_API_URL}/data/{file_id}"
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=desc, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"   Downloaded: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"   Error downloading {file_id}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def pair_files(self, slide_files: List[Dict], 
                   rna_files: List[Dict]) -> List[Dict]:
        """Pair H&E slides with RNA-seq data"""
        print(f"\n[3/4] Pairing H&E slides with RNA-seq data...")
        
        rna_by_case = {}
        for rna in rna_files:
            case_id = rna["cases"][0]["submitter_id"]
            if case_id not in rna_by_case:
                rna_by_case[case_id] = []
            rna_by_case[case_id].append(rna)
        
        pairs = []
        for slide in slide_files:
            case_id = slide["cases"][0]["submitter_id"]
            
            if case_id in rna_by_case:
                for rna in rna_by_case[case_id]:
                    pairs.append({
                        "case_id": case_id,
                        "slide_file_id": slide["file_id"],
                        "slide_file_name": slide["file_name"],
                        "slide_size_mb": slide["file_size"] / (1024*1024),
                        "rna_file_id": rna["file_id"],
                        "rna_file_name": rna["file_name"],
                        "sample_id": slide["cases"][0]["samples"][0]["submitter_id"]
                    })
        
        print(f"   Created {len(pairs)} H&E-RNA pairs")
        return pairs
    
    def download_pairs(self, pairs: List[Dict], max_pairs: Optional[int] = None):
        """Download paired data"""
        print(f"\n[4/4] Downloading paired data...")
        
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        downloaded = {"slides": 0, "rna": 0, "failed": 0}
        
        for i, pair in enumerate(pairs, 1):
            print(f"\n--- Pair {i}/{len(pairs)}: {pair['case_id']} ---")
            
            slide_path = self.images_dir / pair["slide_file_name"]
            rna_path = self.rna_dir / pair["rna_file_name"]
            
            slide_ok = self.download_file(
                pair["slide_file_id"], 
                slide_path,
                f"Slide {i}/{len(pairs)}"
            )
            
            rna_ok = self.download_file(
                pair["rna_file_id"],
                rna_path,
                f"RNA {i}/{len(pairs)}"
            )
            
            if slide_ok:
                downloaded["slides"] += 1
            if rna_ok:
                downloaded["rna"] += 1
            if not (slide_ok and rna_ok):
                downloaded["failed"] += 1
            
            time.sleep(0.5)
        
        return downloaded, pairs
    
    def save_metadata(self, pairs: List[Dict], project_id: str):
        """Save metadata about downloaded pairs"""
        df = pd.DataFrame(pairs)
        manifest_path = self.metadata_dir / f"{project_id}_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"\nSaved manifest: {manifest_path}")
        
        summary = {
            "project": project_id,
            "total_pairs": len(pairs),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.metadata_dir / f"{project_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_path}")
    
    def download_cancer_type(self, cancer_type: str, num_samples: int = 50):
        """Download data for one cancer type"""
        project_id = f"TCGA-{cancer_type}"
        print(f"\n{'='*60}")
        print(f"Downloading: {project_id}")
        print(f"{'='*60}")
        
        slide_files = self.get_slide_files(project_id, limit=num_samples * 2)
        
        if not slide_files:
            print(f"No slides found for {project_id}")
            return
        
        case_ids = [f["cases"][0]["submitter_id"] for f in slide_files]
        rna_files = self.get_rna_files(case_ids, project_id)
        
        if not rna_files:
            print(f"No RNA data found for {project_id}")
            return
        
        pairs = self.pair_files(slide_files, rna_files)
        
        if not pairs:
            print(f"No pairs found for {project_id}")
            return
        
        stats, downloaded_pairs = self.download_pairs(pairs, max_pairs=num_samples)
        
        self.save_metadata(downloaded_pairs, project_id)
        
        print(f"\n{'='*60}")
        print(f"Download Summary for {project_id}:")
        print(f"  Slides downloaded: {stats['slides']}")
        print(f"  RNA files downloaded: {stats['rna']}")
        print(f"  Failed: {stats['failed']}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download paired H&E and RNA-seq data from TCGA"
    )
    parser.add_argument(
        "--cancer_types",
        nargs="+",
        required=True,
        help="Cancer types (e.g., BRCA LUAD KIRC)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples per cancer type (default: 50)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TCGA H&E + RNA Data Downloader")
    print("="*60)
    print(f"Cancer types: {', '.join(args.cancer_types)}")
    print(f"Samples per type: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    downloader = TCGADownloader(args.output_dir)
    
    for cancer_type in args.cancer_types:
        try:
            downloader.download_cancer_type(cancer_type, args.num_samples)
        except Exception as e:
            print(f"\nError processing {cancer_type}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Download complete!")
    print(f"Images: {downloader.images_dir}")
    print(f"RNA data: {downloader.rna_dir}")
    print(f"Metadata: {downloader.metadata_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
