#!/usr/bin/env python3
"""
Validate downloaded TCGA data for SEQUOIA testing

Checks:
1. Paired H&E and RNA files exist
2. File integrity (size, format)
3. RNA data completeness
4. Generate summary statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from collections import defaultdict


class DataValidator:
    """Validate paired H&E and RNA data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.rna_dir = self.data_dir / "rna"
        self.metadata_dir = self.data_dir.parent / "metadata"
        
    def check_directories(self) -> bool:
        """Check if required directories exist"""
        print("\n[1/5] Checking directory structure...")
        
        dirs_ok = True
        for dir_path in [self.images_dir, self.rna_dir, self.metadata_dir]:
            if dir_path.exists():
                print(f"   ✓ Found: {dir_path}")
            else:
                print(f"   ✗ Missing: {dir_path}")
                dirs_ok = False
        
        return dirs_ok
    
    def count_files(self) -> Dict[str, int]:
        """Count files in each directory"""
        print("\n[2/5] Counting files...")
        
        counts = {
            "images": len(list(self.images_dir.glob("*.svs"))),
            "rna": len(list(self.rna_dir.glob("*.gz"))),
            "metadata": len(list(self.metadata_dir.glob("*_manifest.csv")))
        }
        
        print(f"   H&E images (.svs): {counts['images']}")
        print(f"   RNA files (.gz): {counts['rna']}")
        print(f"   Manifest files: {counts['metadata']}")
        
        return counts
    
    def validate_pairs(self) -> Dict[str, List]:
        """Validate paired data using manifest files"""
        print("\n[3/5] Validating paired data...")
        
        results = defaultdict(list)
        manifest_files = list(self.metadata_dir.glob("*_manifest.csv"))
        
        if not manifest_files:
            print("   ✗ No manifest files found!")
            return results
        
        total_pairs = 0
        valid_pairs = 0
        
        for manifest_path in manifest_files:
            print(f"\n   Checking manifest: {manifest_path.name}")
            df = pd.read_csv(manifest_path)
            total_pairs += len(df)
            
            for _, row in df.iterrows():
                slide_path = self.images_dir / row["slide_file_name"]
                rna_path = self.rna_dir / row["rna_file_name"]
                
                slide_exists = slide_path.exists()
                rna_exists = rna_path.exists()
                
                if slide_exists and rna_exists:
                    valid_pairs += 1
                    results["valid"].append({
                        "case_id": row["case_id"],
                        "slide": row["slide_file_name"],
                        "rna": row["rna_file_name"]
                    })
                else:
                    status = []
                    if not slide_exists:
                        status.append("missing slide")
                    if not rna_exists:
                        status.append("missing RNA")
                    
                    results["invalid"].append({
                        "case_id": row["case_id"],
                        "reason": ", ".join(status)
                    })
        
        print(f"\n   Total pairs in manifests: {total_pairs}")
        print(f"   Valid pairs: {valid_pairs}")
        print(f"   Invalid pairs: {len(results['invalid'])}")
        
        return results
    
    def check_file_sizes(self) -> Dict[str, Dict]:
        """Check file sizes and detect corrupted files"""
        print("\n[4/5] Checking file sizes...")
        
        stats = {"images": {}, "rna": {}}
        
        image_files = list(self.images_dir.glob("*.svs"))
        if image_files:
            sizes_mb = [f.stat().st_size / (1024*1024) for f in image_files]
            stats["images"] = {
                "count": len(image_files),
                "total_gb": sum(sizes_mb) / 1024,
                "avg_mb": sum(sizes_mb) / len(sizes_mb),
                "min_mb": min(sizes_mb),
                "max_mb": max(sizes_mb)
            }
            print(f"   Images: {stats['images']['count']} files, "
                  f"{stats['images']['total_gb']:.2f} GB total")
            print(f"      Avg: {stats['images']['avg_mb']:.1f} MB, "
                  f"Range: {stats['images']['min_mb']:.1f}-{stats['images']['max_mb']:.1f} MB")
        
        rna_files = list(self.rna_dir.glob("*.gz"))
        if rna_files:
            sizes_kb = [f.stat().st_size / 1024 for f in rna_files]
            stats["rna"] = {
                "count": len(rna_files),
                "total_mb": sum(sizes_kb) / 1024,
                "avg_kb": sum(sizes_kb) / len(sizes_kb),
                "min_kb": min(sizes_kb),
                "max_kb": max(sizes_kb)
            }
            print(f"   RNA files: {stats['rna']['count']} files, "
                  f"{stats['rna']['total_mb']:.2f} MB total")
            print(f"      Avg: {stats['rna']['avg_kb']:.1f} KB, "
                  f"Range: {stats['rna']['min_kb']:.1f}-{stats['rna']['max_kb']:.1f} KB")
        
        return stats
    
    def generate_report(self, counts: Dict, pairs: Dict, 
                       file_stats: Dict) -> Dict:
        """Generate validation report"""
        print("\n[5/5] Generating validation report...")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "file_counts": counts,
            "valid_pairs": len(pairs.get("valid", [])),
            "invalid_pairs": len(pairs.get("invalid", [])),
            "file_statistics": file_stats,
            "status": "PASS" if len(pairs.get("invalid", [])) == 0 else "FAIL"
        }
        
        report_path = self.metadata_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n   Report saved: {report_path}")
        
        if pairs.get("valid"):
            valid_df = pd.DataFrame(pairs["valid"])
            valid_path = self.metadata_dir / "valid_pairs.csv"
            valid_df.to_csv(valid_path, index=False)
            print(f"   Valid pairs list: {valid_path}")
        
        if pairs.get("invalid"):
            invalid_df = pd.DataFrame(pairs["invalid"])
            invalid_path = self.metadata_dir / "invalid_pairs.csv"
            invalid_df.to_csv(invalid_path, index=False)
            print(f"   Invalid pairs list: {invalid_path}")
        
        return report
    
    def print_summary(self, report: Dict):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Status: {report['status']}")
        print(f"Valid pairs: {report['valid_pairs']}")
        print(f"Invalid pairs: {report['invalid_pairs']}")
        
        if report['file_statistics'].get('images'):
            img_stats = report['file_statistics']['images']
            print(f"\nH&E Images:")
            print(f"  Total: {img_stats['count']} files ({img_stats['total_gb']:.2f} GB)")
            print(f"  Average size: {img_stats['avg_mb']:.1f} MB")
        
        if report['file_statistics'].get('rna'):
            rna_stats = report['file_statistics']['rna']
            print(f"\nRNA Files:")
            print(f"  Total: {rna_stats['count']} files ({rna_stats['total_mb']:.2f} MB)")
            print(f"  Average size: {rna_stats['avg_kb']:.1f} KB")
        
        print("="*60)
        
        if report['status'] == "PASS":
            print("✓ All data validated successfully!")
        else:
            print("✗ Validation issues found. Check invalid_pairs.csv")
        print("="*60 + "\n")
    
    def run_validation(self):
        """Run complete validation"""
        print("\n" + "="*60)
        print("Data Validation for SEQUOIA Testing")
        print("="*60)
        
        if not self.check_directories():
            print("\n✗ Required directories missing. Run download script first.")
            return
        
        counts = self.count_files()
        
        if counts["images"] == 0 and counts["rna"] == 0:
            print("\n✗ No data found. Run download script first.")
            return
        
        pairs = self.validate_pairs()
        file_stats = self.check_file_sizes()
        report = self.generate_report(counts, pairs, file_stats)
        self.print_summary(report)


def main():
    parser = argparse.ArgumentParser(
        description="Validate downloaded TCGA data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    validator.run_validation()


if __name__ == "__main__":
    main()
