#!/usr/bin/env python3
"""
Download SEQUOIA pre-trained models from HuggingFace

This script downloads:
1. SEQUOIA model weights for specified cancer types and folds
2. Model configuration files
3. Gene list for predictions

Usage:
    python download_sequoia_model.py --cancer_types BRCA --folds 0 1 2 3 4
    python download_sequoia_model.py --cancer_types BRCA LUAD --folds 0
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import sys

try:
    from huggingface_hub import snapshot_download, login, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("\n✗ Error: huggingface_hub not installed!")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)


class SEQUOIADownloader:
    """Download SEQUOIA pre-trained models from HuggingFace"""
    
    AVAILABLE_CANCERS = [
        'BRCA', 'LUAD', 'LUSC', 'PRAD', 'KIRC', 'KIRP', 'UCEC',
        'HNSC', 'COAD', 'STAD', 'BLCA', 'LIHC', 'CESC', 'THCA',
        'OV', 'PAAD'
    ]
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "sequoia"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def check_login(self):
        """Check if user is logged in to HuggingFace"""
        print("\n[1/3] Checking HuggingFace authentication...")
        
        try:
            # Try to access a public model to check if we're logged in
            # SEQUOIA models are public, but let's check anyway
            from huggingface_hub import whoami
            try:
                user_info = whoami()
                print(f"   ✓ Logged in as: {user_info['name']}")
                return True
            except:
                print("   ⚠ Not logged in (this may be fine for public models)")
                print("   If download fails, login with: huggingface-cli login")
                return False
        except Exception as e:
            print(f"   ⚠ Could not verify login status: {e}")
            return False
    
    def download_model(self, cancer_type: str, fold: int) -> bool:
        """Download a specific SEQUOIA model"""
        cancer_lower = cancer_type.lower()
        model_id = f"gevaertlab/sequoia-{cancer_lower}-{fold}"
        model_path = self.models_dir / f"{cancer_lower}-{fold}"
        
        print(f"\n   Downloading: {model_id}")
        
        try:
            # Download the entire model repository
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            print(f"   ✓ Downloaded to: {model_path}")
            return True
            
        except HfHubHTTPError as e:
            if "404" in str(e):
                print(f"   ✗ Model not found: {model_id}")
                print(f"      (This cancer type or fold may not be available)")
            else:
                print(f"   ✗ Download failed: {e}")
            return False
        except Exception as e:
            print(f"   ✗ Error downloading {model_id}: {e}")
            return False
    
    def download_gene_list(self) -> bool:
        """Download the gene list from SEQUOIA repository"""
        print("\n   Downloading gene list...")
        
        gene_list_url = "https://raw.githubusercontent.com/gevaertlab/sequoia-pub/master/evaluation/gene_list.csv"
        gene_list_path = self.output_dir / "gene_list.csv"
        
        try:
            import requests
            response = requests.get(gene_list_url)
            response.raise_for_status()
            
            with open(gene_list_path, 'w') as f:
                f.write(response.text)
            
            print(f"   ✓ Gene list saved to: {gene_list_path}")
            return True
            
        except Exception as e:
            print(f"   ✗ Could not download gene list: {e}")
            print(f"      Download manually from: {gene_list_url}")
            return False
    
    def verify_downloads(self, cancer_types: List[str], folds: List[int]) -> dict:
        """Verify downloaded models"""
        print("\n[3/3] Verifying downloads...")
        
        results = {
            "successful": [],
            "failed": []
        }
        
        for cancer in cancer_types:
            for fold in folds:
                cancer_lower = cancer.lower()
                model_path = self.models_dir / f"{cancer_lower}-{fold}"
                
                # Check if model directory exists and has files
                if model_path.exists():
                    files = list(model_path.glob("*"))
                    if files:
                        # Look for model weights file
                        has_weights = any(
                            f.suffix in ['.pt', '.pth', '.bin', '.safetensors']
                            for f in files
                        )
                        
                        if has_weights:
                            results["successful"].append(f"{cancer}-{fold}")
                            print(f"   ✓ {cancer}-{fold}: OK ({len(files)} files)")
                        else:
                            results["failed"].append(f"{cancer}-{fold}")
                            print(f"   ✗ {cancer}-{fold}: No model weights found")
                    else:
                        results["failed"].append(f"{cancer}-{fold}")
                        print(f"   ✗ {cancer}-{fold}: Empty directory")
                else:
                    results["failed"].append(f"{cancer}-{fold}")
                    print(f"   ✗ {cancer}-{fold}: Not downloaded")
        
        return results
    
    def save_download_manifest(self, cancer_types: List[str], 
                               folds: List[int], results: dict):
        """Save manifest of downloaded models"""
        manifest = {
            "cancer_types": cancer_types,
            "folds": folds,
            "successful_downloads": results["successful"],
            "failed_downloads": results["failed"],
            "models_directory": str(self.models_dir),
            "total_models": len(results["successful"])
        }
        
        manifest_path = self.output_dir / "download_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n   Manifest saved: {manifest_path}")
    
    def download_models(self, cancer_types: List[str], folds: List[int]):
        """Download SEQUOIA models for specified cancer types and folds"""
        print("\n" + "="*60)
        print("SEQUOIA Model Downloader")
        print("="*60)
        print(f"Cancer types: {', '.join(cancer_types)}")
        print(f"Folds: {', '.join(map(str, folds))}")
        print(f"Output directory: {self.models_dir}")
        print("="*60)
        
        # Validate cancer types
        invalid = [c for c in cancer_types if c.upper() not in self.AVAILABLE_CANCERS]
        if invalid:
            print(f"\n⚠ Warning: Unknown cancer types: {', '.join(invalid)}")
            print(f"   Available types: {', '.join(self.AVAILABLE_CANCERS)}")
        
        # Check authentication
        self.check_login()
        
        # Download models
        print("\n[2/3] Downloading models...")
        total = len(cancer_types) * len(folds)
        downloaded = 0
        
        for cancer in cancer_types:
            for fold in folds:
                if self.download_model(cancer, fold):
                    downloaded += 1
        
        # Download gene list
        self.download_gene_list()
        
        # Verify downloads
        results = self.verify_downloads(cancer_types, folds)
        
        # Save manifest
        self.save_download_manifest(cancer_types, folds, results)
        
        # Print summary
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print(f"Total models requested: {total}")
        print(f"Successfully downloaded: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results["successful"]:
            print(f"\n✓ Downloaded models:")
            for model in results["successful"]:
                print(f"   • {model}")
        
        if results["failed"]:
            print(f"\n✗ Failed downloads:")
            for model in results["failed"]:
                print(f"   • {model}")
        
        print("\n" + "="*60)
        print(f"Models location: {self.models_dir}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download SEQUOIA pre-trained models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all folds for BRCA
  python download_sequoia_model.py --cancer_types BRCA --folds 0 1 2 3 4
  
  # Download fold 0 for multiple cancer types
  python download_sequoia_model.py --cancer_types BRCA LUAD KIRC --folds 0
  
  # Download specific fold for testing
  python download_sequoia_model.py --cancer_types BRCA --folds 0

Available cancer types:
  BRCA, LUAD, LUSC, PRAD, KIRC, KIRP, UCEC, HNSC, 
  COAD, STAD, BLCA, LIHC, CESC, THCA, OV, PAAD
        """
    )
    
    parser.add_argument(
        "--cancer_types",
        nargs="+",
        required=True,
        help="Cancer types to download (e.g., BRCA LUAD)"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0],
        help="Model folds to download (0-4, default: 0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Output directory for models (default: models)"
    )
    
    args = parser.parse_args()
    
    # Validate folds
    if any(f < 0 or f > 4 for f in args.folds):
        print("Error: Folds must be between 0 and 4")
        sys.exit(1)
    
    # Convert cancer types to uppercase
    cancer_types = [c.upper() for c in args.cancer_types]
    
    downloader = SEQUOIADownloader(args.output_dir)
    downloader.download_models(cancer_types, args.folds)


if __name__ == "__main__":
    main()
