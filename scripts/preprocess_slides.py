#!/usr/bin/env python3
"""
SEQUOIA Preprocessing Wrapper Script

This script runs the complete SEQUOIA preprocessing pipeline:
1. Extract patches from WSI slides
2. Compute features (ResNet or UNI)
3. Compute k-means clusters (100 clusters per slide)

Usage:
    # Full pipeline with ResNet features
    python preprocess_slides.py --ref_file data/metadata/reference.csv --feat_type resnet
    
    # Full pipeline with UNI features (recommended)
   python preprocess_slides.py --ref_file data/metadata/reference.csv --feat_type uni --uni_model_dir models/uni
    
    # Only patch extraction
    python preprocess_slides.py --ref_file data/metadata/reference.csv --steps patch
    
    # Resume from feature extraction
    python preprocess_slides.py --ref_file data/metadata/reference.csv --feat_type resnet --steps features kmeans
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import os


class SEQUOIAPreprocessor:
    """Wrapper for SEQUOIA preprocessing pipeline"""
    
    def __init__(self, ref_file: str, wsi_path: str = "data/raw/slides",
                 output_dir: str = "data/processed", sequoia_dir: str = "sequoia-pub"):
        self.ref_file = Path(ref_file)
        self.wsi_path = Path(wsi_path)
        self.output_dir = Path(output_dir)
        self.sequoia_dir = Path(sequoia_dir)
        
        # Output directories
        self.patches_dir = self.output_dir / "patches"
        self.features_dir = self.output_dir / "features"
        self.masks_dir = self.output_dir / "masks"
        
        # SEQUOIA script paths
        self.patch_script = self.sequoia_dir / "pre_processing" / "patch_gen_hdf5.py"
        self.features_script = self.sequoia_dir / "pre_processing" / "compute_features_hdf5.py"
        self.kmeans_script = self.sequoia_dir / "pre_processing" / "kmean_features.py"
        
        # Create directories
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
    
    def verify_setup(self) -> bool:
        """Verify SEQUOIA setup"""
        print("\n[Verification] Checking setup...")
        
        if not self.ref_file.exists():
            print(f"   ✗ Reference file not found: {self.ref_file}")
            print(f"      Run: python scripts/create_reference_csv.py")
            return False
        
        if not self.sequoia_dir.exists():
            print(f"   ✗ SEQUOIA directory not found: {self.sequoia_dir}")
            print(f"      Run: bash scripts/setup_sequoia.sh")
            return False
        
        if not self.patch_script.exists():
            print(f"   ✗ Patch extraction script not found: {self.patch_script}")
            return False
        
        if not self.wsi_path.exists():
            print(f"   ✗ WSI directory not found: {self.wsi_path}")
            return False
        
        print("   ✓ All requirements verified")
        return True
    
    def run_patch_extraction(self, patch_size: int = 256, 
                            max_patches_per_slide: int = 2000,
                            parallel: bool = False) -> bool:
        """Step 1: Extract patches from WSI slides"""
        print("\n" + "="*70)
        print("[Step 1/3] Patch Extraction")
        print("="*70)
        
        cmd = [
            "python", str(self.patch_script),
            "--ref_file", str(self.ref_file),
            "--wsi_path", str(self.wsi_path),
            "--patch_path", str(self.patches_dir),
            "--mask_path", str(self.masks_dir),
            "--patch_size", str(patch_size),
            "--max_patches_per_slide", str(max_patches_per_slide),
            "--parallel", "1" if parallel else "0"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        # Set PYTHONPATH and offline mode for all subprocess calls
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.sequoia_dir.absolute()) + ':' + env.get('PYTHONPATH', '')
        env['HF_HUB_OFFLINE'] = '1'
        
        try:
            result = subprocess.run(cmd, check=True, env=env)
            print("\n✓ Patch extraction complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Patch extraction failed: {e}")
            return False
    
    def run_feature_extraction(self, feat_type: str = "uni",
                              uni_model_dir: str = None,
                              max_patches: int = 4000) -> bool:
        """Step 2: Extract features using ResNet or UNI
        
        UNI is loaded directly from Hugging Face Hub (MahmoodLab/uni).
        Requires: Accept license at huggingface.co/MahmoodLab/uni and login via huggingface-cli.
        """
        print("\n" + "="*70)
        print(f"[Step 2/3] Feature Extraction (feat_type: {feat_type})")
        print("="*70)
        
        if feat_type == "uni":
            print(f"\n✓ Using UNI (Universal histopathology foundation model)")
            print("  Model: MahmoodLab/uni from Hugging Face")
            print("  Output: 1024-dimensional features")
            print("\n  Prerequisites:")
            print("  1. Accept license: https://huggingface.co/MahmoodLab/uni")
            print("  2. Login: huggingface-cli login\n")
        
        cmd = [
            "python", str(self.features_script),
            "--feat_type", feat_type,
            "--ref_file", str(self.ref_file),
            "--patch_data_path", str(self.patches_dir),
            "--feature_path", str(self.features_dir),
            "--max_patch_number", str(max_patches)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        # Set PYTHONPATH to include sequoia-pub directory
        # Set HF_HUB_OFFLINE=1 to use cached models without internet
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.sequoia_dir.absolute()) + ':' + env.get('PYTHONPATH', '')
        env['HF_HUB_OFFLINE'] = '1'  # Use cached models only (for HPC compute nodes)
        
        try:
            result = subprocess.run(cmd, check=True, env=env)
            print("\n✓ Feature extraction complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Feature extraction failed: {e}")
            return False
    
    def run_kmeans_clustering(self, feat_type: str = "resnet", 
                             n_clusters: int = 100) -> bool:
        """Step 3: Compute k-means features"""
        print("\n" + "="*70)
        print(f"[Step 3/3] K-means Clustering (n_clusters: {n_clusters})")
        print("="*70)
        
        if not self.kmeans_script.exists():
            print(f"\n⚠ K-means script not found: {self.kmeans_script}")
            print("   Creating simplified k-means computation...")
            return self._run_simple_kmeans(feat_type, n_clusters)
        
        cmd = [
            "python", str(self.kmeans_script),
            "--ref_file", str(self.ref_file),
            "--patch_data_path", str(self.patches_dir),
            "--feature_path", str(self.features_dir),
            "--num_clusters", str(n_clusters)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        # Set PYTHONPATH to include sequoia-pub directory
        # Set HF_HUB_OFFLINE=1 to use cached models without internet
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.sequoia_dir.absolute()) + ':' + env.get('PYTHONPATH', '')
        env['HF_HUB_OFFLINE'] = '1'  # Use cached models only (for HPC compute nodes)
        
        try:
            result = subprocess.run(cmd, check=True, env=env)
            print("\n✓ K-means clustering complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ K-means clustering failed: {e}")
            return False
    
    def _run_simple_kmeans(self, feat_type: str, n_clusters: int) -> bool:
        """Fallback: Simple k-means computation if script not found"""
        print("\n   Running simplified k-means clustering...")
        
        try:
            import h5py
            import numpy as np
            from sklearn.cluster import KMeans
            import pandas as pd
            from tqdm import tqdm
            
            # Read reference file
            ref_df = pd.read_csv(self.ref_file)
            
            for _, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
                wsi_name = row['wsi_file_name'].replace('.svs', '')
                project = row.get('tcga_project', 'UNKNOWN')
                
                # Find feature file
                feat_file = self.features_dir / project / wsi_name / f"{wsi_name}.h5"
                
                if not feat_file.exists():
                    print(f"   ⚠ Feature file not found: {feat_file}")
                    continue
                
                # Read features
                with h5py.File(feat_file, 'r+') as f:
                    if f"{feat_type}_features" not in f:
                        print(f"   ⚠ {feat_type}_features not found in {feat_file.name}")
                        continue
                    
                    features = f[f"{feat_type}_features"][:]
                    
                    # Check if cluster features already exist
                    if "cluster_features" in f:
                        print(f"   ✓ Cluster features already exist for {wsi_name}")
                        continue
                    
                    # Compute k-means
                    if len(features) < n_clusters:
                        n_clusters_use = len(features)
                        print(f"   ⚠ Only {len(features)} patches, using {n_clusters_use} clusters")
                    else:
                        n_clusters_use = n_clusters
                    
                    kmeans = KMeans(n_clusters=n_clusters_use, random_state=0, n_init=10)
                    cluster_labels = kmeans.fit_predict(features)
                    
                    # Create cluster features (one-hot encoding)
                    cluster_features = np.zeros((len(features), n_clusters_use))
                    cluster_features[np.arange(len(features)), cluster_labels] = 1
                    
                    # Save to h5 file
                    f.create_dataset("cluster_features", data=cluster_features)
                    print(f"   ✓ Computed k-means for {wsi_name}")
            
            print("\n✓ K-means clustering complete!")
            return True
            
        except Exception as e:
            print(f"\n✗ K-means computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self, steps: List[str], feat_type: str = "resnet",
                    uni_model_dir: str = None, patch_size: int = 256,
                    max_patches_per_slide: int = 2000, 
                    max_patches_for_features: int = 4000,
                    n_clusters: int = 100, parallel: bool = False):
        """Run complete or partial preprocessing pipeline"""
        
        print("\n" + "="*70)
        print("SEQUOIA Preprocessing Pipeline")
        print("="*70)
        print(f"Reference file: {self.ref_file}")
        print(f"WSI directory: {self.wsi_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Steps to run: {', '.join(steps)}")
        print(f"Feature type: {feat_type}")
        print("="*70)
        
        if not self.verify_setup():
            print("\n✗ Setup verification failed!")
            return False
        
        success = True
        
        # Step 1: Patch extraction
        if "patch" in steps or "patches" in steps:
            if not self.run_patch_extraction(patch_size, max_patches_per_slide, parallel):
                print("\n✗ Pipeline failed at patch extraction")
                return False
        
        # Step 2: Feature extraction
        if "features" in steps or "feature" in steps:
            if not self.run_feature_extraction(feat_type, uni_model_dir, max_patches_for_features):
                print("\n✗ Pipeline failed at feature extraction")
                return False
        
        # Step 3: K-means clustering
        if "kmeans" in steps or "clustering" in steps:
            if not self.run_kmeans_clustering(feat_type, n_clusters):
                print("\n✗ Pipeline failed at k-means clustering")
                return False
        
        print("\n" + "="*70)
        print("✓ Preprocessing Pipeline Complete!")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  Patches: {self.patches_dir}")
        print(f"  Features: {self.features_dir}")
        print(f"  Masks: {self.masks_dir}")
        print("\nNext step: Run SEQUOIA inference")
        print("  python scripts/run_sequoia_inference.py")
        print("="*70 + "\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="SEQUOIA preprocessing wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with ResNet
  python preprocess_slides.py --ref_file data/metadata/reference.csv --feat_type resnet
  
  # Full pipeline with UNI (requires UNI model download)
  python preprocess_slides.py --ref_file data/metadata/reference.csv --feat_type uni
  
  # Only patch extraction
  python preprocess_slides.py --ref_file data/metadata/reference.csv --steps patch
  
  # Resume from features
  python preprocess_slides.py --ref_file data/metadata/reference.csv --steps features kmeans
        """
    )
    
    parser.add_argument("--ref_file", type=str, required=True,
                       help="Path to reference.csv file")
    parser.add_argument("--wsi_path", type=str, default="data/raw/slides",
                       help="Directory containing WSI files")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--sequoia_dir", type=str, default="sequoia-pub",
                       help="SEQUOIA repository directory")
    parser.add_argument("--steps", nargs="+", 
                       default=["patch", "features", "kmeans"],
                       choices=["patch", "patches", "feature", "features", "kmeans", "clustering"],
                       help="Steps to run")
    parser.add_argument("--feat_type", type=str, default="resnet",
                       choices=["resnet", "uni"],
                       help="Feature extraction type")
    parser.add_argument("--uni_model_dir", type=str, default=None,
                       help="Directory containing UNI model weights (required for feat_type=uni)")
    parser.add_argument("--patch_size", type=int, default=256,
                       help="Patch size (default: 256)")
    parser.add_argument("--max_patches_per_slide", type=int, default=2000,
                       help="Max patches to extract per slide")
    parser.add_argument("--max_patches_for_features", type=int, default=4000,
                       help="Max patches to use for feature extraction")
    parser.add_argument("--n_clusters", type=int, default=100,
                       help="Number of k-means clusters")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing for patch extraction")
    
    args = parser.parse_args()
    
    preprocessor = SEQUOIAPreprocessor(
        args.ref_file,
        args.wsi_path,
        args.output_dir,
        args.sequoia_dir
    )
    
    success = preprocessor.run_pipeline(
        steps=args.steps,
        feat_type=args.feat_type,
        uni_model_dir=args.uni_model_dir,
        patch_size=args.patch_size,
        max_patches_per_slide=args.max_patches_per_slide,
        max_patches_for_features=args.max_patches_for_features,
        n_clusters=args.n_clusters,
        parallel=args.parallel
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
