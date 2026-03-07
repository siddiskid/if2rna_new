#!/usr/bin/env python3
"""
Download UNI model from Hugging Face Hub to local cache.

Run this on Sockeye LOGIN NODE (has internet) before submitting GPU jobs.
The model will be cached so GPU compute nodes can use it offline.

Usage:
    python scripts/download_uni_model.py

Prerequisites:
    1. Accept license at https://huggingface.co/MahmoodLab/uni
    2. Run: huggingface-cli login
"""

import sys
import os

def download_uni():
    print("\n" + "="*70)
    print("Downloading UNI Model from Hugging Face Hub")
    print("="*70)
    
    print("\nChecking prerequisites...")
    
    # Check if hugging face token is configured
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if not token:
            print("\n✗ Error: Not logged in to Hugging Face")
            print("\nPlease run: huggingface-cli login")
            print("Then paste your token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        print("✓ Hugging Face authentication found")
    except ImportError:
        print("\n✗ Error: huggingface_hub not installed")
        print("\nPlease run: pip install huggingface_hub")
        sys.exit(1)
    
    # Check timm
    try:
        import timm
        print(f"✓ timm version {timm.__version__} installed")
    except ImportError:
        print("\n✗ Error: timm not installed")
        print("\nPlease run: pip install timm")
        sys.exit(1)
    
    print("\n" + "-"*70)
    print("Downloading UNI model (this may take a few minutes)...")
    print("-"*70)
    
    try:
        # This will download and cache the model
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )
        
        print("\n✓ UNI model downloaded successfully!")
        
        # Try to get cache location (optional)
        try:
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                print(f"\nCache location: {cache_dir}")
                # Try to find UNI model directory
                uni_dirs = [d for d in os.listdir(cache_dir) if 'mahmood' in d.lower() and 'uni' in d.lower()]
                if uni_dirs:
                    uni_path = os.path.join(cache_dir, uni_dirs[0])
                    # Get size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(uni_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)
                    print(f"Model size: {total_size / (1024**3):.2f} GB")
        except:
            pass  # Cache info is optional
        
        # Show UNI model info
        print("\nModel Info:")
        print(f"  - Architecture: Vision Transformer Large")
        print(f"  - Parameters: ~307M")
        print(f"  - Input size: 224x224")
        print(f"  - Output dims: 1024")
        print(f"  - Training: 100M+ histopathology images")
        
        print("\n" + "="*70)
        print("✓ Setup complete! You can now run GPU jobs offline.")
        print("="*70)
        
        print("\nNext steps:")
        print("1. Submit GPU job: sbatch jobs/preprocess_slides.job")
        print("2. The model will be loaded from cache (no internet needed)")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nPossible issues:")
        print("1. License not accepted: https://huggingface.co/MahmoodLab/uni")
        print("2. Invalid token: Run 'huggingface-cli login' again")
        print("3. Network issues: Try again or check internet connection")
        sys.exit(1)

if __name__ == "__main__":
    download_uni()
