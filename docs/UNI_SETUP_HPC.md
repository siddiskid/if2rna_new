# Using UNI on Sockeye HPC (Offline GPU Nodes)

## Problem
GPU compute nodes on Sockeye don't have internet access, but UNI needs to be downloaded from Hugging Face Hub.

## Solution
Download the model on the login node (which has internet), then use the cached model on GPU nodes.

---

## One-Time Setup

### Step 1: Accept UNI License
Visit https://huggingface.co/MahmoodLab/uni and click "Agree and access repository"

### Step 2: Get Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or use existing)
3. Copy the token

### Step 3: Login on Sockeye
```bash
# SSH to Sockeye login node
ssh sockeye.arc.ubc.ca

# Login to Hugging Face
huggingface-cli login
# Paste your token when prompted
```

### Step 4: Download UNI Model
```bash
# Navigate to your project
cd ~/if2rna_new

# Run download script (this runs on login node with internet)
python scripts/download_uni_model.py
```

This will download ~1.2GB and cache it at `~/.cache/huggingface/hub/`

---

## Running GPU Jobs

Now you can submit GPU jobs - they'll use the cached model:

```bash
# Submit preprocessing job
sbatch jobs/preprocess_slides.job

# Or run interactively
salloc --time=2:00:00 --mem=32G --gpus=1 --account=st-username-gpu
python sequoia-pub/pre_processing/compute_features_hdf5.py \
    --feat_type uni \
    --ref_file data/metadata/reference.csv \
    --patch_data_path data/processed/patches \
    --feature_path data/processed/features
```

The script automatically runs in offline mode when on compute nodes.

---

## Troubleshooting

### Error: "UNI model not found in cache"
Solution: Run `python scripts/download_uni_model.py` on login node first

### Error: "You have not accepted the license"  
Solution: Visit https://huggingface.co/MahmoodLab/uni and accept

### Error: "Invalid token"
Solution: Run `huggingface-cli login` again with a valid token

### Check if model is cached
```bash
ls ~/.cache/huggingface/hub/ | grep mahmoodlab
# Should show: models--MahmoodLab--uni
```

---

## Model Details

- **Architecture**: Vision Transformer Large (ViT-L/16)
- **Parameters**: ~307M
- **Input**: 224×224 RGB images
- **Output**: 1024-dimensional features
- **Training**: 100M+ histopathology images from 100K+ slides
- **Paper**: "Towards a general-purpose foundation model for computational pathology" (Nature Medicine, 2024)
