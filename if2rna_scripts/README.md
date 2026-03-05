# IF2RNA Scripts

Scripts for building the IF2RNA model - predicting RNA expression from immunofluorescence images using SEQUOIA architecture.

## Overview

IF2RNA adapts SEQUOIA's architecture to work with NanoString GeoMx WTA immunofluorescence data instead of H&E whole slide images.

## Workflow

### Phase 1: Data Preparation
Prepare IF images and RNA expression data for model training.

#### 1. Extract IF Images
```bash
python if2rna_scripts/extract_if_images.py --extract_segments
```
Extracts ROI images from NanoString zip files.
- **Input:** `data/if_data/{organ}/workflow_and_count_files/workflow/roi_report/*.zip`
- **Output:** `data/if_images/{organ}/{slide}/*.png`

#### 2. Create Reference CSV
```bash
python if2rna_scripts/create_if_reference_csv.py --use_segments
```
Matches IF images to gene expression from NanoString Excel files.
- **Input:** 
  - Images: `data/if_images/`
  - Expression: `data/if_data/{organ}/workflow_and_count_files/count/Export4_NormalizationQ3.xlsx`
- **Output:** `data/metadata/if_reference.csv` (942 samples × 15,830 genes)

#### 3. Validate Data
```bash
python if2rna_scripts/validate_if_data.py
```
Checks that all images and RNA data are properly matched.

---

### Phase 2: Preprocessing (SEQUOIA Pipeline)
Extract features from IF images using the same pipeline as SEQUOIA.

#### Important: Download Models First (for HPC without internet)

Since Sockeye doesn't have internet access, download pretrained models on your local machine first:

**ResNet50:**
```bash
# On your local machine (with internet)
python if2rna_scripts/download_resnet50.py --output_dir models/resnet50

# Transfer to HPC
scp models/resnet50/resnet50_imagenet.pth username@sockeye.arc.ubc.ca:/path/to/if2rna_new/models/resnet50/
```

**UNI (optional, if using):**
Follow UNI installation instructions and download model, then transfer to HPC.

---

#### Option A: Run Full Pipeline (Recommended)
```bash
# With ResNet50 features (faster, recommended for baseline)
python if2rna_scripts/run_if_preprocessing.py \
    --ref_file data/metadata/if_reference.csv \
    --feat_type resnet \
    --model_dir models/resnet50

# With UNI features (requires UNI model download first)
python if2rna_scripts/run_if_preprocessing.py \
    --ref_file data/metadata/if_reference.csv \
    --feat_type uni \
    --model_dir models/uni
```

#### Option B: Run Steps Individually

**Step 1: Extract Patches**
```bash
python if2rna_scripts/preprocess_if_patches.py \
    --ref_file data/metadata/if_reference.csv \
    --patch_size 256 \
    --max_patches 100
```
- Extracts 256×256 patches from each IF image
- Up to 100 patches per image
- **Output:** `data/if_patches/{organ}/{sample}/{sample}.h5`

**Step 2: Extract Features**
```bash
# ResNet50 features (recommended baseline)
python if2rna_scripts/preprocess_if_features.py \
    --ref_file data/metadata/if_reference.csv \
    --patch_dir data/if_patches \
    --feat_type resnet \
    --model_dir models/resnet50

# OR UNI features (requires UNI model download)
python if2rna_scripts/preprocess_if_features.py \
    --ref_file data/metadata/if_reference.csv \
    --patch_dir data/if_patches \
    --feat_type uni \
    --model_dir models/uni
```
- Extracts feature vectors from each patch
- ResNet50: 2048-dim features
- UNI: 1024-dim features
- **Output:** Adds `resnet_features` or `uni_features` to HDF5 files

**Step 3: K-means Clustering**
```bash
python if2rna_scripts/preprocess_if_kmeans.py \
    --ref_file data/metadata/if_reference.csv \
    --feature_dir data/if_features \
    --feat_type resnet \
    --num_clusters 100
```
- Clusters features into 100 representative prototypes per image
- **Output:** Adds `cluster_features` (100 × feature_dim) to HDF5 files

---

### Phase 3: Model Training
Train IF2RNA model using SEQUOIA architecture.

```bash
python if2rna_scripts/train_if2rna.py \
    --ref_file data/metadata/if_reference.csv \
    --feature_dir data/if_features \
    --model_type vis \
    --train \
    --num_epochs 200
```

**Parameters:**
- `--model_type`: `vis` (linearized transformer) or `vit` (standard transformer)
- `--depth`: Transformer depth (default: 6)
- `--num_heads`: Number of attention heads (default: 16)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--k`: Number of cross-validation folds (default: 5)

---

## Data Structure

After preprocessing, your data structure should look like:

```
data/
  if_images/                    # [Phase 1] Extracted IF images
    Colon/
      hu_colon_001/
        hu_colon_001_025247T1_ROI_001_PanCK+.png
        ...
  metadata/
    if_reference.csv            # [Phase 1] Images + RNA expression
  if_patches/                   # [Phase 2.1] Extracted patches
    Colon/
      hu_colon_001_025247T1_ROI_001_PanCK+/
        hu_colon_001_025247T1_ROI_001_PanCK+.h5  # Contains patch_0, patch_1, ...
  if_features/                  # [Phase 2.2-2.3] Features + clusters
    Colon/
      hu_colon_001_025247T1_ROI_001_PanCK+/
        hu_colon_001_025247T1_ROI_001_PanCK+.h5  # Contains:
                                                  # - resnet_features (N × 2048)
                                                  # - cluster_features (100 × 2048)
```

---

## Command Summary (On HPC)

Run these commands in sequence on Sockeye:

```bash
# Navigate to project
cd /path/to/if2rna_new

# Phase 1: Data preparation (already done ✓)
python if2rna_scripts/extract_if_images.py --extract_segments
python if2rna_scripts/create_if_reference_csv.py --use_segments
python if2rna_scripts/validate_if_data.py

# Phase 2: Preprocessing
# Note: Make sure you've transferred the ResNet50 model first!
python if2rna_scripts/run_if_preprocessing.py \
    --ref_file data/metadata/if_reference.csv \
    --feat_type resnet \
    --model_dir models/resnet50

# Phase 3: Training (coming next)
python if2rna_scripts/train_if2rna.py \
    --ref_file data/metadata/if_reference.csv \
    --feature_dir data/if_features \
    --model_type vis \
    --train
```

---

## Steps to Run on HPC

**On your local machine (with internet):**
```bash
# Download ResNet50 model
python if2rna_scripts/download_resnet50.py --output_dir models/resnet50

# Transfer to HPC
scp -r models/resnet50 username@sockeye.arc.ubc.ca:/path/to/if2rna_new/models/
```

**On Sockeye HPC:**
```bash
# Run preprocessing
python if2rna_scripts/run_if_preprocessing.py \
    --ref_file data/metadata/if_reference.csv \
    --feat_type resnet \
    --model_dir models/resnet50
```

---

## Notes

- **Baseline approach:** Uses RGB composite IF images with ResNet50/UNI (no modifications to SEQUOIA architecture)
- **Future optimization:** Can modify ResNet/UNI first layer to handle multi-channel IF data directly
- **Spatial resolution:** Each sample is a single ROI/segment, preserving spatial information from NanoString data
- **Training data:** 942 samples (173 Colon + 203 Kidney + 379 Liver + 187 Lymph Node)
- **Genes:** 15,830 genes from WTA panel

---

## Troubleshooting

**"No patches extracted"**
- Image may be too small or mostly background
- Try reducing `--max_patches` or `--patch_size`

**"Features not found"**
- Make sure patches were extracted first
- Check that patch HDF5 files exist in `data/if_patches/`

**CUDA out of memory**
- Reduce `--batch_size` during training
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

**Liver has no data**
- Liver zip files are empty in the source data
- This is expected - model can still train on other organs
