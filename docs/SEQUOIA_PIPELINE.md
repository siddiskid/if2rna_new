# SEQUOIA End-to-End Pipeline

Complete automated pipeline for running SEQUOIA gene expression prediction from raw slides to evaluation.

## Overview

The pipeline consists of 4 main steps:

1. **Preprocessing**: Extract patches → UNI features → K-means clustering
2. **Model Download**: Get pretrained SEQUOIA model from HuggingFace
3. **Inference**: Run predictions on test slides
4. **Evaluation**: Calculate correlations with ground truth RNA-seq

## Prerequisites

### One-Time Setup (on login node)

```bash
# 1. Download UNI model
python scripts/download_uni_model.py

# 2. Create reference CSV
python scripts/create_reference_csv.py \
    --slide_dir data/raw/tcga_slides \
    --rna_dir data/raw/tcga_rna \
    --output data/metadata/tcga_reference.csv
```

### Required Files

- **Slides**: `data/raw/tcga_slides/*.svs` (or other WSI formats)
- **RNA-seq**: `data/raw/tcga_rna/*.csv` (gene expression data)
- **Reference**: `data/metadata/tcga_reference.csv` (links slides to RNA-seq)
- **Gene list**: `sequoia-pub/examples/gene_list.csv` (genes to predict)

## Usage

### Quick Start

Submit the default job (BRCA fold 0):

```bash
sbatch jobs/run_sequoia_end_to_end.sh
```

### Custom Configuration

Edit the configuration section in the script:

```bash
# In jobs/run_sequoia_end_to_end.sh
CANCER_TYPE="BRCA"  # Change to: BRCA, COAD, GBMLGG, HNSC, KIRC, etc.
FOLD=0              # Change to: 0, 1, 2, 3, 4
```

Or create a custom job script:

```bash
#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=sequoia_coad
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate

# Run with different cancer type
export CANCER_TYPE="COAD"
export FOLD=2

bash jobs/run_sequoia_end_to_end.sh
```

### Resume from Checkpoint

If the job fails or times out, simply resubmit - it will resume from the last completed step:

```bash
sbatch jobs/run_sequoia_end_to_end.sh
```

Checkpoints are stored in `logs/.checkpoint_*`

To force rerun from scratch:

```bash
rm logs/.checkpoint_*
sbatch jobs/run_sequoia_end_to_end.sh
```

## Pipeline Steps in Detail

### Step 1: Preprocessing (~4-8 hours)

Extracts patches and UNI features from WSI slides, then clusters features.

**Input**: 
- WSI slides (`data/raw/tcga_slides/*.svs`)
- Reference CSV (`data/metadata/tcga_reference.csv`)

**Output**:
- Patches: `data/processed/patches/*.h5`
- Features: `data/processed/features/*.h5` (1024-dim UNI embeddings)
- K-means: Features file contains `uni_kmeans` (100 clusters per slide)

**Resources**: 8 CPUs, 1 GPU, 32 GB RAM

### Step 2: Model Download (~1-2 minutes)

Downloads pretrained SEQUOIA model from HuggingFace.

**Input**: Cancer type + fold number

**Output**: `models/sequoia/{cancer}-{fold}/`
- `model.pth` - Model weights
- `config.json` - Model configuration
- `gene_list.txt` - Genes the model predicts

**Available models**:
- BRCA (Breast Cancer): folds 0-4
- COAD (Colon Cancer): folds 0-4  
- GBMLGG (Brain Cancer): folds 0-4
- HNSC (Head/Neck): folds 0-4
- KIRC (Kidney): folds 0-4
- LUAD (Lung Adenocarcinoma): folds 0-4
- LUSC (Lung Squamous): folds 0-4

### Step 3: Inference (~30 min - 2 hours)

Runs SEQUOIA model to predict gene expression for each slide.

**Input**:
- Features: `data/processed/features/*.h5`
- Model: `models/sequoia/{cancer}-{fold}/`
- Gene list: `sequoia-pub/examples/gene_list.csv`

**Output**: `results/sequoia/predictions_{cancer}-{fold}.csv`
- Columns: `wsi_file_name`, `patient_id`, `pred_gene_0`, `pred_gene_1`, ...

**Resources**: 1 GPU, 32 GB RAM

### Step 4: Evaluation (~1-5 minutes)

Compares predictions with ground truth RNA-seq and calculates per-gene correlations.

**Input**:
- Predictions: `results/sequoia/predictions_{cancer}-{fold}.csv`
- Ground truth: `data/metadata/tcga_reference.csv` (contains RNA-seq values)
- Gene list: `sequoia-pub/examples/gene_list.csv`

**Output**:
- `predictions_{cancer}-{fold}_fixed.csv` - Predictions with proper gene names
- `predictions_{cancer}-{fold}_correlations.csv` - Per-gene Pearson correlations

**Metrics**: Pearson correlation per gene, mean/median correlation, MAE

## Expected Results

Based on published SEQUOIA paper (Nature 2024):

| Cancer Type | Median Correlation | Top Genes |
|-------------|-------------------|-----------|
| BRCA        | 0.72 - 0.76       | ESR1, PGR, ERBB2 |
| COAD        | 0.68 - 0.73       | CDX2, FABP1 |
| KIRC        | 0.70 - 0.75       | CA9, NDUFA4L2 |
| GBMLGG      | 0.65 - 0.71       | IDH1, ATRX |

## Output Structure

```
results/sequoia/
├── predictions_brca-0.csv              # Raw predictions
├── predictions_brca-0_fixed.csv        # With gene names
└── predictions_brca-0_correlations.csv # Per-gene correlations

data/processed/
├── patches/
│   └── slide_001.h5                    # Extracted patches
├── features/
│   └── slide_001.h5                    # UNI features + kmeans
└── masks/
    └── slide_001_mask.png              # Tissue masks

models/sequoia/
└── brca-0/
    ├── model.pth                       # Model weights
    ├── config.json                     # Configuration
    └── gene_list.txt                   # Predicted genes
```

## Troubleshooting

### Error: "UNI model not found in cache"

**Solution**: Run on login node first:
```bash
python scripts/download_uni_model.py
```

### Error: "Reference file not found"

**Solution**: Create reference CSV:
```bash
python scripts/create_reference_csv.py \
    --slide_dir data/raw/tcga_slides \
    --rna_dir data/raw/tcga_rna \
    --output data/metadata/tcga_reference.csv
```

### Error: "Model download failed"

**Solution**: Check HuggingFace authentication:
```bash
huggingface-cli login
```

### Job times out

**Solution**: Increase time limit or resume from checkpoint:
```bash
# Increase time to 48 hours
#SBATCH --time=48:00:00

# Or just resubmit - it will resume
sbatch jobs/run_sequoia_end_to_end.sh
```

### Want to skip preprocessing

If you already have features, delete preprocessing checkpoint:

```bash
rm logs/.checkpoint_preprocess_done
# Or manually skip in script
touch logs/.checkpoint_preprocess_done
sbatch jobs/run_sequoia_end_to_end.sh
```

## Running Multiple Folds

Create a job array to run all 5 folds:

```bash
#!/bin/bash
#SBATCH --array=0-4
#SBATCH --job-name=sequoia_array

FOLD=$SLURM_ARRAY_TASK_ID
export CANCER_TYPE="BRCA"

# Edit script to use $FOLD variable instead of hardcoded value
```

Or submit individually:

```bash
for fold in {0..4}; do
    # Edit FOLD= in script
    sed -i "s/FOLD=.*/FOLD=$fold/" jobs/run_sequoia_end_to_end.sh
    sbatch jobs/run_sequoia_end_to_end.sh
done
```

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f logs/sequoia_e2e_JOBID.out
```

Check which step is running:
```bash
ls logs/.checkpoint_*
```

## Resource Usage

| Step          | Time    | GPU | CPUs | Memory |
|---------------|---------|-----|------|--------|
| Preprocessing | 4-8h    | Yes | 8    | 32 GB  |
| Model DL      | 1-2min  | No  | 1    | 4 GB   |
| Inference     | 0.5-2h  | Yes | 4    | 32 GB  |
| Evaluation    | 1-5min  | No  | 1    | 8 GB   |
| **Total**     | **5-10h** | **Yes** | **8** | **64 GB** |

Adjust based on dataset size:
- Small dataset (<50 slides): 16 GB RAM, 4h
- Medium dataset (50-200 slides): 32 GB RAM, 12h  
- Large dataset (>200 slides): 64 GB RAM, 24h

## Contact

For issues with:
- **SEQUOIA model**: See https://github.com/Gevaert-Lab/SEQUOIA
- **UNI features**: See https://github.com/mahmoodlab/UNI
- **This pipeline**: Contact your supervisor or check project README
