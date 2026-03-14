#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=sequoia_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/sequoia_smoke_%j.out
#SBATCH --error=logs/sequoia_smoke_%j.err

################################################################################
# SEQUOIA Smoke Test (Fast Validation)
#
# What it does:
# 1) Creates a tiny reference file from the first N rows
# 2) Runs preprocessing with reduced patch limits
# 3) Downloads one SEQUOIA model fold
# 4) Runs inference
# 5) Runs evaluation
#
# Intended use:
# - Verify environment, paths, model loading, and end-to-end outputs
# - Not intended for final performance numbers
################################################################################

set -e

echo "========================================================================"
echo "SEQUOIA Smoke Test"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================================================"

echo "[Setup] Activating environment"
VENV_PATH="${VENV_PATH:-.venv/bin/activate}"
source "$VENV_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p data/processed/{patches,features,masks}
mkdir -p models/sequoia
mkdir -p results/sequoia

# Configurable inputs
REF_FILE="${REF_FILE:-data/metadata/tcga_reference.csv}"
SMOKE_REF_FILE="${SMOKE_REF_FILE:-data/metadata/tcga_reference_smoke.csv}"
GENE_LIST="${GENE_LIST:-sequoia-pub/examples/gene_list.csv}"
WSI_PATH="${WSI_PATH:-data/raw/tcga_slides}"
FEAT_TYPE="${FEAT_TYPE:-uni}"
CANCER_TYPE="${CANCER_TYPE:-BRCA}"
FOLD="${FOLD:-0}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-10}"
PATCH_SIZE="${PATCH_SIZE:-256}"
MAX_PATCHES_PER_SLIDE="${MAX_PATCHES_PER_SLIDE:-200}"
MAX_PATCHES_FOR_FEATURES="${MAX_PATCHES_FOR_FEATURES:-400}"
N_CLUSTERS="${N_CLUSTERS:-100}"

echo "Configuration:"
echo "  Project dir: $PROJECT_DIR"
echo "  Reference: $REF_FILE"
echo "  Smoke ref: $SMOKE_REF_FILE"
echo "  WSI path: $WSI_PATH"
echo "  Gene list: $GENE_LIST"
echo "  Feature type: $FEAT_TYPE"
echo "  Cancer/Fold: $CANCER_TYPE/$FOLD"
echo "  Smoke samples: $SMOKE_SAMPLES"

if [ ! -f "$REF_FILE" ]; then
  echo "ERROR: Reference file not found: $REF_FILE"
  exit 1
fi
if [ ! -d "$WSI_PATH" ]; then
  echo "ERROR: WSI directory not found: $WSI_PATH"
  exit 1
fi
if [ ! -f "$GENE_LIST" ]; then
  echo "ERROR: Gene list not found: $GENE_LIST"
  exit 1
fi

MODEL_DIR="models/sequoia/${CANCER_TYPE,,}-${FOLD}"
PREDICTIONS_FILE="results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}.csv"


echo "[1/5] Building smoke reference file"
head -n 1 "$REF_FILE" > "$SMOKE_REF_FILE"
tail -n +2 "$REF_FILE" | head -n "$SMOKE_SAMPLES" >> "$SMOKE_REF_FILE"


echo "[2/5] Preprocessing smoke subset"
export HF_HUB_OFFLINE=1
python scripts/preprocess_slides.py \
  --ref_file "$SMOKE_REF_FILE" \
  --wsi_path "$WSI_PATH" \
  --output_dir data/processed \
  --feat_type "$FEAT_TYPE" \
  --steps patch features kmeans \
  --patch_size "$PATCH_SIZE" \
  --max_patches_per_slide "$MAX_PATCHES_PER_SLIDE" \
  --max_patches_for_features "$MAX_PATCHES_FOR_FEATURES" \
  --n_clusters "$N_CLUSTERS"


echo "[3/5] Downloading model"
unset HF_HUB_OFFLINE
python scripts/download_sequoia_model.py \
  --cancer_types "$CANCER_TYPE" \
  --folds "$FOLD" \
  --output_dir models/sequoia
export HF_HUB_OFFLINE=1


echo "[4/5] Running inference"
python scripts/run_sequoia_inference.py \
  --model_dir "$MODEL_DIR" \
  --ref_file "$SMOKE_REF_FILE" \
  --feature_dir data/processed/features \
  --gene_list "$GENE_LIST" \
  --output_dir results/sequoia \
  --feat_type "$FEAT_TYPE"


echo "[5/5] Evaluating predictions"
python scripts/evaluate_predictions.py \
  --predictions_dir results/sequoia \
  --gene_list "$GENE_LIST" \
  --reference "$SMOKE_REF_FILE" \
  --output_dir results/sequoia \
  --folds "$FOLD"


echo "========================================================================"
echo "Smoke test complete"
echo "Predictions: $PREDICTIONS_FILE"
echo "Correlations: results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv"
echo "Finished at: $(date)"
echo "========================================================================"
