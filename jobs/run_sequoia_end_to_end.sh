#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=sequoia_e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/sequoia_e2e_%j.out
#SBATCH --error=logs/sequoia_e2e_%j.err

################################################################################
# SEQUOIA End-to-End Pipeline
#
# Steps:
# 1) Preprocess slides (patches + features + kmeans)
# 2) Ensure/download SEQUOIA model
# 3) Run inference
# 4) Evaluate predictions
#
# Notes:
# - This script does NOT create directories; required paths must already exist.
################################################################################

set -euo pipefail

echo "========================================================================"
echo "SEQUOIA End-to-End Pipeline"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================================================"

nvidia-smi || true

# Setup
PROJECT_DIR="${PROJECT_DIR:-/scratch/st-singha53-1/schiluku/if2rna_new}"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"

cd "$PROJECT_DIR"
source "$VENV_PATH"

module load gcc/9.4.0 || true
module load openslide/3.4.1 || true

export PYTHONPATH="$PROJECT_DIR/sequoia-pub:${PYTHONPATH:-}"

# Config (can be overridden via environment variables)
REF_FILE="${REF_FILE:-data/hne_data/metadata/tcga_reference.csv}"
GENE_LIST="${GENE_LIST:-models/gene_list.csv}"
WSI_PATH="${WSI_PATH:-data/hne_data/raw/images}"
FEAT_TYPE="${FEAT_TYPE:-uni}"
CANCER_TYPE="${CANCER_TYPE:-BRCA}"
FOLD="${FOLD:-0}"
FEATURE_DIR="${FEATURE_DIR:-data/processed/features}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sequoia}"
MODEL_ROOT="${MODEL_ROOT:-models/sequoia}"
STRICT_SAMPLE_MATCH="${STRICT_SAMPLE_MATCH:-1}"

REF_TAG="$(basename "$REF_FILE")"
REF_TAG="${REF_TAG%.*}"
MODEL_DIR="$MODEL_ROOT/${CANCER_TYPE,,}-${FOLD}"
PREDICTIONS_FILE="$OUTPUT_DIR/predictions_${CANCER_TYPE,,}-${FOLD}.csv"

# Required path checks
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
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "ERROR: Output directory not found: $OUTPUT_DIR"
  exit 1
fi
if [ ! -d "$(dirname "$FEATURE_DIR")" ]; then
  echo "ERROR: Parent of feature directory not found: $(dirname "$FEATURE_DIR")"
  exit 1
fi
if [ ! -d "logs" ]; then
  echo "ERROR: logs directory not found: logs"
  exit 1
fi

echo ""
echo "Configuration:"
echo "  PROJECT_DIR:  $PROJECT_DIR"
echo "  REF_FILE:     $REF_FILE"
echo "  WSI_PATH:     $WSI_PATH"
echo "  GENE_LIST:    $GENE_LIST"
echo "  FEAT_TYPE:    $FEAT_TYPE"
echo "  CANCER_TYPE:  $CANCER_TYPE"
echo "  FOLD:         $FOLD"
echo "  MODEL_DIR:    $MODEL_DIR"
echo "  FEATURE_DIR:  $FEATURE_DIR"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"
echo "  REF_TAG:      $REF_TAG"
echo "  STRICT_MATCH: $STRICT_SAMPLE_MATCH"
echo ""

REF_ROWS=$(python - << PY_EOF
import pandas as pd
print(len(pd.read_csv("$REF_FILE")))
PY_EOF
)

echo "Expected samples from reference: $REF_ROWS"
echo ""

################################################################################
# Step 1: Preprocessing
################################################################################

CHECKPOINT_PREPROCESS="logs/.checkpoint_preprocess_${REF_TAG}_${FEAT_TYPE}.done"
CHECKPOINT_INFERENCE="logs/.checkpoint_inference_${CANCER_TYPE,,}-${FOLD}_${REF_TAG}.done"

# Prevent stale files from previous runs from being reused silently.
rm -f "$OUTPUT_DIR/predictions_${CANCER_TYPE,,}-${FOLD}.csv"
rm -f "$OUTPUT_DIR/predictions_${CANCER_TYPE,,}-${FOLD}_fixed.csv"
rm -f "$OUTPUT_DIR/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv"

# If inference output is removed, force inference to rerun even if stale checkpoint exists.
if [ -f "$CHECKPOINT_INFERENCE" ] && [ ! -f "$PREDICTIONS_FILE" ]; then
  rm -f "$CHECKPOINT_INFERENCE"
fi

if [ -f "$CHECKPOINT_PREPROCESS" ]; then
  echo "[Step 1] Preprocessing already complete (checkpoint found)"
else
  echo "========================================================================"
  echo "[Step 1/4] Preprocessing"
  echo "========================================================================"

  export HF_HUB_OFFLINE=1

  python scripts/preprocess_slides.py \
    --ref_file "$REF_FILE" \
    --wsi_path "$WSI_PATH" \
    --output_dir data/processed \
    --feat_type "$FEAT_TYPE" \
    --steps patch features kmeans \
    --patch_size 256 \
    --max_patches_per_slide 2000 \
    --max_patches_for_features 4000 \
    --n_clusters 100

  touch "$CHECKPOINT_PREPROCESS"
  echo "✓ Preprocessing complete"
fi

# SEQUOIA inference expects "cluster_features". The upstream kmeans script is
# often tied to resnet_features; for UNI runs we build cluster_features here.
if [ "$FEAT_TYPE" = "uni" ]; then
  echo ""
  echo "========================================================================"
  echo "[Step 1b/4] Ensuring cluster_features for UNI"
  echo "========================================================================"

  python - << 'PY_EOF'
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import KMeans

feature_root = Path("data/processed/features")
created = 0
checked = 0

for h5_path in feature_root.rglob("*.h5"):
    checked += 1
    with h5py.File(h5_path, "r+") as f:
        if "cluster_features" in f:
            continue
        if "uni_features" not in f:
            continue

        feats = f["uni_features"][:]
        if feats.shape[0] == 0:
            continue

        n_clusters = min(100, feats.shape[0])
        km = KMeans(n_clusters=n_clusters, random_state=99, n_init=10)
        km.fit(feats)
        f.create_dataset("cluster_features", data=km.cluster_centers_.astype(np.float32))
        created += 1

print(f"Checked feature files: {checked}")
print(f"Created cluster_features: {created}")
PY_EOF
fi

# Hard check: cluster_features must exist for inference.
CLUSTER_COUNT=$(python - << 'PY_EOF'
from pathlib import Path
import h5py

count = 0
for h5_path in Path("data/processed/features").rglob("*.h5"):
    with h5py.File(h5_path, "r") as f:
        if "cluster_features" in f:
            count += 1
print(count)
PY_EOF
)

if [ "$CLUSTER_COUNT" -eq 0 ]; then
  echo "ERROR: No feature files with cluster_features found. Inference would be invalid."
  exit 1
fi

################################################################################
# Step 2: Model download/verification
################################################################################

echo ""
echo "========================================================================"
echo "[Step 2/4] Model"
echo "========================================================================"

if [ ! -d "$MODEL_DIR" ]; then
  unset HF_HUB_OFFLINE
  python scripts/download_sequoia_model.py \
    --cancer_types "$CANCER_TYPE" \
    --folds "$FOLD" \
    --output_dir "$MODEL_ROOT"
  export HF_HUB_OFFLINE=1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: Model directory still missing after download: $MODEL_DIR"
  exit 1
fi

echo "✓ Model ready: $MODEL_DIR"

################################################################################
# Step 3: Inference
################################################################################

if [ -f "$CHECKPOINT_INFERENCE" ]; then
  echo "[Step 3] Inference already complete (checkpoint found)"
else
  echo ""
  echo "========================================================================"
  echo "[Step 3/4] Inference"
  echo "========================================================================"

  python scripts/run_sequoia_inference.py \
    --model_dir "$MODEL_DIR" \
    --ref_file "$REF_FILE" \
    --feature_dir "$FEATURE_DIR" \
    --gene_list "$GENE_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --feat_type "$FEAT_TYPE"

  touch "$CHECKPOINT_INFERENCE"
  echo "✓ Inference complete"
fi

# Hard check: predictions must exist and contain at least 1 row.
if [ ! -f "$PREDICTIONS_FILE" ]; then
  echo "ERROR: Inference did not create predictions file: $PREDICTIONS_FILE"
  exit 1
fi

PRED_ROWS=$(python - << PY_EOF
import pandas as pd
df = pd.read_csv("$PREDICTIONS_FILE")
print(len(df))
PY_EOF
)

if [ "$PRED_ROWS" -eq 0 ]; then
  echo "ERROR: Inference produced 0 prediction rows. Evaluation aborted."
  exit 1
fi

if [ "$STRICT_SAMPLE_MATCH" = "1" ] && [ "$PRED_ROWS" -ne "$REF_ROWS" ]; then
  echo "ERROR: Prediction row count ($PRED_ROWS) does not match reference rows ($REF_ROWS)."
  echo "       Set STRICT_SAMPLE_MATCH=0 to allow partial outputs."
  exit 1
fi

################################################################################
# Step 4: Evaluation
################################################################################

echo ""
echo "========================================================================"
echo "[Step 4/4] Evaluation"
echo "========================================================================"

python scripts/evaluate_predictions.py \
  --predictions_dir "$OUTPUT_DIR" \
  --gene_list "$GENE_LIST" \
  --reference "$REF_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --folds "$FOLD"

echo ""
echo "========================================================================"
echo "Pipeline Complete"
echo "========================================================================"
echo "Predictions: $PREDICTIONS_FILE"
echo "Correlations: $OUTPUT_DIR/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv"
echo "Finished at: $(date)"
echo "========================================================================"
