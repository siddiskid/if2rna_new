#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=rosie_brca88
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/rosie_brca88_%j.out
#SBATCH --error=logs/rosie_brca88_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/st-singha53-1/schiluku/if2rna_new}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Avoid read-only $HOME cache/config paths on HPC compute nodes.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"

# Inputs
REF_FILE="${REF_FILE:-data/hne_data/metadata/tcga_reference_brca_88.csv}"
WSI_DIR="${WSI_DIR:-data/hne_data/raw/images}"

# ROSIE assets
ROSIE_DIR="${ROSIE_DIR:-models/rosie}"
ROSIE_MODEL="${ROSIE_MODEL:-models/rosie/best_model_single.pth}"

# Intermediate/output dirs
ROSIE_INPUT_DIR="${ROSIE_INPUT_DIR:-data/rosie_if/he_png_inputs}"
ROSIE_TIFF_DIR="${ROSIE_TIFF_DIR:-data/rosie_if/rosie_tiff_outputs}"
ROSIE_RGB_DIR="${ROSIE_RGB_DIR:-data/rosie_if/rosie_rgb_outputs}"
ROSIE_REF_FILE="${ROSIE_REF_FILE:-data/hne_data/metadata/rosie_if_reference_brca_88.csv}"

# IF2RNA preprocessing dirs
IF_PATCH_DIR="${IF_PATCH_DIR:-data/rosie_if/patches}"
IF_FEATURE_DIR="${IF_FEATURE_DIR:-data/rosie_if/features}"
RESNET_DIR="${RESNET_DIR:-models/resnet50}"

# IF2RNA inference assets
IF2RNA_CKPT="${IF2RNA_CKPT:-results/if2rna_models/baseline_resnet_log/model_best_4.pt}"
PRED_OUT="${PRED_OUT:-results/rosie_if2rna/predictions_rosie_brca_88_fold4.csv}"
CORR_OUT="${CORR_OUT:-results/rosie_if2rna/correlations_rosie_brca_88_fold4.csv}"

echo "[1/7] Preparing ROSIE PNG inputs from WSI"
python scripts/prepare_rosie_inputs_from_wsi.py \
  --reference_csv "$REF_FILE" \
  --wsi_dir "$WSI_DIR" \
  --output_dir "$ROSIE_INPUT_DIR" \
  --max_side 2048

echo "[2/7] Running ROSIE conversion (H&E -> multiplex TIFF)"
python scripts/run_rosie_conversion.py \
  --rosie_dir "$ROSIE_DIR" \
  --input_dir "$ROSIE_INPUT_DIR" \
  --output_dir "$ROSIE_TIFF_DIR" \
  --model_path "$ROSIE_MODEL"

echo "[3/7] Converting ROSIE TIFF outputs to RGB composites"
python scripts/convert_rosie_tiff_to_rgb.py \
  --input_dir "$ROSIE_TIFF_DIR" \
  --output_dir "$ROSIE_RGB_DIR"

echo "[4/7] Building ROSIE IF reference CSV"
python if2rna_scripts/create_rosie_if_reference.py \
  --source_reference "$REF_FILE" \
  --rosie_image_dir "$ROSIE_RGB_DIR" \
  --output_reference "$ROSIE_REF_FILE" \
  --organ_type BRCA

echo "[5/7] IF patch extraction + feature extraction + kmeans"
python if2rna_scripts/run_if_preprocessing.py \
  --ref_file "$ROSIE_REF_FILE" \
  --image_dir "$ROSIE_RGB_DIR" \
  --patch_dir "$IF_PATCH_DIR" \
  --feature_dir "$IF_FEATURE_DIR" \
  --feat_type resnet \
  --model_dir "$RESNET_DIR"

echo "[6/7] IF2RNA inference on ROSIE-converted inputs"
python if2rna_scripts/run_if2rna_inference.py \
  --reference_csv "$ROSIE_REF_FILE" \
  --feature_dir "$IF_FEATURE_DIR" \
  --checkpoint "$IF2RNA_CKPT" \
  --output_predictions "$PRED_OUT" \
  --output_correlations "$CORR_OUT"

echo "[7/7] Done"
echo "Predictions: $PRED_OUT"
echo "Correlations: $CORR_OUT"
