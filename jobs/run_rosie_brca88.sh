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
ROSIE_WSI_LEVEL="${ROSIE_WSI_LEVEL:--1}"
ROSIE_TARGET_DOWNSAMPLE="${ROSIE_TARGET_DOWNSAMPLE:-16}"
ROSIE_MAX_SIDE="${ROSIE_MAX_SIDE:-4096}"
SKIP_ROSIE_INPUT_PREP="${SKIP_ROSIE_INPUT_PREP:-0}"
SKIP_ROSIE_CONVERSION="${SKIP_ROSIE_CONVERSION:-0}"
SKIP_ROSIE_RGB_CONVERSION="${SKIP_ROSIE_RGB_CONVERSION:-0}"
TILE_NATIVE="${TILE_NATIVE:-0}"
ROSIE_TILE_INPUT_DIR="${ROSIE_TILE_INPUT_DIR:-data/rosie_if/he_tile_inputs}"
ROSIE_TILE_RGB_DIR="${ROSIE_TILE_RGB_DIR:-data/rosie_if/rosie_rgb_tiles}"
ROSIE_SLIDE_RGB_DIR="${ROSIE_SLIDE_RGB_DIR:-data/rosie_if/rosie_rgb_slides}"
ROSIE_TILE_SIZE="${ROSIE_TILE_SIZE:-512}"
ROSIE_MAX_TILES="${ROSIE_MAX_TILES:-64}"
ROSIE_TARGET_MPP="${ROSIE_TARGET_MPP:-0.5}"

# IF2RNA preprocessing dirs
IF_PATCH_DIR="${IF_PATCH_DIR:-data/rosie_if/patches}"
IF_FEATURE_DIR="${IF_FEATURE_DIR:-data/rosie_if/features}"
RESNET_DIR="${RESNET_DIR:-models/resnet50}"

# IF2RNA inference assets
IF2RNA_CKPT="${IF2RNA_CKPT:-results/if2rna_models/baseline_resnet_log/model_best_4.pt}"
IF2RNA_GENE_PKL="${IF2RNA_GENE_PKL:-results/if2rna_models/baseline_resnet_log/test_results.pkl}"
PRED_OUT="${PRED_OUT:-results/rosie_if2rna/predictions_rosie_brca_88_fold4.csv}"
CORR_OUT="${CORR_OUT:-results/rosie_if2rna/correlations_rosie_brca_88_fold4.csv}"

if [ "$SKIP_ROSIE_INPUT_PREP" = "1" ]; then
  echo "[1/7] Skipping ROSIE PNG input prep (SKIP_ROSIE_INPUT_PREP=1)"
else
  if [ "$TILE_NATIVE" = "1" ]; then
    echo "[1/7] Preparing ROSIE TILE inputs from WSI (tile-native mode)"
    python scripts/prepare_rosie_tile_inputs_from_wsi.py \
      --reference_csv "$REF_FILE" \
      --wsi_dir "$WSI_DIR" \
      --output_dir "$ROSIE_TILE_INPUT_DIR" \
      --target_mpp "$ROSIE_TARGET_MPP" \
      --tile_size "$ROSIE_TILE_SIZE" \
      --max_tiles_per_slide "$ROSIE_MAX_TILES"
  else
    echo "[1/7] Preparing ROSIE PNG inputs from WSI"
    python scripts/prepare_rosie_inputs_from_wsi.py \
      --reference_csv "$REF_FILE" \
      --wsi_dir "$WSI_DIR" \
      --output_dir "$ROSIE_INPUT_DIR" \
      --wsi_level "$ROSIE_WSI_LEVEL" \
      --target_downsample "$ROSIE_TARGET_DOWNSAMPLE" \
      --max_side "$ROSIE_MAX_SIDE"
  fi
fi

if [ "$SKIP_ROSIE_CONVERSION" = "1" ]; then
  echo "[2/7] Skipping ROSIE conversion (SKIP_ROSIE_CONVERSION=1)"
else
  echo "[2/7] Running ROSIE conversion (H&E -> multiplex TIFF)"
  if [ "$TILE_NATIVE" = "1" ]; then
    ROSIE_CONVERT_INPUT_DIR="$ROSIE_TILE_INPUT_DIR"
  else
    ROSIE_CONVERT_INPUT_DIR="$ROSIE_INPUT_DIR"
  fi
  python scripts/run_rosie_conversion.py \
    --rosie_dir "$ROSIE_DIR" \
    --input_dir "$ROSIE_CONVERT_INPUT_DIR" \
    --output_dir "$ROSIE_TIFF_DIR" \
    --model_path "$ROSIE_MODEL"
fi

if [ "$SKIP_ROSIE_RGB_CONVERSION" = "1" ]; then
  echo "[3/7] Skipping RGB conversion (SKIP_ROSIE_RGB_CONVERSION=1)"
else
  echo "[3/7] Converting ROSIE TIFF outputs to RGB composites"
  if [ "$TILE_NATIVE" = "1" ]; then
    ROSIE_RGB_OUTPUT_DIR="$ROSIE_TILE_RGB_DIR"
  else
    ROSIE_RGB_OUTPUT_DIR="$ROSIE_RGB_DIR"
  fi
  python scripts/convert_rosie_tiff_to_rgb.py \
    --input_dir "$ROSIE_TIFF_DIR" \
    --output_dir "$ROSIE_RGB_OUTPUT_DIR"

  if [ "$TILE_NATIVE" = "1" ]; then
    echo "[3b/7] Aggregating converted ROSIE tiles to slide mosaics"
    python scripts/aggregate_rosie_tiles_to_slide_rgb.py \
      --input_dir "$ROSIE_TILE_RGB_DIR" \
      --output_dir "$ROSIE_SLIDE_RGB_DIR" \
      --max_tiles_per_slide "$ROSIE_MAX_TILES" \
      --tile_size "$ROSIE_TILE_SIZE"
  fi
fi

echo "[4/7] Building ROSIE IF reference CSV"
if [ "$TILE_NATIVE" = "1" ]; then
  ROSIE_REF_IMAGE_DIR="$ROSIE_SLIDE_RGB_DIR"
else
  ROSIE_REF_IMAGE_DIR="$ROSIE_RGB_DIR"
fi
python if2rna_scripts/create_rosie_if_reference.py \
  --source_reference "$REF_FILE" \
  --rosie_image_dir "$ROSIE_REF_IMAGE_DIR" \
  --output_reference "$ROSIE_REF_FILE" \
  --organ_type BRCA

echo "[5/7] IF patch extraction + feature extraction + kmeans"
python if2rna_scripts/run_if_preprocessing.py \
  --ref_file "$ROSIE_REF_FILE" \
  --image_dir "$ROSIE_REF_IMAGE_DIR" \
  --patch_dir "$IF_PATCH_DIR" \
  --feature_dir "$IF_FEATURE_DIR" \
  --feat_type resnet \
  --no_duplicate_patches \
  --model_dir "$RESNET_DIR"

echo "[6/7] IF2RNA inference on ROSIE-converted inputs"
python if2rna_scripts/run_if2rna_inference.py \
  --reference_csv "$ROSIE_REF_FILE" \
  --feature_dir "$IF_FEATURE_DIR" \
  --checkpoint "$IF2RNA_CKPT" \
  --gene_list_pkl "$IF2RNA_GENE_PKL" \
  --output_predictions "$PRED_OUT" \
  --output_correlations "$CORR_OUT"

echo "[7/7] Done"
echo "Predictions: $PRED_OUT"
echo "Correlations: $CORR_OUT"
