#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=tcga_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/tcga_download_%j.out
#SBATCH --error=logs/tcga_download_%j.err

################################################################################
# TCGA Download Job
#
# Downloads paired TCGA H&E slides + RNA files using scripts/download_tcga_data.py
#
# Edit the CONFIG section below, then run:
#   sbatch jobs/download_tcga_data.sh
################################################################################

set -euo pipefail

echo "========================================================================"
echo "TCGA Download Job"
echo "========================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================================================"

# Setup
PROJECT_DIR="/scratch/st-singha53-1/schiluku/if2rna_new"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"

cd "$PROJECT_DIR"
source "$VENV_PATH"

echo "Working directory: $PWD"
echo "Python: $(which python)"

# CONFIG
# CANCER_TYPES can include multiple values, e.g.: "BRCA LUAD KIRC"
CANCER_TYPES="BRCA"
NUM_SAMPLES=100
OUTPUT_DIR="data/raw"

# Validation
if [ ! -f "scripts/download_tcga_data.py" ]; then
  echo "ERROR: Missing script scripts/download_tcga_data.py"
  exit 1
fi

if [ ! -d "logs" ]; then
  echo "ERROR: logs directory not found"
  exit 1
fi

echo ""
echo "Configuration:"
echo "  CANCER_TYPES: $CANCER_TYPES"
echo "  NUM_SAMPLES:  $NUM_SAMPLES"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"
echo ""

python scripts/download_tcga_data.py \
  --cancer_types $CANCER_TYPES \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Download complete"
echo "========================================================================"
echo "Slides:   $OUTPUT_DIR/images"
echo "RNA:      $OUTPUT_DIR/rna"
echo "Metadata: $(dirname "$OUTPUT_DIR")/metadata"
echo "Finished at: $(date)"
echo "========================================================================"
