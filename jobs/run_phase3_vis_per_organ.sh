#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_vis_org
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase3_vis_org_%j.out
#SBATCH --error=logs/phase3_vis_org_%j.err

set -euo pipefail

if [ -z "${ORGAN:-}" ]; then
  echo "ERROR: ORGAN environment variable is required"
  exit 1
fi

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

REF_FILE="data/hne_data_archive_20260403_153424/metadata/organ_refs/if_reference_${ORGAN}.csv"
EXP_NAME="phase3_vis_${ORGAN}"

if [ ! -f "$REF_FILE" ]; then
  echo "ERROR: Missing organ reference: $REF_FILE"
  exit 1
fi

python if2rna_scripts/train_if2rna.py \
  --ref_file "$REF_FILE" \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name "$EXP_NAME" \
  --model_type vis \
  --k 4 \
  --log_transform \
  --train \
  --num_epochs 200 \
  --batch_size 16 \
  --seed 42
