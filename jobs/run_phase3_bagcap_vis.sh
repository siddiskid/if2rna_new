#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p3_bag_vis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase3_bagcap_vis_%j.out
#SBATCH --error=logs/phase3_bagcap_vis_%j.err

set -euo pipefail

if [ -z "${BAG_CAP:-}" ]; then
  echo "ERROR: BAG_CAP env var is required"
  exit 1
fi

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/train_if2rna.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name "phase3_vis_bagcap_${BAG_CAP}" \
  --model_type vis \
  --feature_key cluster_features \
  --bag_cap "$BAG_CAP" \
  --log_transform \
  --train \
  --num_epochs 200 \
  --batch_size 16 \
  --seed 42
