#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_roi_cons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/roi_consistency_amil_%j.out
#SBATCH --error=logs/roi_consistency_amil_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/evaluate_roi_consistency_from_tiles.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --feature_key resnet_features \
  --checkpoint results/if2rna_models/phase2_attention_mil/model_best.pt \
  --model_type attention_mil \
  --output_dir results/final_package/virtual_maps/roi_consistency_amil_20260410 \
  --log_transform
