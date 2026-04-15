#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p3_slide_mean
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase3_heldout_slide_meanpool_%j.out
#SBATCH --error=logs/phase3_heldout_slide_meanpool_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/run_heldout_slide_meanpool.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --output_dir results/phase3_ood/heldout_slide_meanpool \
  --alpha 1.0 \
  --seed 42 \
  --log_transform \
  --min_test_samples 10
