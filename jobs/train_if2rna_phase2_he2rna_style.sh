#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p2_he2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase2_he2rna_style_%j.out
#SBATCH --error=logs/phase2_he2rna_style_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/train_if2rna_he2rna_style.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --feature_key cluster_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_he2rna_style_baseline_20260410 \
  --k 5 \
  --seed 42 \
  --log_transform \
  --num_epochs 120 \
  --patience 20 \
  --batch_size 16 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --dropout 0.2 \
  --ks 5 10 20
