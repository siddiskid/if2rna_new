#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_he2_hvg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/he2rna_style_hvg_%j.out
#SBATCH --error=logs/he2rna_style_hvg_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/train_if2rna_he2rna_style.py \
  --ref_file results/final_package/targets/if_reference_hvg2000_plus_markers.csv \
  --feature_dir data/if_features \
  --feature_key cluster_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_he2rna_style_hvg2000_20260410 \
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
