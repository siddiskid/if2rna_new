#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p2_amil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/phase2_amil_%j.out
#SBATCH --error=logs/phase2_amil_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/train_if2rna.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_attention_mil \
  --model_type attention_mil \
  --mil_hidden_dim 512 \
  --mil_attn_dim 256 \
  --mil_dropout 0.1 \
  --log_transform \
  --train \
  --num_epochs 200 \
  --batch_size 16 \
  --seed 42
