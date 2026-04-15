#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p2_enet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/phase2_enet_%j.out
#SBATCH --error=logs/phase2_enet_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/train_if2rna_elastic_net.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --feature_key cluster_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_elastic_net_baseline \
  --k 5 \
  --seed 42 \
  --log_transform
