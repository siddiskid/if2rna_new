#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_vmaps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/virtual_maps_amil_%j.out
#SBATCH --error=logs/virtual_maps_amil_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python if2rna_scripts/generate_virtual_expression_maps.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --image_root data/if_images \
  --checkpoint results/if2rna_models/phase2_attention_mil/model_best.pt \
  --model_type attention_mil \
  --resnet_weights models/resnet50/resnet50_imagenet.pth \
  --output_dir results/final_package/virtual_maps/amil_dense_maps_20260411_fine \
  --max_slides 4 \
  --max_rois_per_slide 12 \
  --patch_size 224 \
  --stride 112 \
  --batch_size 128 \
  --top_k_genes 8 \
  --smooth_sigma 4
