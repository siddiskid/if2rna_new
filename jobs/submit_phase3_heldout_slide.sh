#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p3_slide_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --output=logs/phase3_heldout_slide_submit_%j.out
#SBATCH --error=logs/phase3_heldout_slide_submit_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
mkdir -p logs

echo "Submitting held-out-slide meanpool"
out_mean=$(sbatch jobs/run_phase3_heldout_slide_meanpool.sh)
echo "$out_mean"

echo "Submitting held-out-slide ViS"
out_vis=$(sbatch jobs/run_phase3_heldout_slide_vis.sh)
echo "$out_vis"

echo "Submitting held-out-slide ViT"
out_vit=$(sbatch jobs/run_phase3_heldout_slide_vit.sh)
echo "$out_vit"

echo "Track with: squeue -u $USER"
