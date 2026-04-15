#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_ood_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=logs/phase3_ood_submit_%j.out
#SBATCH --error=logs/phase3_ood_submit_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
mkdir -p logs

echo "Submitting held-out-organ ViS..."
vis_submit=$(sbatch jobs/run_phase3_heldout_organ_vis.sh)
vis_jobid=$(echo "$vis_submit" | awk '{print $4}')
echo "$vis_submit"

echo "Submitting held-out-organ ViT..."
vit_submit=$(sbatch jobs/run_phase3_heldout_organ_vit.sh)
vit_jobid=$(echo "$vit_submit" | awk '{print $4}')
echo "$vit_submit"

echo "Submitted in parallel"
echo "  ViS job id: $vis_jobid"
echo "  ViT job id: $vit_jobid"
echo "Track with: squeue -u $USER"
