#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p3_vit_ovr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --output=logs/phase3_overnight_submit_vit_%j.out
#SBATCH --error=logs/phase3_overnight_submit_vit_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

# Ensure organ-specific reference files exist.
python if2rna_scripts/create_organ_specific_refs.py

for organ in colon kidney liver lymph_node; do
  echo "Submitting organ-specific ViT for: $organ"
  submit_out=$(sbatch --export=ALL,ORGAN=$organ jobs/run_phase3_vit_per_organ.sh)
  echo "$submit_out"
done

echo "Overnight ViT queue submitted. Track with: squeue -u $USER"
