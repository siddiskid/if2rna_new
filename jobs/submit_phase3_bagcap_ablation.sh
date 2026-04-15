#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p3_bag_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --output=logs/phase3_bagcap_submit_%j.out
#SBATCH --error=logs/phase3_bagcap_submit_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
mkdir -p logs

# Current cached feature bags are size 100, so run practical ablation 64 vs 100.
for cap in 64 100; do
  echo "Submitting bag-cap ablation with BAG_CAP=$cap"
  out=$(sbatch --export=ALL,BAG_CAP=$cap jobs/run_phase3_bagcap_vis.sh)
  echo "$out"
done

echo "Track jobs with: squeue -u $USER"
