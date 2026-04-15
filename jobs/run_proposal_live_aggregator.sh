#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_agg_live
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/proposal_live_aggregator_%j.out
#SBATCH --error=logs/proposal_live_aggregator_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
mkdir -p logs

python scripts/aggregate_proposal_results_live.py
