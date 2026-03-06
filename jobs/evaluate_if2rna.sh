#!/bin/bash
#SBATCH --job-name=eval_if2rna
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Evaluation script for IF2RNA model
# Can be run interactively or submitted with sbatch

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source /arc/project/st-singha53-1/if2rna/.venv/bin/activate

# Run evaluation
python if2rna_scripts/evaluate_if2rna.py \
    --results_file results/if2rna_models/baseline_resnet_log/test_results.pkl \
    --reference_file data/metadata/if_reference.csv \
    --output_dir results/if2rna_models/baseline_resnet_log/evaluation

echo "Job finished at: $(date)"
