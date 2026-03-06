#!/bin/bash

# Direct evaluation script - run on HPC without SLURM
# Usage: bash run_evaluation.sh

echo "Starting IF2RNA evaluation..."
echo "Time: $(date)"

# Set environment variables
export MPLCONFIGDIR=/tmp/matplotlib_config_$$
export FONTCONFIG_PATH=/tmp
mkdir -p $MPLCONFIGDIR

# Activate virtual environment
source /arc/project/st-singha53-1/if2rna/.venv/bin/activate

# Run evaluation
python if2rna_scripts/evaluate_if2rna.py \
    --results_file results/if2rna_models/baseline_resnet_log/test_results.pkl \
    --reference_file data/metadata/if_reference.csv \
    --output_dir results/if2rna_models/baseline_resnet_log/evaluation

echo ""
echo "Evaluation complete!"
echo "Time: $(date)"
