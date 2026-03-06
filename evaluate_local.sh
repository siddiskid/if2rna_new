#!/bin/bash

# Local evaluation script - run directly without SLURM
# Usage: ./evaluate_local.sh

echo "Starting IF2RNA evaluation..."
echo "Running at: $(date)"

# Run evaluation
python if2rna_scripts/evaluate_if2rna.py \
    --results_file results/if2rna_models/baseline_resnet_log/test_results.pkl \
    --reference_file data/metadata/if_reference.csv \
    --output_dir results/if2rna_models/baseline_resnet_log/evaluation

echo "Evaluation complete at: $(date)"
echo ""
echo "Results saved to: results/if2rna_models/baseline_resnet_log/evaluation/"
echo "  - gene_correlations.csv: Per-gene Pearson correlations"
echo "  - organ_statistics.csv: Performance by organ type"
echo "  - summary_statistics.csv: Overall metrics"
echo "  - *.png: Visualization plots"
