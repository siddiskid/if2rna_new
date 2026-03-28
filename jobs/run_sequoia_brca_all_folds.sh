#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=sequoia_brca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=logs/sequoia_brca_f%a_%j.out
#SBATCH --error=logs/sequoia_brca_f%a_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/st-singha53-1/schiluku/if2rna_new}"
cd "$PROJECT_DIR"

# Defaults target your BRCA subset run; override as needed at submission time.
export CANCER_TYPE="${CANCER_TYPE:-BRCA}"
export FOLD="${SLURM_ARRAY_TASK_ID}"
export REF_FILE="${REF_FILE:-data/hne_data/metadata/tcga_reference_brca_88.csv}"
export WSI_PATH="${WSI_PATH:-data/hne_data/raw/images}"
export GENE_LIST="${GENE_LIST:-models/gene_list.csv}"
export FEAT_TYPE="${FEAT_TYPE:-uni}"
export PROCESSED_ROOT="${PROCESSED_ROOT:-data/hne_data/processed}"
export FEATURE_DIR="${FEATURE_DIR:-data/hne_data/processed/features}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/sequoia}"
export MODEL_ROOT="${MODEL_ROOT:-models/sequoia}"
export STRICT_SAMPLE_MATCH="${STRICT_SAMPLE_MATCH:-1}"
export PREPROCESS_WAIT_MINUTES="${PREPROCESS_WAIT_MINUTES:-360}"

# Avoid races: only one fold should preprocess shared H&E slides/features.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
	export SKIP_PREPROCESS=0
else
	export SKIP_PREPROCESS=1
fi

# Backward compatibility: remap legacy metadata path if it is injected via env.
if [ "$REF_FILE" = "data/metadata/tcga_reference_brca_88.csv" ] && [ -f "data/hne_data/metadata/tcga_reference_brca_88.csv" ]; then
	echo "Remapping legacy REF_FILE to data/hne_data/metadata/tcga_reference_brca_88.csv"
	export REF_FILE="data/hne_data/metadata/tcga_reference_brca_88.csv"
fi

echo "Running fold $FOLD for $CANCER_TYPE"
echo "Reference: $REF_FILE"
echo "SKIP_PREPROCESS: $SKIP_PREPROCESS"

bash jobs/run_sequoia_end_to_end.sh
