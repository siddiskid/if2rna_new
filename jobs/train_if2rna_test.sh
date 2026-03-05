#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Activate virtual environment
source .venv/bin/activate

# Change to project directory
cd /scratch/st-singha53-1/schiluku/if2rna_new

# Create logs directory if needed
mkdir -p logs

# Run quick test with 10% data
python if2rna_scripts/train_if2rna.py \
    --ref_file data/metadata/if_reference.csv \
    --feature_dir data/if_features \
    --save_dir results/if2rna_models \
    --exp_name test_10pct_log \
    --sample_percent 0.1 \
    --log_transform \
    --train \
    --num_epochs 10

echo "Job finished at: $(date)"
