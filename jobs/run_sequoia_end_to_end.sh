#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=sequoia_e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/sequoia_e2e_%j.out
#SBATCH --error=logs/sequoia_e2e_%j.err

################################################################################
# SEQUOIA End-to-End Pipeline
# 
# This script runs the complete SEQUOIA pipeline:
# 1. Preprocessing: Patch extraction + UNI feature extraction + K-means
# 2. Model download: Get pretrained SEQUOIA model from HuggingFace
# 3. Inference: Run predictions on test data
# 4. Evaluation: Calculate correlations with ground truth RNA-seq
#
# Prerequisites (run on login node):
#   - python scripts/download_uni_model.py
#   - python scripts/create_reference_csv.py
#
# Usage:
#   sbatch jobs/run_sequoia_end_to_end.sh
################################################################################

set -e  # Exit on error

# Print job info
echo "========================================================================"
echo "SEQUOIA End-to-End Pipeline"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================================================"
echo ""

# GPU info
nvidia-smi

# Load required modules
echo ""
echo "[Setup] Loading required modules..."
module load gcc/9.4.0
module load openslide/3.4.1

# Export library paths explicitly
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(module show openslide/3.4.1 2>&1 | grep LIBRARY_PATH | cut -d'"' -f2)

# Verify OpenSlide is accessible
echo "Checking OpenSlide library..."
ldconfig -p | grep openslide || echo "OpenSlide not in ldconfig, but module loaded"

# Activate virtual environment
echo ""
echo "[Setup] Activating virtual environment..."
source .venv/bin/activate

# Change to project directory (adjust path as needed)
PROJECT_DIR="/scratch/st-singha53-1/schiluku/if2rna_new"
cd $PROJECT_DIR
echo "Working directory: $PWD"

# Create required directories
mkdir -p logs
mkdir -p data/processed/{patches,features,masks}
mkdir -p models/sequoia
mkdir -p results/sequoia

# Configuration
REF_FILE="data/metadata/tcga_reference.csv"
GENE_LIST="sequoia-pub/examples/gene_list.csv"
WSI_PATH="data/hne_data/raw/images"  # H&E slides location
FEAT_TYPE="uni"
CANCER_TYPE="BRCA"  # Change to your cancer type (BRCA, COAD, GBMLGG, etc.)
FOLD=0  # Start with fold 0

echo ""
echo "Configuration:"
echo "  Reference file: $REF_FILE"
echo "  WSI path: $WSI_PATH"
echo "  Feature type: $FEAT_TYPE"
echo "  Cancer type: $CANCER_TYPE"
echo "  Fold: $FOLD"
echo ""

################################################################################
# Step 0: Ensure UNI Model is Available
################################################################################

if [ "$FEAT_TYPE" = "uni" ]; then
    echo "========================================================================"
    echo "[Step 0/4] UNI Model Check"
    echo "========================================================================"
    echo ""
    
    if [ -d "models/uni" ] && [ -f "models/uni/pytorch_model.bin" ]; then
        echo "✓ UNI model already exists at models/uni"
    else
        echo "Downloading UNI model..."
        # Try to download to models/uni
        python scripts/download_uni_model.py --output_dir models/uni 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo "Direct download failed, trying alternative location..."
            # Download to /tmp and copy
            python scripts/download_uni_model.py --output_dir /tmp/uni_model
            if [ $? -eq 0 ]; then
                mkdir -p models/uni
                cp -r /tmp/uni_model/uni/* models/uni/ 2>/dev/null || cp -r /tmp/uni_model/* models/uni/
                rm -rf /tmp/uni_model
                echo "✓ UNI model downloaded and copied to models/uni"
            else
                echo "✗ UNI model download failed!"
                echo "Please download manually on login node:"
                echo "  python scripts/download_uni_model.py --output_dir models/uni"
                exit 1
            fi
        else
            echo "✓ UNI model downloaded successfully"
        fi
    fi
    echo ""
fi

################################################################################
# Step 1: Preprocessing (Patches + Features + K-means)
################################################################################

CHECKPOINT_PREPROCESS="logs/.checkpoint_preprocess_done"

if [ -f "$CHECKPOINT_PREPROCESS" ]; then
    echo "[Step 1] Preprocessing already complete (checkpoint found)"
    echo "  To rerun, delete: $CHECKPOINT_PREPROCESS"
else
    echo "========================================================================"
    echo "[Step 1/4] Preprocessing Pipeline"
    echo "========================================================================"
    echo ""
    
    # Enable offline mode for HF Hub (UNI already cached)
    export HF_HUB_OFFLINE=1
    
    echo "[1.1] Starting SEQUOIA preprocessing (patches + features + kmeans)..."
    python scripts/preprocess_slides.py \
        --ref_file $REF_FILE \
        --wsi_path $WSI_PATH \
        --output_dir data/processed \
        --feat_type $FEAT_TYPE \
        --steps patches features kmeans \
        --patch_size 256 \
        --max_patches_per_slide 2000 \
        --max_patches_for_features 4000 \
        --n_clusters 100
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Preprocessing complete!"
        touch $CHECKPOINT_PREPROCESS
        echo "  Checkpoint saved: $CHECKPOINT_PREPROCESS"
    else
        echo ""
        echo "✗ Preprocessing failed! Check logs above."
        exit 1
    fi
fi

echo ""

################################################################################
# Step 2: Verify SEQUOIA Model Exists
################################################################################

MODEL_DIR="models/sequoia/${CANCER_TYPE,,}-${FOLD}"

echo "========================================================================"
echo "[Step 2/4] SEQUOIA Model Verification"
echo "========================================================================"
echo ""

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.pt" ]; then
    echo "✓ SEQUOIA model found at: $MODEL_DIR"
    ls -lh "$MODEL_DIR"
elif [ -d "$MODEL_DIR" ]; then
    echo "✓ Model directory exists: $MODEL_DIR"
    echo "Contents:"
    ls -lh "$MODEL_DIR"
else
    echo "✗ Model not found at: $MODEL_DIR"
    echo "Available models:"
    ls -lh models/sequoia/
    echo ""
    echo "Error: Model ${CANCER_TYPE,,}-${FOLD} does not exist!"
    echo "Available models are listed above."
    exit 1
fi

echo ""

################################################################################
# Step 3: Run SEQUOIA Inference
################################################################################

CHECKPOINT_INFERENCE="logs/.checkpoint_inference_done"
PREDICTIONS_FILE="results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}.csv"

if [ -f "$CHECKPOINT_INFERENCE" ]; then
    echo "[Step 3] Inference already complete (checkpoint found)"
    echo "  Predictions: $PREDICTIONS_FILE"
else
    echo "========================================================================"
    echo "[Step 3/4] SEQUOIA Inference"
    echo "========================================================================"
    echo ""
    
    echo "[3.1] Running inference on test set..."
    echo "  Model: $MODEL_DIR"
    echo "  Features: data/processed/features"
    echo "  Output: $PREDICTIONS_FILE"
    echo ""
    
    python scripts/run_sequoia_inference.py \
        --model_dir $MODEL_DIR \
        --ref_file $REF_FILE \
        --feature_dir data/processed/features \
        --gene_list $GENE_LIST \
        --output_dir results/sequoia \
        --feat_type $FEAT_TYPE
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Inference complete!"
        touch $CHECKPOINT_INFERENCE
        echo "  Checkpoint saved: $CHECKPOINT_INFERENCE"
        echo "  Predictions saved: $PREDICTIONS_FILE"
    else
        echo ""
        echo "✗ Inference failed! Check logs above."
        exit 1
    fi
fi

echo ""

################################################################################
# Step 4: Evaluate Predictions
################################################################################

echo "========================================================================"
echo "[Step 4/4] Evaluation"
echo "========================================================================"
echo ""

echo "[4.1] Evaluating predictions against ground truth..."
echo "  Predictions: $PREDICTIONS_FILE"
echo "  Reference: $REF_FILE"
echo ""

python scripts/evaluate_predictions.py \
    --predictions_dir results/sequoia \
    --gene_list $GENE_LIST \
    --reference $REF_FILE \
    --output_dir results/sequoia \
    --folds $FOLD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation complete!"
    echo ""
    echo "Results saved to:"
    echo "  - results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_fixed.csv"
    echo "  - results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv"
else
    echo ""
    echo "✗ Evaluation failed! Check logs above."
    exit 1
fi

################################################################################
# Summary
################################################################################

echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Finished at: $(date)"
echo ""
echo "Output files:"
echo "  Patches: data/processed/patches/"
echo "  Features: data/processed/features/"
echo "  Model: $MODEL_DIR"
echo "  Predictions: results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_fixed.csv"
echo "  Correlations: results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv"
echo ""
echo "To view results:"
echo "  cat results/sequoia/predictions_${CANCER_TYPE,,}-${FOLD}_correlations.csv | head -20"
echo ""
echo "To clean checkpoints and rerun from scratch:"
echo "  rm logs/.checkpoint_*"
echo ""
echo "========================================================================"