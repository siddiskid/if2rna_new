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
# SEQUOIA End-to-End Pipeline (Bulletproof Version)
#
# Calls SEQUOIA scripts DIRECTLY (no wrapper) with verification after every
# step. Fixes known issues inline:
#   1. Removes .svs from reference CSV (SEQUOIA's patch script adds it back)
#   2. Inline k-means reads 'uni_features' (SEQUOIA hardcodes 'resnet_features')
#   3. Verifies output at every step - fails immediately if something is wrong
#
# Usage:
#   sbatch jobs/run_sequoia_end_to_end.sh
################################################################################

set -euo pipefail

echo "========================================================================"
echo "SEQUOIA End-to-End Pipeline"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================================================"

nvidia-smi

########################################
# Setup
########################################

echo ""
echo "[Setup] Loading modules..."
module load gcc/9.4.0
module load openslide/3.4.1

echo "[Setup] Activating virtual environment..."
cd /scratch/st-singha53-1/schiluku/if2rna_new
source .venv/bin/activate
echo "Working directory: $PWD"
echo "Python: $(which python)"

# SEQUOIA imports need src.* from sequoia-pub/
export PYTHONPATH="$(pwd)/sequoia-pub:${PYTHONPATH:-}"
# UNI model already cached - no internet needed on compute node
export HF_HUB_OFFLINE=1

# Config
REF_FILE="data/metadata/tcga_reference.csv"
WSI_PATH="data/hne_data/raw/images"
PATCHES_DIR="data/processed/patches"
FEATURES_DIR="data/processed/features"
MASKS_DIR="data/processed/masks"
GENE_LIST="sequoia-pub/examples/gene_list.csv"
MODEL_DIR="models/sequoia/brca-0"
OUTPUT_DIR="results/sequoia"
FEAT_TYPE="uni"
FOLD=0

echo ""
echo "Configuration:"
echo "  REF_FILE:     $REF_FILE"
echo "  WSI_PATH:     $WSI_PATH"
echo "  FEAT_TYPE:    $FEAT_TYPE"
echo "  MODEL_DIR:    $MODEL_DIR"

########################################
# Step 0: Verify all inputs exist
########################################

echo ""
echo "========================================================================"
echo "[Step 0/7] Verifying inputs"
echo "========================================================================"

if [ ! -f "$REF_FILE" ]; then echo "FATAL: Reference file not found: $REF_FILE"; exit 1; fi
echo "  ✓ Reference CSV exists"

SLIDE_COUNT=$(ls "$WSI_PATH"/*.svs 2>/dev/null | wc -l | tr -d ' ')
echo "  ✓ Slides found: $SLIDE_COUNT"
if [ "$SLIDE_COUNT" -eq 0 ]; then echo "FATAL: No .svs files in $WSI_PATH"; exit 1; fi

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then echo "FATAL: Model not found: $MODEL_DIR/model.safetensors"; exit 1; fi
echo "  ✓ Model found: $MODEL_DIR"

if [ ! -f "$GENE_LIST" ]; then echo "FATAL: Gene list not found: $GENE_LIST"; exit 1; fi
echo "  ✓ Gene list found"

if [ ! -f "sequoia-pub/pre_processing/patch_gen_hdf5.py" ]; then echo "FATAL: SEQUOIA scripts not found"; exit 1; fi
echo "  ✓ SEQUOIA scripts found"

echo "  ✓ All inputs verified"

########################################
# Step 1: Fix reference CSV
########################################

echo ""
echo "========================================================================"
echo "[Step 1/7] Fixing reference CSV"
echo "========================================================================"
echo "  SEQUOIA's patch_gen_hdf5.py adds .svs to wsi_file_name automatically."
echo "  If the CSV already has .svs, we get *.svs.svs and 0 matches."

python3 << 'FIX_REF_EOF'
import pandas as pd

ref = pd.read_csv("data/metadata/tcga_reference.csv")
before = ref['wsi_file_name'].iloc[0]

ref['wsi_file_name'] = ref['wsi_file_name'].str.replace(r'\.svs$', '', regex=True)

after = ref['wsi_file_name'].iloc[0]
ref.to_csv("data/metadata/tcga_reference.csv", index=False)

changed = "CHANGED" if before != after else "already correct"
print(f"  Before: {before}")
print(f"  After:  {after} ({changed})")
print(f"  Rows: {len(ref)}")
print(f"  ✓ Reference CSV ready")
FIX_REF_EOF

########################################
# Step 2: Clean old processed data
########################################

echo ""
echo "========================================================================"
echo "[Step 2/7] Cleaning old processed data"
echo "========================================================================"

rm -rf "$PATCHES_DIR" "$FEATURES_DIR" "$MASKS_DIR"
mkdir -p "$PATCHES_DIR" "$FEATURES_DIR" "$MASKS_DIR" "$OUTPUT_DIR" logs

echo "  ✓ Cleaned and recreated output directories"

########################################
# Step 3: Patch extraction
########################################

echo ""
echo "========================================================================"
echo "[Step 3/7] Patch extraction"
echo "========================================================================"

python sequoia-pub/pre_processing/patch_gen_hdf5.py \
    --ref_file "$REF_FILE" \
    --wsi_path "$WSI_PATH" \
    --patch_path "$PATCHES_DIR" \
    --mask_path "$MASKS_DIR" \
    --patch_size 256 \
    --max_patches_per_slide 2000 \
    --parallel 0

# VERIFY
echo ""
echo "--- Patch Verification ---"
PATCH_COUNT=$(find "$PATCHES_DIR" -name "*.hdf5" 2>/dev/null | wc -l | tr -d ' ')
echo "  HDF5 patch files: $PATCH_COUNT"

if [ "$PATCH_COUNT" -eq 0 ]; then
    echo ""
    echo "FATAL: No patches extracted!"
    echo ""
    echo "DEBUG — Reference CSV wsi_file_name:"
    python3 -c "import pandas as pd; [print(f'  {x}') for x in pd.read_csv('$REF_FILE')['wsi_file_name'].tolist()]"
    echo ""
    echo "DEBUG — Actual .svs files:"
    ls "$WSI_PATH"/*.svs | head -5
    echo ""
    echo "DEBUG — What patch_gen_hdf5.py tried to match:"
    python3 -c "
import pandas as pd
ref = pd.read_csv('$REF_FILE')
names = ref['wsi_file_name'].tolist()
expected = [f'{s}.svs' for s in names]
print('  Expected (name + .svs):')
for e in expected[:5]: print(f'    {e}')
import os
actual = sorted([f for f in os.listdir('$WSI_PATH') if f.endswith('.svs')])
print('  Actual files:')
for a in actual[:5]: print(f'    {a}')
matches = set(expected) & set(actual)
print(f'  Intersection: {len(matches)} matches')
"
    exit 1
fi

for d in "$PATCHES_DIR"/*/; do
    [ -d "$d" ] && echo "  ✓ $(basename "$d")"
done
echo "  ✓ Patch extraction complete ($PATCH_COUNT slides)"

########################################
# Step 4: UNI feature extraction
########################################

echo ""
echo "========================================================================"
echo "[Step 4/7] UNI feature extraction"
echo "========================================================================"

python sequoia-pub/pre_processing/compute_features_hdf5.py \
    --feat_type "$FEAT_TYPE" \
    --ref_file "$REF_FILE" \
    --patch_data_path "$PATCHES_DIR" \
    --feature_path "$FEATURES_DIR" \
    --max_patch_number 4000

# VERIFY
echo ""
echo "--- Feature Verification ---"
FEAT_COUNT=$(find "$FEATURES_DIR" -name "*.h5" 2>/dev/null | wc -l | tr -d ' ')
echo "  H5 feature files: $FEAT_COUNT"

if [ "$FEAT_COUNT" -eq 0 ]; then
    echo "FATAL: No features extracted!"
    echo "DEBUG — Feature dir contents:"
    find "$FEATURES_DIR" -type f 2>/dev/null | head -20
    echo "DEBUG — Patch dir contents:"
    find "$PATCHES_DIR" -type f 2>/dev/null | head -20
    exit 1
fi

python3 << 'FEAT_CHECK_EOF'
import h5py
from pathlib import Path
for h5 in sorted(Path("data/processed/features").rglob("*.h5")):
    with h5py.File(h5, 'r') as f:
        keys = list(f.keys())
        shapes = {k: f[k].shape for k in keys}
        print(f"  ✓ {h5.parent.name}: keys={keys}, shapes={shapes}")
FEAT_CHECK_EOF

echo "  ✓ Feature extraction complete ($FEAT_COUNT slides)"

########################################
# Step 5: K-means clustering
########################################

echo ""
echo "========================================================================"
echo "[Step 5/7] K-means clustering (inline — reads uni_features correctly)"
echo "========================================================================"
echo "  NOTE: SEQUOIA's kmean_features.py hardcodes 'resnet_features' key."
echo "  Since we used UNI, features are saved as 'uni_features'."
echo "  This inline script handles both correctly."

python3 << 'KMEANS_EOF'
import h5py
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd

ref_file = "data/metadata/tcga_reference.csv"
feature_dir = Path("data/processed/features")
n_clusters = 100
success = 0
total = 0

df = pd.read_csv(ref_file)

for _, row in df.iterrows():
    total += 1
    wsi = row['wsi_file_name'].replace('.svs', '')
    project = row['tcga_project']

    h5_path = feature_dir / project / wsi / f"{wsi}.h5"

    if not h5_path.exists():
        print(f"  ✗ Missing: {h5_path}")
        continue

    with h5py.File(h5_path, 'r+') as f:
        if 'cluster_features' in f:
            print(f"  ⊘ {wsi}: already done")
            success += 1
            continue

        # Find the feature key (uni_features or resnet_features)
        feat_key = None
        for key in ['uni_features', 'resnet_features']:
            if key in f:
                feat_key = key
                break

        if feat_key is None:
            print(f"  ✗ {wsi}: no feature key found (keys={list(f.keys())})")
            continue

        features = f[feat_key][:]
        n_use = min(n_clusters, len(features))

        kmeans = KMeans(n_clusters=n_use, random_state=99, n_init=10)
        kmeans.fit(features)

        # cluster_centers_ shape: (n_clusters, feature_dim) e.g. (100, 1024)
        f.create_dataset('cluster_features', data=kmeans.cluster_centers_)
        success += 1
        print(f"  ✓ {wsi}: {feat_key} {features.shape} → cluster_features {kmeans.cluster_centers_.shape}")

print(f"\n  K-means: {success}/{total} slides")
if success == 0:
    print("FATAL: K-means produced 0 results!")
    exit(1)
KMEANS_EOF

echo "  ✓ K-means clustering complete"

########################################
# Step 6: SEQUOIA Inference
########################################

echo ""
echo "========================================================================"
echo "[Step 6/7] SEQUOIA Inference (model: brca-$FOLD)"
echo "========================================================================"

python scripts/run_sequoia_inference.py \
    --model_dir "$MODEL_DIR" \
    --ref_file "$REF_FILE" \
    --feature_dir "$FEATURES_DIR" \
    --gene_list "$GENE_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --feat_type "$FEAT_TYPE"

# VERIFY
echo ""
echo "--- Inference Verification ---"
PRED_FILE="$OUTPUT_DIR/predictions_brca-${FOLD}.csv"
if [ ! -f "$PRED_FILE" ]; then
    echo "FATAL: Predictions file not created: $PRED_FILE"
    echo "DEBUG — Output dir contents:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null
    exit 1
fi

PRED_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$PRED_FILE')))")
PRED_COLS=$(python3 -c "import pandas as pd; print(len([c for c in pd.read_csv('$PRED_FILE').columns if c.startswith('pred_')]))")
echo "  File: $PRED_FILE"
echo "  Slides: $PRED_ROWS"
echo "  Genes: $PRED_COLS"

if [ "$PRED_ROWS" -eq 0 ]; then
    echo "FATAL: Predictions file has 0 rows!"
    exit 1
fi
echo "  ✓ Inference complete"

########################################
# Step 7: Evaluation
########################################

echo ""
echo "========================================================================"
echo "[Step 7/7] Evaluation"
echo "========================================================================"

python scripts/evaluate_predictions.py \
    --predictions_dir "$OUTPUT_DIR" \
    --gene_list "$GENE_LIST" \
    --reference "$REF_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --folds $FOLD

# Show results
echo ""
CORR_FILE="$OUTPUT_DIR/predictions_brca-${FOLD}_correlations.csv"
if [ -f "$CORR_FILE" ]; then
    echo "--- TOP 20 GENES BY CORRELATION ---"
    head -21 "$CORR_FILE"
    echo ""
    python3 << SUMMARY_EOF
import pandas as pd
df = pd.read_csv("$CORR_FILE")
print("--- SUMMARY ---")
print(f"  Total genes evaluated: {len(df)}")
print(f"  Mean correlation:      {df['correlation'].mean():.4f}")
print(f"  Median correlation:    {df['correlation'].median():.4f}")
print(f"  Max correlation:       {df['correlation'].max():.4f}")
print(f"  Genes with r > 0.5:   {(df['correlation'] > 0.5).sum()}")
print(f"  Genes with r > 0.8:   {(df['correlation'] > 0.8).sum()}")
SUMMARY_EOF
else
    echo "  ⚠ Correlation file not found (but predictions were generated)"
fi

########################################
# Done
########################################

echo ""
echo "========================================================================"
echo "✓ PIPELINE COMPLETE"
echo "========================================================================"
echo "Finished at: $(date)"
echo ""
echo "Output files:"
echo "  Patches:      $PATCHES_DIR/"
echo "  Features:     $FEATURES_DIR/"
echo "  Predictions:  $PRED_FILE"
echo "  Correlations: $CORR_FILE"
echo "========================================================================"
