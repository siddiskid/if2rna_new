# SEQUOIA End-to-End Pipeline - Pre-Run Checklist

## ⚠️ CRITICAL ISSUES FOUND IN SCRIPT

### 1. **MISSING: Reference CSV File** ❌
**Location:** `data/metadata/tcga_reference.csv`
**Required by:** Step 1 (Preprocessing)

**Fix on Sockeye (BEFORE running job):**
```bash
# Check if file exists
ls -lh data/metadata/tcga_reference.csv

# If missing, create it:
python scripts/create_reference_csv.py \
    --tcga_dir data/raw/tcga_slides \
    --rna_dir data/raw/rna \
    --output_file data/metadata/tcga_reference.csv
```

---

### 2. **MISSING: Gene List File** ⚠️
**Location:** `sequoia-pub/examples/gene_list.csv`
**Required by:** Step 3 (Inference) and Step 4 (Evaluation)

**Check on Sockeye:**
```bash
ls -lh sequoia-pub/examples/gene_list.csv
```

If missing, this file should be in the sequoia-pub submodule. Make sure submodule is initialized:
```bash
git submodule update --init --recursive
```

---

### 3. **MISSING: UNI Model** ❌
**Location:** `models/uni/`
**Required by:** Step 1 (Feature extraction with `feat_type=uni`)

**Fix on Sockeye LOGIN NODE (has internet):**
```bash
# Download UNI model (must be on login node with internet)
python scripts/download_uni_model.py --output_dir models/uni

# Transfer to compute node if needed
# (Script handles offline mode via HF_HUB_OFFLINE=1)
```

---

### 4. **MISSING: TCGA Slide Files** ❌
**Location:** `data/raw/tcga_slides/`
**Required by:** Step 1 (Patch extraction)

**Check on Sockeye:**
```bash
ls data/raw/tcga_slides/*.svs | wc -l
# Should show at least a few slide files
```

If missing, slides need to be downloaded first using:
```bash
python scripts/download_tcga_data.py --cancer_types BRCA --output_dir data/raw
```

---

### 5. **MISSING: Virtual Environment** ⚠️
**Location:** `.venv/bin/activate`
**Required by:** Script line 47

**Check on Sockeye:**
```bash
ls -lh .venv/bin/activate
```

If missing, create environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 6. **POTENTIAL: Out of Memory** ⚠️
**Script allocates:** 64GB RAM
**Potential issue:** Large slides + UNI model can exceed this

**Monitor during run:**
```bash
# View memory usage in real-time
squeue -j JOBID --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %C %m"
```

**If OOM occurs, increase memory:**
```bash
# In script, change:
#SBATCH --mem=64G   →   #SBATCH --mem=128G
```

---

### 7. **POTENTIAL: Disk Space Issues** ⚠️
**Preprocessing outputs can be large:**
- Patches: ~10-50 GB per dataset
- Features: ~5-20 GB per dataset
- Clusters: ~1-5 GB per dataset

**Check disk space on Sockeye:**
```bash
df -h /scratch/st-singha53-1/
quota -s
```

---

### 8. **ARGUMENT ISSUES (ALREADY FIXED)** ✅
- ~~`--max_patches` ambiguity~~ → Fixed to use `--max_patches_per_slide` and `--max_patches_for_features`
- ~~`--steps all` invalid~~ → Fixed to use `patches features kmeans`

---

## 🔍 COMPLETE PRE-RUN CHECKLIST

Run these commands **ON SOCKEYE** before submitting the job:

```bash
# Navigate to project
cd /scratch/st-singha53-1/schiluku/if2rna_new

# 1. Check reference file
echo "1. Checking reference file..."
if [ -f "data/metadata/tcga_reference.csv" ]; then
    echo "   ✓ Reference file exists"
    head -2 data/metadata/tcga_reference.csv
else
    echo "   ✗ MISSING: data/metadata/tcga_reference.csv"
    echo "   Run: python scripts/create_reference_csv.py"
fi

# 2. Check gene list
echo "2. Checking gene list..."
if [ -f "sequoia-pub/examples/gene_list.csv" ]; then
    echo "   ✓ Gene list exists"
    wc -l sequoia-pub/examples/gene_list.csv
else
    echo "   ✗ MISSING: sequoia-pub/examples/gene_list.csv"
    echo "   Run: git submodule update --init --recursive"
fi

# 3. Check UNI model
echo "3. Checking UNI model..."
if [ -d "models/uni" ]; then
    echo "   ✓ UNI model directory exists"
    ls -lh models/uni/
else
    echo "   ✗ MISSING: models/uni/"
    echo "   Run on LOGIN NODE: python scripts/download_uni_model.py"
fi

# 4. Check TCGA slides
echo "4. Checking TCGA slides..."
SLIDE_COUNT=$(ls data/raw/tcga_slides/*.svs 2>/dev/null | wc -l)
if [ "$SLIDE_COUNT" -gt 0 ]; then
    echo "   ✓ Found $SLIDE_COUNT slide files"
else
    echo "   ✗ No slides found in data/raw/tcga_slides/"
    echo "   Need to download TCGA data first"
fi

# 5. Check virtual environment
echo "5. Checking virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    echo "   ✓ Virtual environment exists"
    source .venv/bin/activate
    python --version
else
    echo "   ✗ MISSING: .venv/"
    echo "   Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# 6. Check disk space
echo "6. Checking disk space..."
df -h /scratch/st-singha53-1/ | head -2

# 7. Check if already on correct branch
echo "7. Checking git branch..."
git branch | grep '*'

echo ""
echo "========================================="
echo "Pre-flight check complete!"
echo "========================================="
```

---

## 📋 RECOMMENDED RUN SEQUENCE

### **Option A: First Time Setup (Need to download everything)**
```bash
# 1. Download TCGA data (login node, has internet)
python scripts/download_tcga_data.py --cancer_types BRCA --output_dir data/raw --max_slides 50

# 2. Download UNI model (login node, has internet)
python scripts/download_uni_model.py --output_dir models/uni

# 3. Create reference CSV (compute node OK)
python scripts/create_reference_csv.py \
    --tcga_dir data/raw/tcga_slides \
    --rna_dir data/raw/rna \
    --output_file data/metadata/tcga_reference.csv

# 4. Submit job
sbatch jobs/run_sequoia_end_to_end.sh
```

### **Option B: Data Already Exists (Just need reference)**
```bash
# 1. Create reference CSV if missing
python scripts/create_reference_csv.py \
    --tcga_dir data/raw/tcga_slides \
    --rna_dir data/raw/rna \
    --output_file data/metadata/tcga_reference.csv

# 2. Submit job
sbatch jobs/run_sequoia_end_to_end.sh
```

### **Option C: Everything Ready**
```bash
# Just submit
sbatch jobs/run_sequoia_end_to_end.sh
```

---

## 🚨 WHAT TO DO IF JOB FAILS

### Check Error Logs
```bash
# View job output
tail -100 logs/sequoia_e2e_JOBID.out

# View errors
tail -100 logs/sequoia_e2e_JOBID.err
```

### Common Errors & Solutions

**Error:** `Reference file not found`
```bash
python scripts/create_reference_csv.py --tcga_dir data/raw/tcga_slides --output_file data/metadata/tcga_reference.csv
```

**Error:** `No module named 'huggingface_hub'`
```bash
source .venv/bin/activate
pip install huggingface-hub
```

**Error:** `CUDA out of memory`
```bash
# Reduce batch size in script or increase GPU memory allocation
```

**Error:** `Permission denied` for git operations
```bash
git config --global --add safe.directory /scratch/st-singha53-1/schiluku/if2rna_new
```

---

## ⏱️ EXPECTED RUNTIME

- **Step 1 (Preprocessing):** 2-4 hours (depends on number of slides)
  - Patch extraction: ~30-60 min
  - UNI features: ~1-2 hours (GPU)
  - K-means: ~15-30 min
  
- **Step 2 (Model download):** 5-10 minutes

- **Step 3 (Inference):** 20-40 minutes

- **Step 4 (Evaluation):** 2-5 minutes

**Total:** ~3-5 hours for full pipeline

---

## 💾 EXPECTED OUTPUT FILES

After successful completion:
```
data/processed/
├── patches/              # HDF5 files with image patches
├── features/             # HDF5 files with UNI features
└── masks/                # (optional) tissue masks

models/sequoia/
└── brca-0/              # Downloaded SEQUOIA model
    ├── model.pt
    └── config.json

results/sequoia/
├── predictions_brca-0.csv                  # Raw predictions
├── predictions_brca-0_fixed.csv           # Column-aligned predictions
└── predictions_brca-0_correlations.csv    # Per-gene Pearson correlations
```

---

## 🎯 SUCCESS CRITERIA

Job succeeded if you see:
```
Pipeline Complete!
Results saved to:
  - results/sequoia/predictions_brca-0_fixed.csv
  - results/sequoia/predictions_brca-0_correlations.csv
```

Check correlations:
```bash
head -20 results/sequoia/predictions_brca-0_correlations.csv
```

**Expected:** Many genes with r > 0.7-0.9 (SEQUOIA paper shows strong performance on BRCA)
