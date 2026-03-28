#!/usr/bin/env python3
"""
Quick job submission helper for SEQUOIA pipeline

Usage:
    # Interactive mode
    python scripts/submit_sequoia_job.py
    
    # Command line mode
    python scripts/submit_sequoia_job.py --cancer BRCA --fold 0
    
    # Multiple folds
    python scripts/submit_sequoia_job.py --cancer BRCA --folds 0 1 2 3 4
    
    # Multiple cancer types
    python scripts/submit_sequoia_job.py --cancer BRCA COAD KIRC --fold 0
    
    # Dry run (generate scripts without submitting)
    python scripts/submit_sequoia_job.py --cancer BRCA --fold 0 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Available cancer types in SEQUOIA
AVAILABLE_CANCERS = {
    'BLCA': 'Bladder Urothelial Carcinoma',
    'BRCA': 'Breast Invasive Carcinoma',
    'COAD': 'Colon Adenocarcinoma',
    'GBMLGG': 'Glioblastoma + Low Grade Glioma',
    'HNSC': 'Head and Neck Squamous Cell Carcinoma',
    'KIRC': 'Kidney Renal Clear Cell Carcinoma',
    'KIRP': 'Kidney Renal Papillary Cell Carcinoma',
    'LIHC': 'Liver Hepatocellular Carcinoma',
    'LUAD': 'Lung Adenocarcinoma',
    'LUSC': 'Lung Squamous Cell Carcinoma',
    'OV': 'Ovarian Serous Cystadenocarcinoma',
    'PRAD': 'Prostate Adenocarcinoma',
    'READ': 'Rectum Adenocarcinoma',
    'STAD': 'Stomach Adenocarcinoma',
    'UCEC': 'Uterine Corpus Endometrial Carcinoma'
}

def create_custom_job_script(cancer: str, fold: int, output_path: Path, 
                             account: str = "st-singha53-1-gpu",
                             time: str = "24:00:00",
                             mem: str = "64G") -> Path:
    """Create a custom job script for specific cancer type and fold"""
    
    job_name = f"sequoia_{cancer.lower()}_fold{fold}"
    output_file = output_path / f"run_sequoia_{cancer.lower()}_fold{fold}.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

set -e

echo "========================================================================"
echo "SEQUOIA Pipeline: {AVAILABLE_CANCERS.get(cancer, cancer)} (Fold {fold})"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

nvidia-smi

source .venv/bin/activate
cd /scratch/st-singha53-1/schiluku/if2rna_new

mkdir -p logs data/hne_data/processed/{{patches,features,masks}} models/sequoia results/sequoia

# Configuration
export CANCER_TYPE="{cancer}"
export FOLD={fold}
export REF_FILE="data/hne_data/metadata/tcga_reference.csv"
export GENE_LIST="sequoia-pub/examples/gene_list.csv"
export WSI_PATH="data/hne_data/raw/images"
export FEAT_TYPE="uni"

echo "Configuration:"
echo "  Cancer: $CANCER_TYPE ({AVAILABLE_CANCERS.get(cancer, cancer)})"
echo "  Fold: $FOLD"
echo "  Reference: $REF_FILE"
echo "  Feature type: $FEAT_TYPE"
echo ""

# Step 1: Preprocessing
if [ ! -f "logs/.checkpoint_preprocess_${{CANCER_TYPE}}_done" ]; then
    echo "[1/4] Preprocessing..."
    export HF_HUB_OFFLINE=1
    python scripts/preprocess_slides.py \\
        --ref_file $REF_FILE \\
        --wsi_path $WSI_PATH \\
        --output_dir data/hne_data/processed \
        --feat_type $FEAT_TYPE \\
        --steps all \\
        --max_patches 4000
    touch "logs/.checkpoint_preprocess_${{CANCER_TYPE}}_done"
    echo "✓ Preprocessing complete"
else
    echo "[1/4] Preprocessing already done"
fi

# Step 2: Download model
if [ ! -f "logs/.checkpoint_model_${{CANCER_TYPE}}_${{FOLD}}_done" ]; then
    echo "[2/4] Downloading model..."
    unset HF_HUB_OFFLINE
    python scripts/download_sequoia_model.py \\
        --cancer_type $CANCER_TYPE \\
        --fold $FOLD \\
        --output_dir models/sequoia
    touch "logs/.checkpoint_model_${{CANCER_TYPE}}_${{FOLD}}_done"
    echo "✓ Model downloaded"
else
    echo "[2/4] Model already downloaded"
fi

# Step 3: Inference
if [ ! -f "logs/.checkpoint_inference_${{CANCER_TYPE}}_${{FOLD}}_done" ]; then
    echo "[3/4] Running inference..."
    export HF_HUB_OFFLINE=1
    python scripts/run_sequoia_inference.py \\
        --model_dir "models/sequoia/${{CANCER_TYPE,,}}-$FOLD" \\
        --ref_file $REF_FILE \\
        --feature_dir data/hne_data/processed/features \
        --gene_list $GENE_LIST \\
        --output_dir results/sequoia \\
        --feat_type $FEAT_TYPE \\
        --fold $FOLD \\
        --cancer_type $CANCER_TYPE
    touch "logs/.checkpoint_inference_${{CANCER_TYPE}}_${{FOLD}}_done"
    echo "✓ Inference complete"
else
    echo "[3/4] Inference already done"
fi

# Step 4: Evaluation
echo "[4/4] Evaluating..."
python scripts/evaluate_predictions.py \\
    --predictions_file "results/sequoia/predictions_${{CANCER_TYPE,,}}-$FOLD.csv" \\
    --gene_list $GENE_LIST \\
    --reference_file $REF_FILE \\
    --output_dir results/sequoia

echo ""
echo "========================================================================"
echo "✓ Pipeline Complete!"
echo "========================================================================"
echo "Results: results/sequoia/predictions_${{CANCER_TYPE,,}}-${{FOLD}}_correlations.csv"
echo "Finished: $(date)"
"""
    
    output_file.write_text(script_content)
    output_file.chmod(0o755)
    
    return output_file

def interactive_mode():
    """Interactive job configuration"""
    print("\n" + "="*70)
    print("SEQUOIA Pipeline Job Submission")
    print("="*70)
    
    print("\nAvailable cancer types:")
    for i, (code, name) in enumerate(sorted(AVAILABLE_CANCERS.items()), 1):
        print(f"  {i:2d}. {code:8s} - {name}")
    
    cancer_input = input("\nSelect cancer type (number or code, e.g., '2' or 'BRCA'): ").strip()
    
    if cancer_input.isdigit():
        idx = int(cancer_input) - 1
        cancer = list(sorted(AVAILABLE_CANCERS.keys()))[idx]
    else:
        cancer = cancer_input.upper()
    
    if cancer not in AVAILABLE_CANCERS:
        print(f"✗ Invalid cancer type: {cancer}")
        sys.exit(1)
    
    print(f"\nSelected: {cancer} - {AVAILABLE_CANCERS[cancer]}")
    
    fold_input = input("\nEnter fold (0-4 or 'all'): ").strip()
    
    if fold_input.lower() == 'all':
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [int(f) for f in fold_input.split()]
    
    print(f"\nFolds: {folds}")
    
    submit = input("\nSubmit jobs now? (y/n): ").strip().lower()
    
    return [cancer], folds, submit == 'y'

def main():
    parser = argparse.ArgumentParser(
        description="Submit SEQUOIA pipeline jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/submit_sequoia_job.py
  
  # Single job
  python scripts/submit_sequoia_job.py --cancer BRCA --fold 0
  
  # All folds for one cancer
  python scripts/submit_sequoia_job.py --cancer BRCA --folds 0 1 2 3 4
  
  # Multiple cancers
  python scripts/submit_sequoia_job.py --cancer BRCA COAD KIRC --fold 0
  
  # Generate scripts without submitting
  python scripts/submit_sequoia_job.py --cancer BRCA --fold 0 --dry-run
        """
    )
    
    parser.add_argument('--cancer', nargs='+', choices=list(AVAILABLE_CANCERS.keys()),
                       help='Cancer type(s)')
    parser.add_argument('--fold', type=int, help='Single fold (0-4)')
    parser.add_argument('--folds', nargs='+', type=int, help='Multiple folds')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts without submitting')
    parser.add_argument('--output-dir', default='jobs/generated',
                       help='Output directory for generated scripts')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or (not args.cancer and not args.fold and not args.folds):
        cancers, folds, submit = interactive_mode()
    else:
        if not args.cancer:
            parser.error("--cancer is required")
        
        cancers = args.cancer
        
        if args.fold is not None:
            folds = [args.fold]
        elif args.folds is not None:
            folds = args.folds
        else:
            parser.error("Either --fold or --folds is required")
        
        submit = not args.dry_run
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job scripts
    print("\n" + "="*70)
    print("Generating job scripts...")
    print("="*70)
    
    scripts = []
    for cancer in cancers:
        for fold in folds:
            script = create_custom_job_script(cancer, fold, output_dir)
            scripts.append(script)
            print(f"  ✓ {script.name}")
    
    print(f"\nGenerated {len(scripts)} job script(s) in {output_dir}/")
    
    # Submit jobs
    if submit:
        print("\n" + "="*70)
        print("Submitting jobs...")
        print("="*70)
        
        job_ids = []
        for script in scripts:
            try:
                result = subprocess.run(
                    ['sbatch', str(script)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"  ✓ Submitted {script.name} (Job ID: {job_id})")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to submit {script.name}: {e}")
        
        print(f"\n✓ Submitted {len(job_ids)} job(s)")
        print("\nMonitor jobs:")
        print(f"  squeue -u $USER")
        print(f"  tail -f logs/sequoia_*.out")
    else:
        print("\nTo submit manually:")
        for script in scripts:
            print(f"  sbatch {script}")

if __name__ == "__main__":
    main()
