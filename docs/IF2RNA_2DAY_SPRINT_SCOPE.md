# IF2RNA 2-Day Sprint Scope

## Goal
Deliver a reproducible, submission-ready MVP that shows IF2RNA performance on ROI-level gene prediction with clear baselines, limited ablations, and a concise report package.

## Time Box
- Day 1: Data + benchmark core
- Day 2: Ablations + packaging

## Phase 1 Part 2 Freeze (Locked)

### Primary dataset for this sprint
- Reference file: data/hne_data_archive_20260403_153424/metadata/if_reference.csv
- Samples: 942
- Patients: 17
- Organs: 4 (Liver=379, Kidney=203, Lymph_Node=187, Colon=173)
- Gene targets available: 15,830 rna_* columns

### Split protocol for this sprint
- Default split: patient-level 5-fold cross-validation
- Implementation source: patient_kfold in if2rna_scripts/train_if2rna.py
- Seed: 42 (fixed)
- Validation policy: validation subset from train patients per fold (as implemented)

### Primary run configuration (for comparability)
- Feature input: cluster_features from data/if_features
- Model family: IF2RNA (ViS default)
- Train mode: 5-fold CV with fixed seed
- Reporting metrics: mean/median gene-wise Pearson r, MAE, RMSE, #genes above threshold

### Notes
- KIRC 20-sample references are considered auxiliary and excluded from the primary benchmark.
- Any deviation from this freeze must be recorded as a separate experiment track.

## Must Ship

### 1) Scope and experiment freeze
- Freeze one primary dataset configuration and one split protocol.
- Freeze one primary IF2RNA model configuration (feature type, bag cap, seed, epochs).
- Freeze one metric set for all comparisons: mean/median gene-wise Pearson r, MAE, RMSE, #genes above threshold.

### 2) Data and preprocessing verification
- Run data validation and save logs.
- Run preprocessing validation and save logs.
- Produce a dataset summary table (samples, organs/studies, genes, split counts).

### 3) Core benchmark results
- Train/evaluate one main IF2RNA run with fixed config.
- Train/evaluate at least one simple baseline (mean pooling or handcrafted + elastic net).
- Produce one comparison table: baseline vs IF2RNA.

### 4) Minimal ablations
- Run exactly two ablations:
  - pooling: attention vs mean
  - bag cap: 128 vs 256
- Produce one compact ablation table with deltas.

### 5) Final artifacts for submission
- One reproducibility command list (commands used + outputs generated).
- Final figures:
  - benchmark summary figure
  - gene-correlation distribution figure
  - one qualitative expression map figure (if available)
- Final results bundle with CSVs and plots in a single output folder.

## Nice To Have
- Second baseline (HE2RNA-style pooling adaptation).
- OOD held-out-study run with one delta table.
- ROI mask vs bounding-box ablation.
- Additional qualitative virtual maps for 2-3 genes.
- Organ-level breakdown table and figure.

## Not In Scope (for this 2-day sprint)
- Full multi-organ generalization matrix.
- Full paper-quality model sweep across many seeds/hyperparameters.
- Large-scale virtual mapping on all slides.

## Definition of Done
- A reviewer can rerun the key commands from a clean environment and regenerate:
  - main benchmark table
  - ablation table
  - key figures
- All reported numbers in slides/report are traceable to saved CSV outputs.

## Risks and fallback decisions
- If runtime is too high: reduce to one baseline + two ablations only.
- If OOD cannot finish: document as deferred and keep in future work.
- If virtual map generation is unstable: ship one robust qualitative map only.

## Working rules for the sprint
- Keep one run = one output folder.
- Do not start a new experiment until the previous one has a metrics CSV.
- Log command, config, and runtime for every run.
- Prefer fewer complete experiments over many partial runs.
