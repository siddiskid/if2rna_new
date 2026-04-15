# IF2RNA Comprehensive Methods, Models, Findings, and Execution Log

Document version: 1.0  
Date: 2026-04-06  
Repository: if2rna_new (main)

## 1. Purpose and Scope

This document is an end-to-end technical record of what has been built, run, validated, and observed in the IF2RNA project up to 2026-04-06. It is intentionally exhaustive and includes:

- Data pipeline design and implementation details
- Model architecture details and training configuration
- Evaluation protocols for in-distribution and OOD settings
- Experiment status including completed, failed, timed-out, and in-progress runs
- Quantitative findings from both historical and latest experiments
- Known failure modes, root causes, and remediation plan
- Reproducibility paths, outputs, and command templates

## 2. High-Level Project Objective

Primary goal:
Predict large-panel RNA expression vectors from immunofluorescence-derived image features at ROI/slide level.

Secondary goals:

- Compare multiple model families under the same IF feature pipeline
- Stress test generalization under out-of-distribution settings
- Produce publication/presentation-ready artifacts with verified numbers

## 3. Repository and Pipeline Components

Top-level project artifacts relevant to this report:

- if2rna_scripts/extract_if_images.py
- if2rna_scripts/create_if_reference_csv.py
- if2rna_scripts/preprocess_if_patches.py
- if2rna_scripts/preprocess_if_features.py
- if2rna_scripts/preprocess_if_kmeans.py
- if2rna_scripts/train_if2rna.py
- if2rna_scripts/train_if2rna_elastic_net.py
- if2rna_scripts/run_heldout_slide_meanpool.py
- if2rna_scripts/run_heldout_slide_neural.py
- scripts/build_presentation_assets.py
- results/phase3_summary/*
- results/presentation_assets/2026-04-06/*

Historical summary anchors:

- VERIFIED_RESULTS_SUMMARY.md
- PRESENTATION_SLIDES_CONTENT.md

## 4. Data Pipeline: Detailed Methodology

### 4.1 Phase 1: IF image extraction

Script: if2rna_scripts/extract_if_images.py

Process summary:

- Read NanoString archive structures
- Parse ROI image outputs from workflow exports
- Normalize naming and directory placement
- Save composite images by organ and sample ID

Output pattern:

- data/if_images/{organ}/{sample_id}/composite.png

### 4.2 Phase 2: Metadata and RNA linkage

Script: if2rna_scripts/create_if_reference_csv.py

Process summary:

- Join image metadata with RNA normalization outputs
- Build unified reference table containing sample identity, organ, patient, and gene targets
- Use RNA columns prefixed with rna_

Output:

- data/metadata/if_reference.csv

### 4.3 Phase 3: Patch generation

Script: if2rna_scripts/preprocess_if_patches.py

Process summary:

- Tile IF composites into patches
- Standardize patch extraction logic and storage
- Preserve per-sample patch bags in H5

Output pattern:

- data/if_patches/{organ}/{sample_id}/{sample_id}.h5

### 4.4 Phase 4: Feature extraction

Script: if2rna_scripts/preprocess_if_features.py

Process summary:

- Load patch images and feed forward through ResNet50 feature extractor
- Persist per-patch embeddings (resnet_features)
- Use fixed feature dimensionality

### 4.5 Phase 5: K-means compression for fixed-length bag representation

Script: if2rna_scripts/preprocess_if_kmeans.py

Process summary:

- Cluster patch embeddings into K=100 centroids
- Store cluster_features for model input
- Provide fixed-size bag representation for transformer-style encoders

## 5. Training and Evaluation Framework

Primary script: if2rna_scripts/train_if2rna.py

### 5.1 Data loading and cleaning logic

The training script includes explicit safeguards:

- Detect RNA columns by prefix rna_
- Remove genes with more than 50 percent missing values
- Remove samples with more than 90 percent missing gene values
- Replace remaining NaN and inf values with zero
- Optional log1p transform

### 5.2 Split strategy

- Patient-level K-fold split (default K=5)
- Train, validation, and test are disjoint by patient ID
- Validation split is derived from train patients

This avoids patient leakage and keeps cross-fold integrity.

### 5.3 Dataset and feature options

Implemented in IFRNADataset in train_if2rna.py:

- Feature source key can be selected:
  - cluster_features
  - resnet_features
- Optional bag_cap subsamples patch/cluster bags if bag length exceeds cap

### 5.4 Model families in active code

Model choice via model_type:

- vis
- vit
- attention_mil

#### A) ViS

- Imported from sequoia-pub/src/tformer_lin.py
- Transformer-style architecture adapted for this setting
- Uses depth and number of heads as configurable hyperparameters

#### B) ViT baseline

- Imported from sequoia-pub/src/vit.py
- Uses dim, depth, heads, mlp_dim and dim_head configuration

#### C) Attention MIL

Custom class added in train_if2rna.py and reused in run_heldout_slide_neural.py.

Core structure:

1. Encoder block:
- LayerNorm(input)
- Linear(input_dim to hidden_dim)
- GELU
- Dropout

2. Gated attention pooling:
- V branch with tanh projection
- U branch with sigmoid projection
- Elementwise product
- Linear to scalar attention logits per instance
- Softmax over bag instances

3. Bag aggregation:
- Weighted sum of encoded instances

4. Decoder:
- LayerNorm(hidden)
- Linear(hidden_dim to num_outputs)

Intuition:
Attention MIL learns which patch-level or cluster-level instances should dominate prediction for each sample.

### 5.5 Optimization defaults observed

From train_if2rna.py and related runners:

- Optimizer: AdamW
- Typical learning rate: 1e-3 in current scripts (earlier baselines may use different values)
- Batch size: often 16 in neural scripts
- Epochs vary by script and phase (for OOD slide neural runner default 60)
- Deterministic seed initialization enabled

## 6. OOD Evaluation Protocols

Two OOD axes are used.

### 6.1 Held-out organ OOD

- Aggregate metrics available in results/phase3_summary/ood_aggregate_model_comparison.csv
- Evaluates transfer to unseen organ partitions

### 6.2 Held-out slide OOD

Scripts:

- if2rna_scripts/run_heldout_slide_meanpool.py
- if2rna_scripts/run_heldout_slide_neural.py

Protocol per held-out slide S:

1. Train on all samples with slide_name not equal to S
2. Test on samples with slide_name equal to S
3. Compute per-gene correlations and error metrics
4. Aggregate across valid held-out slides

Meanpool script specifics:

- Feature = mean(cluster_features) per sample
- Model = Ridge regression
- Handles missing features by skip

Neural held-out slide script specifics:

- Supports vis, vit, attention_mil
- Builds patient-split train and validation from training pool
- Trains per held-out slide, then evaluates on held-out slide

## 7. Experiment Inventory and Status

Current run status table source:
results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv

Status as of 2026-04-06:

1. attention_mil: completed (Strong ID result)
2. bag_cap_100: completed (Ablation baseline-equivalent cap)
3. bag_cap_64: failed (ViS patch-count mismatch)
4. heldout_slide_vit: completed (OOD slide summary available)
5. heldout_slide_vis: timeout (20h time limit reached)
6. elastic_net: oom (Memory exceeded)

## 8. Quantitative Findings

## 8.1 Latest ID benchmark with Attention MIL

Source: results/presentation_assets/2026-04-06/table_id_main_with_amil.csv

| Model | n_samples | n_genes | Mean r | Median r | MAE | RMSE | Genes r > 0.3 | Genes r > 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_mil | 941 | 11981 | 0.7441 | 0.7879 | 0.4435 | 0.6210 | 11814 | 11084 |
| vis_main | 941 | 11981 | 0.6793 | 0.6973 | 0.4710 | 0.7413 | 11834 | 11057 |
| vit_baseline | 941 | 11981 | 0.6473 | 0.6633 | 0.5376 | 0.7620 | 11833 | 10785 |
| meanpool_ridge_baseline | 941 | 11981 | 0.6213 | 0.6471 | 0.5730 | 0.7624 | 11347 | 9149 |

Interpretation:

- Attention MIL is the current best model in this benchmark snapshot by mean and median gene-wise correlation.
- Attention MIL also improves MAE and RMSE versus ViS and ViT in this table.
- ViS and ViT remain competitive but behind Attention MIL in aggregate correlation.

### 8.2 OOD aggregate (organ + slide)

Source: results/presentation_assets/2026-04-06/table_ood_aggregate.csv

Held-out organ rows:

- meanpool_ridge_baseline: mean-of-means 0.00978, MAE 1.1391, RMSE 1.4295
- vis: mean-of-means 0.00037, MAE 1.1622, RMSE 1.4749
- vit: mean-of-means 0.00683, MAE 1.1140, RMSE 1.4051

Held-out slide rows:

- meanpool_ridge_baseline: mean-of-means 0.02528, MAE 0.5113, RMSE 0.6669
- vit: mean-of-means -0.00025, MAE 0.4625, RMSE 0.6304
- vis: in progress values for all aggregate metric fields

Interpretation:

- OOD remains difficult: correlation values are close to zero in multiple settings.
- On held-out organ aggregate, vit has better MAE/RMSE than vis in this snapshot.
- Held-out slide is partially complete due vis timeout; table intentionally shows in progress in metric cells where unavailable.

### 8.3 Per-organ specialization: ViS vs ViT

Source: results/phase3_summary/phase3_per_organ_model_comparison.csv

Mean correlation by organ:

- Colon: ViS 0.0288, ViT 0.0950 (ViT higher)
- Kidney: ViS -0.0241, ViT -0.0475 (ViS higher)
- Liver: ViS 0.0656, ViT 0.0408 (ViS higher)
- Lymph node: ViS -0.1357, ViT -0.0647 (ViT higher)

Interpretation:

- No single neural model dominates every organ.
- There is strong evidence of organ-dependent specialization behavior.

### 8.4 Historical verified baseline summary (publication-ready checkpoint)

Source: VERIFIED_RESULTS_SUMMARY.md

Reported historical IF2RNA baseline (baseline_resnet_log):

- Test samples: 941
- Genes: 11981
- Median r: 0.766
- Mean r: 0.735
- MAE: 0.431
- RMSE: 0.639
- Genes r > 0.5: 11288 (94.2 percent)

Reported historical comparison to SEQUOIA H and E baseline:

- IF2RNA median and mean correlations substantially higher than reported H and E baseline numbers in that file
- Biological rationale offered: IF captures protein-linked molecular signal beyond morphology

Important note:
The repository now includes additional/newer model variants and OOD analyses that extend beyond the historical snapshot.

## 9. Detailed Failure Analysis and Root Causes

### 9.1 heldout_slide_vis timeout

Observed status:

- Timed out at job wall-time limit (20h)

Impact:

- Missing held-out-slide aggregate metrics for vis in current OOD table

Current handling:

- Downstream tables and presentation assets mark unavailable metrics as in progress

### 9.2 elastic-net OOM

Script:

- if2rna_scripts/train_if2rna_elastic_net.py

Likely root driver:

- MultiTaskElasticNetCV over high-dimensional handcrafted vectors plus high-output RNA target dimensionality can exceed memory budget

Impact:

- No valid final summary for elastic-net baseline run

### 9.3 bag_cap_64 failure (ViS mismatch)

Status description:

- ViS patch-count or token-count mismatch when using altered bag size/cap

Impact:

- bag-cap ablation incomplete for this setting

## 10. Reporting and Visualization Pipeline

Script:

- scripts/build_presentation_assets.py

What it does:

- Builds report-ready CSV and PNG tables
- Generates multiple publication/presentation figures
- Includes ID comparisons, OOD summaries, per-organ charts, slide-level distributions, and run-status visuals
- Handles missing metrics by writing in progress in-cell for OOD table

Output root:

- results/presentation_assets/2026-04-06

Notable outputs:

- table_id_main_with_amil.csv and .png
- table_ood_aggregate.csv and .png
- table_per_organ_specialization.csv and .png
- table_run_status_and_blockers.csv and .png
- fig_01 through fig_18 PNG figures

Recent readability and consistency improvements implemented:

- Consistent numeric labels on bars
- Improved font scale and spacing
- Legend placement above plots for overlap avoidance
- Annotation headroom to avoid clipping
- Layout rebalancing to preserve plot area size

## 11. Reproducibility: Key Command Templates

Neural training/evaluation (example pattern):

python if2rna_scripts/train_if2rna.py \
  --ref_file data/metadata/if_reference.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_attention_mil \
  --model_type attention_mil \
  --train

Held-out-slide meanpool:

python if2rna_scripts/run_heldout_slide_meanpool.py \
  --ref_file data/metadata/if_reference.csv \
  --feature_dir data/if_features \
  --output_dir results/phase3_ood/heldout_slide_meanpool

Held-out-slide neural:

python if2rna_scripts/run_heldout_slide_neural.py \
  --ref_file data/metadata/if_reference.csv \
  --feature_dir data/if_features \
  --output_dir results/phase3_ood/heldout_slide_vit \
  --model_type vit

Presentation assets regeneration:

python scripts/build_presentation_assets.py

## 12. What Has Been Completed vs Pending

Completed:

- Added and trained Attention MIL path
- Added held-out-slide evaluation scripts (meanpool and neural)
- Added bag-cap controls to training path
- Added elastic-net baseline script
- Generated extensive polished presentation asset pack
- Produced integrated summary tables and chart set for reporting

Partially complete or blocked:

- Held-out-slide vis aggregate is incomplete due timeout
- Elastic-net run failed OOM
- bag_cap_64 failed and requires model/input compatibility fix

## 13. Project-Level Conclusions (Current State)

1. The IF2RNA pipeline is operational end-to-end from IF images to RNA prediction outputs.
2. Attention MIL currently provides the strongest ID aggregate performance in the latest comparison table.
3. OOD generalization remains challenging and is highly shift-dependent.
4. Per-organ results indicate model specialization effects that may justify organ-aware selection or ensembling.
5. Operationally, key bottlenecks are runtime limits and memory pressure in some baseline/ablation runs.

## 14. Recommended Next Technical Actions

1. Resume or rerun heldout_slide_vis with adjusted runtime/resources or reduced per-slide training cost.
2. Refactor elastic-net baseline for memory efficiency:
- lower-dimensional handcrafted vector
- chunked or gene-block fitting
- fewer CV alphas or folds
3. Resolve bag_cap_64 shape compatibility for ViS path and rerun ablation.
4. Add confidence intervals across folds or held-out units for major aggregate metrics.
5. Finalize a locked benchmark table after all in-progress OOD rows are complete.

## 15. File Reference Index

Primary summaries:

- VERIFIED_RESULTS_SUMMARY.md
- PRESENTATION_SLIDES_CONTENT.md

Core code:

- if2rna_scripts/train_if2rna.py
- if2rna_scripts/train_if2rna_elastic_net.py
- if2rna_scripts/run_heldout_slide_meanpool.py
- if2rna_scripts/run_heldout_slide_neural.py
- scripts/build_presentation_assets.py

Primary numeric outputs used in this document:

- results/phase3_summary/phase2_model_comparison_snapshot.csv
- results/phase3_summary/ood_aggregate_model_comparison.csv
- results/phase3_summary/phase3_per_organ_model_comparison.csv
- results/presentation_assets/2026-04-06/table_id_main_with_amil.csv
- results/presentation_assets/2026-04-06/table_ood_aggregate.csv
- results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv

---

End of report.
