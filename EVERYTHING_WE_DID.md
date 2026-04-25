# IF2RNA / IF2RNA_New — exhaustive “everything we did” ledger

**Repo:** `siddiskid/if2rna_new` (branch `main`)  
**Snapshot date (this writeup):** 2026-04-15  
**Purpose of this file:** a single, excruciatingly detailed, source-linked record of *what exists in this repository*, *what experiments were run (as evidenced by artifacts)*, *how the pipelines work end-to-end*, and *where every major number/table/figure came from*.

This is intentionally redundant with other repo docs (notably the comprehensive execution log), but the goal here is: **one file you can hand to a reviewer (or your future self) and trace everything.**

> Important: When I say “we ran X”, I’m basing that on **persisted artifacts** under `results/` and/or explicit “execution logs” docs. If an output folder is empty or missing metrics files, I call that out.

---

## Table of contents

1. [What this repository is (two pipelines)](#what-this-repository-is-two-pipelines)
2. [Repository map (directories and what lives where)](#repository-map-directories-and-what-lives-where)
3. [Phase freeze + primary dataset (what was locked)](#phase-freeze--primary-dataset-what-was-locked)
4. [IF2RNA data pipeline (GeoMx IF)](#if2rna-data-pipeline-geomx-if)
   1. [IF image extraction](#if-image-extraction)
   2. [Reference CSV creation](#reference-csv-creation)
   3. [Patch extraction](#patch-extraction)
   4. [Feature extraction](#feature-extraction)
   5. [K-means bag compression](#k-means-bag-compression)
   6. [Validation scripts](#validation-scripts)
5. [IF2RNA modeling + training](#if2rna-modeling--training)
   1. [Common split protocol (patient-level CV)](#common-split-protocol-patient-level-cv)
   2. [Model families implemented](#model-families-implemented)
   3. [Model outputs and evaluation format](#model-outputs-and-evaluation-format)
6. [In-distribution results and baselines (what is actually benchmarked)](#in-distribution-results-and-baselines-what-is-actually-benchmarked)
7. [OOD generalization experiments (held-out slide, held-out organ)](#ood-generalization-experiments-held-out-slide-held-out-organ)
8. [Virtual expression maps + ROI-consistency checks](#virtual-expression-maps--roi-consistency-checks)
9. [Target-set work: HVG-on-train-only package](#target-set-work-hvg-on-train-only-package)
10. [SEQUOIA H&E pipeline (TCGA baseline + verification)](#sequoia-he-pipeline-tcga-baseline--verification)
11. [ROSIE pipeline integration (conversion + inference artifacts)](#rosie-pipeline-integration-conversion--inference-artifacts)
12. [Packaging and presentation assets (tables, figures, “final_package”)](#packaging-and-presentation-assets-tables-figures-final_package)
13. [Full inventory of scripts (IF2RNA, SEQUOIA utilities, jobs)](#full-inventory-of-scripts-if2rna-sequoia-utilities-jobs)
14. [Known mismatches, sharp edges, and failure modes](#known-mismatches-sharp-edges-and-failure-modes)
15. [Appendix: practical “how to rerun” command index](#appendix-practical-how-to-rerun-command-index)

---

## What this repository is (two pipelines)

This repo contains **two** major pipelines:

1. **IF2RNA (main):** predict ROI-level RNA expression from **NanoString GeoMx** immunofluorescence (IF) ROI images.
   - Canonical workflow described in [if2rna_scripts/README.md](if2rna_scripts/README.md).
   - Frozen dataset/config described in [docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml](docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml).

2. **SEQUOIA (secondary / reference baseline):** the original TCGA H&E “SEQUOIA” pipeline (WSI preprocessing → model download → inference → evaluation).
   - Canonical doc: [docs/SEQUOIA_PIPELINE.md](docs/SEQUOIA_PIPELINE.md).
   - Pre-run pitfalls documented in [PRE_RUN_CHECKLIST.md](PRE_RUN_CHECKLIST.md).

The repo is also **HPC/SLURM-first** (Sockeye). Most long-running experiments are captured as job scripts under [jobs/](jobs/).

---

## Repository map (directories and what lives where)

Top-level “human docs” and wrappers:

- [VERIFIED_RESULTS_SUMMARY.md](VERIFIED_RESULTS_SUMMARY.md): a pinned “numbers are checked” summary (notably the historical `baseline_resnet_log` run).
- [PRESENTATION_SLIDES_CONTENT.md](PRESENTATION_SLIDES_CONTENT.md): slide narrative; some sections are outdated vs the frozen dataset (see [Known mismatches](#known-mismatches-sharp-edges-and-failure-modes)).
- [PRE_RUN_CHECKLIST.md](PRE_RUN_CHECKLIST.md): SEQUOIA pipeline gotchas on Sockeye.
- [run_evaluation.sh](run_evaluation.sh) and [evaluate_local.sh](evaluate_local.sh): thin wrappers around IF2RNA evaluation.

Primary code:

- [if2rna_scripts/](if2rna_scripts/): IF2RNA pipeline (extract, preprocess, train, evaluate, OOD, virtual maps).
- [scripts/](scripts/): SEQUOIA pipeline utilities + ROSIE conversion utilities + aggregation/packaging scripts.
- [jobs/](jobs/): SLURM job scripts for phase2 baselines, phase3 OOD, ablations, and SEQUOIA runs.

Primary data roots:

- [data/](data/)
  - `if_data/`: raw-ish GeoMx workflow exports (zips + count Excel)
  - `if_images/`: extracted ROI PNGs
  - `if_patches/`: per-ROI HDF5 patch bags
  - `if_features/`: per-ROI HDF5 feature bags + clustering outputs
  - `hne_data_archive_20260403_153424/`: archived “frozen” snapshot that contains the reference CSVs actually used
  - `hne_data/`: a separate (non-archive) H&E-related workspace (not treated as the primary frozen baseline)
  - `rosie_if/`: data products used for ROSIE conversion/inference pipelines

Primary outputs:

- [results/](results/)
  - [results/if2rna_models/](results/if2rna_models/): model checkpoints and per-run outputs
  - [results/phase2_benchmark/](results/phase2_benchmark/): baseline comparison tables/plots
  - [results/phase3_ood/](results/phase3_ood/): held-out slide and held-out organ runs
  - [results/phase3_summary/](results/phase3_summary/): aggregated summaries over phase2/phase3
  - [results/presentation_assets/2026-04-06/](results/presentation_assets/2026-04-06/): *report-ready* figures/tables (PNG + CSV)
  - [results/final_package/](results/final_package/): proposal-style “final” tables + target packages + virtual maps
  - `results/predictions_brca-*.csv`: SEQUOIA outputs for BRCA folds (H&E)

---

## Phase freeze + primary dataset (what was locked)

The repo explicitly “freezes” a Phase 1 Part 2 baseline configuration in:

- [docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml](docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml)

Key frozen facts (verbatim from the YAML):

- **Primary dataset reference:** `data/hne_data_archive_20260403_153424/metadata/if_reference.csv`
- **Training reference:** `data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv`
- **Samples:** 942 total; **training_samples_ready:** 941
- **Patients:** 17
- **Organs:** Liver (379), Kidney (203), Lymph_Node (187), Colon (173)
- **Genes:** 15,830 `rna_*` columns (available in the raw reference); some evaluations use subsets after filtering
- **Split protocol:** patient-level 5-fold CV implemented by `patient_kfold()` in [if2rna_scripts/train_if2rna.py](if2rna_scripts/train_if2rna.py)
- **Primary model track:** model_type `vis` (ViS), input `cluster_features`, feature_dir `data/if_features`

Additional dataset summary CSVs exist:

- [docs/PHASE1_PART3_DATASET_SUMMARY.csv](docs/PHASE1_PART3_DATASET_SUMMARY.csv)
- [docs/PHASE1_PART3_SPLIT_SUMMARY.csv](docs/PHASE1_PART3_SPLIT_SUMMARY.csv)
- [docs/PHASE1_PART3_ORGAN_DISTRIBUTION.csv](docs/PHASE1_PART3_ORGAN_DISTRIBUTION.csv)

The “2-day sprint” scope doc (what was supposed to ship in that window) is:

- [docs/IF2RNA_2DAY_SPRINT_SCOPE.md](docs/IF2RNA_2DAY_SPRINT_SCOPE.md)

---

## IF2RNA data pipeline (GeoMx IF)

The canonical “how to run it” is in [if2rna_scripts/README.md](if2rna_scripts/README.md). Below is the full pipeline with details and the exact scripts.

### IF image extraction

- Script: [if2rna_scripts/extract_if_images.py](if2rna_scripts/extract_if_images.py)

**Inputs (by convention):**

- `data/if_data/{Organ}/workflow_and_count_files/workflow/roi_report/*.zip`

**What it does:**

- Iterates through `roi_report` zips.
- Extracts `.png` ROI images.
- Handles “segment” images (e.g. `PanCK+`, `PanCK-`) if `--extract_segments` is used.
- Writes a structured folder hierarchy under `data/if_images/`.

**Output layout (actual behavior):**

- `data/if_images/{Organ}/{slide_name}/{slide_name}_{scan_label}_ROI_{roi_num}[_{segment}].png`

### Reference CSV creation

- Script: [if2rna_scripts/create_if_reference_csv.py](if2rna_scripts/create_if_reference_csv.py)

**Inputs:**

- Images under `data/if_images/`
- NanoString expression Excel files under `data/if_data/{Organ}/workflow_and_count_files/count/`
  - normalized counts default to `Export4_NormalizationQ3.xlsx`

**What it does:**

- Reads segment metadata (`SegmentProperties`) and expression matrix (`TargetCountMatrix` or fallback).
- Matches each ROI/segment to an extracted image path.
- Emits a “SEQUOIA-like” reference table:
  - ID columns: `wsi_file_name`, `patient_id`, `organ_type`, `slide_name`, etc.
  - Targets: `rna_*` columns

**Important note (real repo state):**

- Multiple reference CSVs exist. The ones used by the frozen pipeline live under:
  - [data/hne_data_archive_20260403_153424/metadata/](data/hne_data_archive_20260403_153424/metadata)
  - notably: [data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv](data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv)

### Patch extraction

- Script: [if2rna_scripts/preprocess_if_patches.py](if2rna_scripts/preprocess_if_patches.py)

**Inputs:**

- A reference CSV row providing `wsi_file_name`, `organ_type`, `slide_name`, and optionally `image_path`.
- Images in `data/if_images/`.

**Patch method used here (as implemented):**

- Opens ROI PNG as RGB.
- Extracts patches on a grid (stride = `patch_size * (1 - overlap)`).
- Background rejection is **IF-specific**: it keeps patches unless they are “almost completely black” (`gray < 5`), since IF images are naturally dark.
- If there are too many patches, it randomly samples to `--max_patches`.
- If too few patches, it can **duplicate** patches to a fixed count unless `--no_duplicate_patches` is passed.

**Output:**

- `data/if_patches/{organ}/{wsi_file_name}/{wsi_file_name}.h5`
  - datasets `patch_0`, `patch_1`, ...

### Feature extraction

- Script: [if2rna_scripts/preprocess_if_features.py](if2rna_scripts/preprocess_if_features.py)

**Features supported:**

- `resnet` (ResNet50 ImageNet weights), baseline recommended
- `uni` (optional, requires UNI setup and caching)

**Offline/HPC constraint:**

- On compute nodes without internet, weights must be staged.
- ResNet download helper: [if2rna_scripts/download_resnet50.py](if2rna_scripts/download_resnet50.py)
- UNI offline procedure: [docs/UNI_SETUP_HPC.md](docs/UNI_SETUP_HPC.md)

**Output:**

- `data/if_features/{organ}/{wsi_file_name}/{wsi_file_name}.h5`
  - adds `resnet_features` or `uni_features`

### K-means bag compression

- Script: [if2rna_scripts/preprocess_if_kmeans.py](if2rna_scripts/preprocess_if_kmeans.py)

**Why it exists:**

- Patch bags can be variable length and large.
- Many models in this repo are built around a fixed-size “bag” representation.

**Method:**

- K-means over patch embeddings to produce `num_clusters` cluster centers.

**Output:**

- Adds `cluster_features` dataset to each H5 file, typically shape `100 × feature_dim`.

### Validation scripts

- IF input validation: [if2rna_scripts/validate_if_data.py](if2rna_scripts/validate_if_data.py)
- Preprocessing validation: [if2rna_scripts/validate_preprocessing.py](if2rna_scripts/validate_preprocessing.py)

Additionally, there are one-off fixers under [scripts/](scripts/) (e.g., [scripts/fix_kmeans.py](scripts/fix_kmeans.py), [scripts/fix_reference_csv.py](scripts/fix_reference_csv.py)).

---

## IF2RNA modeling + training

### Common split protocol (patient-level CV)

The canonical split logic is `patient_kfold()` in [if2rna_scripts/train_if2rna.py](if2rna_scripts/train_if2rna.py):

- Splits are done at the **patient** level (to avoid leakage).
- For each fold:
  - Train patients → split into train/val patients (via `train_test_split`).
  - Test patients are disjoint.

This protocol is referenced in [docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml](docs/EXPERIMENT_FREEZE_PHASE1_PART2.yaml).

### Model families implemented

Primary training entrypoint:

- [if2rna_scripts/train_if2rna.py](if2rna_scripts/train_if2rna.py)

It supports these `--model_type` values:

- `vis`: ViS (linearized transformer) imported from `sequoia-pub/src`.
- `vit`: transformer baseline imported from `sequoia-pub/src`.
- `attention_mil`: gated Attention-MIL implemented locally in the script.

Other model/baseline scripts (separate entrypoints):

- Mean pooling baseline: [if2rna_scripts/train_if2rna_mean_pool.py](if2rna_scripts/train_if2rna_mean_pool.py)
- Elastic net baseline: [if2rna_scripts/train_if2rna_elastic_net.py](if2rna_scripts/train_if2rna_elastic_net.py)
- HE2RNA-style pooling adaptation: [if2rna_scripts/train_if2rna_he2rna_style.py](if2rna_scripts/train_if2rna_he2rna_style.py)

### Model outputs and evaluation format

Neural training runs commonly write:

- `model_best*.pt` (checkpoints)
- `test_results.pkl`

The expected structure of `test_results.pkl` is probed by:

- [check_results_structure.py](check_results_structure.py)

Evaluation scripts consume `test_results.pkl`:

- [if2rna_scripts/evaluate_if2rna.py](if2rna_scripts/evaluate_if2rna.py)

Outputs commonly include:

- per-gene correlation CSV
- organ stats CSV
- summary metrics CSV
- plots (PNG)

---

## In-distribution results and baselines (what is actually benchmarked)

There are *two* layers of “results” in this repo:

1. A **historical verified run** with very strong performance (`baseline_resnet_log`) documented in [VERIFIED_RESULTS_SUMMARY.md](VERIFIED_RESULTS_SUMMARY.md).
2. A **phase2/phase3 benchmark suite** with multiple models and a standardized results pack (tables/figures under `results/presentation_assets/2026-04-06/`).

### Historical verified summary: `baseline_resnet_log`

- Doc: [VERIFIED_RESULTS_SUMMARY.md](VERIFIED_RESULTS_SUMMARY.md)
- Evidence files it cites include:
  - `results/if2rna_models/baseline_resnet_log/evaluation/summary_statistics.csv`
  - `results/if2rna_models/baseline_resnet_log/evaluation/gene_correlations.csv`
  - etc.

Key numbers (copied from the verified summary):

- Test samples: 941
- Genes evaluated: 11,981
- Median correlation: 0.766
- Mean correlation: 0.735
- MAE (log1p): 0.431

### Phase2 “ID benchmark table” (attention-MIL vs baselines)

The presentation/packaged benchmark comparison is captured in:

- CSV: [results/presentation_assets/2026-04-06/table_id_main_with_amil.csv](results/presentation_assets/2026-04-06/table_id_main_with_amil.csv)
- PNG: [results/presentation_assets/2026-04-06/table_id_main_with_amil.png](results/presentation_assets/2026-04-06/table_id_main_with_amil.png)

The *exact numbers* from the CSV are:

| model | n_samples | n_genes | mean r | median r | MAE | RMSE | #genes r>0.3 | #genes r>0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_mil | 941 | 11981 | 0.7441 | 0.7879 | 0.4435 | 0.6210 | 11814 | 11084 |
| vis_main | 941 | 11981 | 0.6793 | 0.6973 | 0.4710 | 0.7413 | 11834 | 11057 |
| vit_baseline | 941 | 11981 | 0.6473 | 0.6633 | 0.5376 | 0.7620 | 11833 | 10785 |
| meanpool_ridge_baseline | 941 | 11981 | 0.6213 | 0.6471 | 0.5730 | 0.7624 | 11347 | 9149 |

Where these rows come from (evidence folders):

- attention_mil: [results/if2rna_models/phase2_attention_mil/](results/if2rna_models/phase2_attention_mil)
- vis_main: [results/if2rna_models/phase2_vis_main/](results/if2rna_models/phase2_vis_main)
- vit_baseline: [results/if2rna_models/phase2_vit_baseline/](results/if2rna_models/phase2_vit_baseline)
- meanpool_ridge_baseline: [results/if2rna_models/phase2_meanpool_baseline/](results/if2rna_models/phase2_meanpool_baseline)

### HE2RNA-style pooling adaptation (exists + has metrics)

There is a completed HE2RNA-style adaptation run with saved metrics:

- Folder: [results/if2rna_models/phase2_he2rna_style_baseline_20260410/](results/if2rna_models/phase2_he2rna_style_baseline_20260410)
- Summary metrics: [results/if2rna_models/phase2_he2rna_style_baseline_20260410/summary_metrics.csv](results/if2rna_models/phase2_he2rna_style_baseline_20260410/summary_metrics.csv)

Its summary row (from that CSV) includes:

- mean r = 0.6712, median r = 0.7059, MAE = 0.5458, RMSE = 0.7420 (with `ks = 5|10|20`).

### Elastic-net baseline status

The repo includes an elastic-net baseline script:

- [if2rna_scripts/train_if2rna_elastic_net.py](if2rna_scripts/train_if2rna_elastic_net.py)

However, at least one expected output folder is empty:

- [results/if2rna_models/phase2_elastic_net_baseline/](results/if2rna_models/phase2_elastic_net_baseline) (empty at time of this writeup)

And the run status table labels elastic net as OOM:

- [results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv](results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv)

---

## OOD generalization experiments (held-out slide, held-out organ)

OOD summary tables exist in:

- Aggregate OOD summary (phase3): [results/phase3_summary/ood_aggregate_model_comparison.csv](results/phase3_summary/ood_aggregate_model_comparison.csv)
- Presentation OOD table: [results/presentation_assets/2026-04-06/table_ood_aggregate.csv](results/presentation_assets/2026-04-06/table_ood_aggregate.csv)
- Slide-level per-split table (example): [results/presentation_assets/2026-04-06/table_slide_level_model_comparison.csv](results/presentation_assets/2026-04-06/table_slide_level_model_comparison.csv)

### Held-out organ OOD

The aggregate OOD table indicates held-out-organ performance is near-zero mean correlations for several models (this is a key “generalization is hard” finding).

Evidence folders exist for held-out organ:

- [results/phase3_ood/heldout_organ_vis_overnight_20260410/](results/phase3_ood/heldout_organ_vis_overnight_20260410)
- [results/phase3_ood/heldout_organ_vit_overnight_20260410/](results/phase3_ood/heldout_organ_vit_overnight_20260410)
- [results/phase3_ood/heldout_organ_amil_overnight_20260410/](results/phase3_ood/heldout_organ_amil_overnight_20260410)
- [results/phase3_ood/heldout_organ_meanpool_overnight_20260410/](results/phase3_ood/heldout_organ_meanpool_overnight_20260410)

### Held-out slide OOD

Held-out slide experiments exist for:

- Meanpool: [results/phase3_ood/heldout_slide_meanpool_overnight_20260410/](results/phase3_ood/heldout_slide_meanpool_overnight_20260410)
- ViT: [results/phase3_ood/heldout_slide_vit_overnight_20260410/](results/phase3_ood/heldout_slide_vit_overnight_20260410)
- Attention-MIL: [results/phase3_ood/heldout_slide_amil_overnight_20260410/](results/phase3_ood/heldout_slide_amil_overnight_20260410)

There are also folders suggesting ViS was attempted but may have timed out:

- [results/phase3_ood/heldout_slide_vis/](results/phase3_ood/heldout_slide_vis)
- [results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv](results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv)

---

## Virtual expression maps + ROI-consistency checks

Two separate (but related) deliverables exist:

1. **Dense per-tile prediction heatmaps** (virtual maps)
2. **ROI-consistency** (aggregate tile predictions within an ROI and compare to the measured ROI RNA)

### Dense virtual expression maps

- Generator script: [if2rna_scripts/generate_virtual_expression_maps.py](if2rna_scripts/generate_virtual_expression_maps.py)

Evidence of generated maps (PNG + NPY) exists under:

- [results/final_package/virtual_maps/amil_dense_maps_20260411_stride28/](results/final_package/virtual_maps/amil_dense_maps_20260411_stride28)

Example concrete outputs include:

- `heatmap_*.png` + `heatmap_*.npy` per ROI (e.g. in nested folders under the slide directory)
- `tile_coords.npy` and `tile_preds.npy`
- [results/final_package/virtual_maps/amil_dense_maps_20260411_stride28/virtual_map_summary.csv](results/final_package/virtual_maps/amil_dense_maps_20260411_stride28/virtual_map_summary.csv)

### ROI-consistency check

- Script: [if2rna_scripts/evaluate_roi_consistency_from_tiles.py](if2rna_scripts/evaluate_roi_consistency_from_tiles.py)

Evidence outputs exist under:

- [results/final_package/virtual_maps/roi_consistency_amil_20260410/](results/final_package/virtual_maps/roi_consistency_amil_20260410)

Including the summary:

- [results/final_package/virtual_maps/roi_consistency_amil_20260410/roi_consistency_summary.csv](results/final_package/virtual_maps/roi_consistency_amil_20260410/roi_consistency_summary.csv)

The “proposal live artifact check” also confirms availability:

- [results/final_package/report/proposal_virtual_map_artifacts_live.csv](results/final_package/report/proposal_virtual_map_artifacts_live.csv)

---

## Target-set work: HVG-on-train-only package

The project proposal asked for: “top 2000 HVGs computed on training only (+ markers)”.

A concrete implementation exists:

- Script: [scripts/build_hvg_reference.py](scripts/build_hvg_reference.py)

Its output package exists under:

- [results/final_package/targets/](results/final_package/targets)

Notable files:

- [results/final_package/targets/hvg_train_only_top2000.csv](results/final_package/targets/hvg_train_only_top2000.csv)
- [results/final_package/targets/marker_gene_list.csv](results/final_package/targets/marker_gene_list.csv)
- [results/final_package/targets/selected_targets_hvg_plus_markers.csv](results/final_package/targets/selected_targets_hvg_plus_markers.csv)
- Split patient lists:
  - [results/final_package/targets/split_train_patients.csv](results/final_package/targets/split_train_patients.csv)
  - [results/final_package/targets/split_val_patients.csv](results/final_package/targets/split_val_patients.csv)
  - [results/final_package/targets/split_test_patients.csv](results/final_package/targets/split_test_patients.csv)

This is *proposal-compliant target selection* (train-only statistics), even if not used as the default across every legacy experiment.

---

## SEQUOIA H&E pipeline (TCGA baseline + verification)

The SEQUOIA pipeline is documented in:

- [docs/SEQUOIA_PIPELINE.md](docs/SEQUOIA_PIPELINE.md)

And its pre-run check / missing-file list is in:

- [PRE_RUN_CHECKLIST.md](PRE_RUN_CHECKLIST.md)

Key scripts (under [scripts/](scripts/)):

- [scripts/preprocess_slides.py](scripts/preprocess_slides.py)
- [scripts/download_sequoia_model.py](scripts/download_sequoia_model.py)
- [scripts/run_sequoia_inference.py](scripts/run_sequoia_inference.py)
- [scripts/evaluate_predictions.py](scripts/evaluate_predictions.py)

Evidence of SEQUOIA inference results exists in:

- [results/predictions_brca-0.csv](results/predictions_brca-0.csv)
- [results/predictions_brca-0_fixed.csv](results/predictions_brca-0_fixed.csv)
- [results/predictions_brca-0_correlations.csv](results/predictions_brca-0_correlations.csv)

…and similarly for folds 1–4.

A “verified SEQUOIA baseline” summary (for BRCA fold 0, 10 samples) is in:

- [VERIFIED_RESULTS_SUMMARY.md](VERIFIED_RESULTS_SUMMARY.md)

---

## ROSIE pipeline integration (conversion + inference artifacts)

This repo includes tooling and outputs for running a “ROSIE” conversion + inference pathway.

Evidence includes:

- ROSIE data workspace: [data/rosie_if/](data/rosie_if)
- ROSIE model/code drop: [models/rosie/](models/rosie)
- ROSIE IF2RNA predictions/correlations:
  - [results/rosie_if2rna/predictions_rosie_brca_88_fold4.csv](results/rosie_if2rna/predictions_rosie_brca_88_fold4.csv)
  - [results/rosie_if2rna/correlations_rosie_brca_88_fold4.csv](results/rosie_if2rna/correlations_rosie_brca_88_fold4.csv)
  - [results/rosie_if2rna/predictions_rosie_kirc_20_fold4.csv](results/rosie_if2rna/predictions_rosie_kirc_20_fold4.csv)
  - [results/rosie_if2rna/correlations_rosie_kirc_20_fold4.csv](results/rosie_if2rna/correlations_rosie_kirc_20_fold4.csv)

Supporting conversion scripts exist under [scripts/](scripts/), including:

- [scripts/convert_rosie_tiff_to_rgb.py](scripts/convert_rosie_tiff_to_rgb.py)
- [scripts/aggregate_rosie_tiles_to_slide_rgb.py](scripts/aggregate_rosie_tiles_to_slide_rgb.py)
- [scripts/prepare_rosie_inputs_from_wsi.py](scripts/prepare_rosie_inputs_from_wsi.py)
- [scripts/prepare_rosie_tile_inputs_from_wsi.py](scripts/prepare_rosie_tile_inputs_from_wsi.py)
- [scripts/run_rosie_conversion.py](scripts/run_rosie_conversion.py)
- [scripts/download_rosie_model.py](scripts/download_rosie_model.py)

---

## Packaging and presentation assets (tables, figures, “final_package”)

### Presentation asset pack

The repo contains a ready-to-drop-in asset pack under:

- [results/presentation_assets/2026-04-06/](results/presentation_assets/2026-04-06)

This includes:

- ID leaderboard / bar charts / error bars:
  - [results/presentation_assets/2026-04-06/fig_01_id_mean_median_corr_grouped_bar.png](results/presentation_assets/2026-04-06/fig_01_id_mean_median_corr_grouped_bar.png)
  - [results/presentation_assets/2026-04-06/fig_03_id_error_bars.png](results/presentation_assets/2026-04-06/fig_03_id_error_bars.png)
- Gene-wise distribution plots:
  - [results/presentation_assets/2026-04-06/fig_06_gene_corr_boxplot.png](results/presentation_assets/2026-04-06/fig_06_gene_corr_boxplot.png)
  - [results/presentation_assets/2026-04-06/fig_07_gene_corr_cdf.png](results/presentation_assets/2026-04-06/fig_07_gene_corr_cdf.png)
  - [results/presentation_assets/2026-04-06/fig_08_gene_corr_rank_curves.png](results/presentation_assets/2026-04-06/fig_08_gene_corr_rank_curves.png)
- OOD figures:
  - [results/presentation_assets/2026-04-06/fig_09_ood_corr_by_split.png](results/presentation_assets/2026-04-06/fig_09_ood_corr_by_split.png)
  - [results/presentation_assets/2026-04-06/fig_10_ood_error_by_split.png](results/presentation_assets/2026-04-06/fig_10_ood_error_by_split.png)

The script that builds many of these assets is:

- [scripts/build_presentation_assets.py](scripts/build_presentation_assets.py)

### Final-package “proposal live” tables

A separate report/aggregation system writes proposal-style tables into:

- [results/final_package/report/](results/final_package/report)

Key file:

- [results/final_package/report/proposal_unified_live_table.csv](results/final_package/report/proposal_unified_live_table.csv)

This unified table includes:

- ID models (phase2 baselines + HVG runs)
- OOD slide summaries
- OOD organ summaries

The script that produces these tables is:

- [scripts/aggregate_proposal_results_live.py](scripts/aggregate_proposal_results_live.py)

SLURM wrappers exist:

- [jobs/run_proposal_live_aggregator.sh](jobs/run_proposal_live_aggregator.sh)
- [jobs/run_proposal_live_aggregator_cpu.sh](jobs/run_proposal_live_aggregator_cpu.sh)

---

## Full inventory of scripts (IF2RNA, SEQUOIA utilities, jobs)

This section is an inventory of key code surfaces *that exist in the repo*, with brief purpose statements.

### IF2RNA scripts ([if2rna_scripts/](if2rna_scripts/))

Core pipeline:

- [if2rna_scripts/extract_if_images.py](if2rna_scripts/extract_if_images.py): extract ROI PNGs from NanoString zips.
- [if2rna_scripts/create_if_reference_csv.py](if2rna_scripts/create_if_reference_csv.py): join images to expression → `if_reference.csv`.
- [if2rna_scripts/preprocess_if_patches.py](if2rna_scripts/preprocess_if_patches.py): tile ROI PNGs into patches stored in HDF5.
- [if2rna_scripts/preprocess_if_features.py](if2rna_scripts/preprocess_if_features.py): patch → feature embeddings (ResNet50/UNI).
- [if2rna_scripts/preprocess_if_kmeans.py](if2rna_scripts/preprocess_if_kmeans.py): K-means over patch embeddings → `cluster_features`.
- [if2rna_scripts/run_if_preprocessing.py](if2rna_scripts/run_if_preprocessing.py): orchestration wrapper to run `patches`→`features`→`kmeans`.

Training / baselines:

- [if2rna_scripts/train_if2rna.py](if2rna_scripts/train_if2rna.py): main trainer for `vis`, `vit`, `attention_mil`.
- [if2rna_scripts/train_if2rna_mean_pool.py](if2rna_scripts/train_if2rna_mean_pool.py): mean pooling baseline (ridge-style).
- [if2rna_scripts/train_if2rna_he2rna_style.py](if2rna_scripts/train_if2rna_he2rna_style.py): HE2RNA-style top-k pooling adaptation.
- [if2rna_scripts/train_if2rna_elastic_net.py](if2rna_scripts/train_if2rna_elastic_net.py): handcrafted-ish baseline via elastic net.

Evaluation / inference:

- [if2rna_scripts/evaluate_if2rna.py](if2rna_scripts/evaluate_if2rna.py): evaluate `test_results.pkl` and write summary CSVs + plots.
- [if2rna_scripts/run_if2rna_inference.py](if2rna_scripts/run_if2rna_inference.py): run inference on new cohorts using a checkpoint.

OOD runners:

- [if2rna_scripts/run_heldout_slide_meanpool.py](if2rna_scripts/run_heldout_slide_meanpool.py)
- [if2rna_scripts/run_heldout_slide_neural.py](if2rna_scripts/run_heldout_slide_neural.py)
- [if2rna_scripts/run_heldout_organ_meanpool.py](if2rna_scripts/run_heldout_organ_meanpool.py)
- [if2rna_scripts/run_heldout_organ_neural.py](if2rna_scripts/run_heldout_organ_neural.py)

Virtual mapping:

- [if2rna_scripts/generate_virtual_expression_maps.py](if2rna_scripts/generate_virtual_expression_maps.py): generate dense per-tile heatmaps.
- [if2rna_scripts/generate_virtual_expression_proxy.py](if2rna_scripts/generate_virtual_expression_proxy.py): proxy mapping/attribution variant.
- [if2rna_scripts/evaluate_roi_consistency_from_tiles.py](if2rna_scripts/evaluate_roi_consistency_from_tiles.py): ROI-consistency evaluation.

Misc:

- [if2rna_scripts/validate_if_data.py](if2rna_scripts/validate_if_data.py)
- [if2rna_scripts/validate_preprocessing.py](if2rna_scripts/validate_preprocessing.py)
- [if2rna_scripts/download_resnet50.py](if2rna_scripts/download_resnet50.py)

### SEQUOIA + utilities ([scripts/](scripts/))

SEQUOIA pipeline:

- [scripts/download_tcga_data.py](scripts/download_tcga_data.py)
- [scripts/create_reference_csv.py](scripts/create_reference_csv.py)
- [scripts/download_uni_model.py](scripts/download_uni_model.py)
- [scripts/preprocess_slides.py](scripts/preprocess_slides.py)
- [scripts/download_sequoia_model.py](scripts/download_sequoia_model.py)
- [scripts/run_sequoia_inference.py](scripts/run_sequoia_inference.py)
- [scripts/evaluate_predictions.py](scripts/evaluate_predictions.py)
- [scripts/fix_prediction_columns.py](scripts/fix_prediction_columns.py)

Aggregation/packaging:

- [scripts/build_presentation_assets.py](scripts/build_presentation_assets.py)
- [scripts/aggregate_proposal_results_live.py](scripts/aggregate_proposal_results_live.py)
- [scripts/update_report_ready_summary.py](scripts/update_report_ready_summary.py)

ROSIE conversion/tooling (subset):

- [scripts/convert_rosie_tiff_to_rgb.py](scripts/convert_rosie_tiff_to_rgb.py)
- [scripts/run_rosie_conversion.py](scripts/run_rosie_conversion.py)
- [scripts/download_rosie_model.py](scripts/download_rosie_model.py)

### SLURM job scripts ([jobs/](jobs/))

This repo is job-script heavy; notable groups:

- Phase2 training baselines:
  - [jobs/train_if2rna_phase2_vis.sh](jobs/train_if2rna_phase2_vis.sh)
  - [jobs/train_if2rna_phase2_vit.sh](jobs/train_if2rna_phase2_vit.sh)
  - [jobs/train_if2rna_phase2_attention_mil.sh](jobs/train_if2rna_phase2_attention_mil.sh)
  - [jobs/train_if2rna_phase2_meanpool.sh](jobs/train_if2rna_phase2_meanpool.sh)
  - [jobs/train_if2rna_phase2_he2rna_style.sh](jobs/train_if2rna_phase2_he2rna_style.sh)
  - elastic-net variants:
    - [jobs/train_if2rna_phase2_elastic_net.sh](jobs/train_if2rna_phase2_elastic_net.sh)
    - [jobs/train_if2rna_phase2_elastic_net_safe.sh](jobs/train_if2rna_phase2_elastic_net_safe.sh)
    - [jobs/train_if2rna_phase2_elastic_net_rescue.sh](jobs/train_if2rna_phase2_elastic_net_rescue.sh)
    - [jobs/train_if2rna_phase2_elastic_net_rescue_cpu.sh](jobs/train_if2rna_phase2_elastic_net_rescue_cpu.sh)

- Phase3 OOD:
  - held-out slide: [jobs/run_phase3_heldout_slide_vis.sh](jobs/run_phase3_heldout_slide_vis.sh), [jobs/run_phase3_heldout_slide_vit.sh](jobs/run_phase3_heldout_slide_vit.sh), [jobs/run_phase3_heldout_slide_meanpool.sh](jobs/run_phase3_heldout_slide_meanpool.sh)
  - held-out organ: [jobs/run_phase3_heldout_organ_vis.sh](jobs/run_phase3_heldout_organ_vis.sh), [jobs/run_phase3_heldout_organ_vit.sh](jobs/run_phase3_heldout_organ_vit.sh), [jobs/run_phase3_heldout_organ_meanpool.sh](jobs/run_phase3_heldout_organ_meanpool.sh)

- Virtual maps + ROI consistency:
  - [jobs/run_virtual_maps_amil.sh](jobs/run_virtual_maps_amil.sh)
  - [jobs/run_virtual_maps_amil_stride28.sh](jobs/run_virtual_maps_amil_stride28.sh)
  - [jobs/run_roi_consistency_amil.sh](jobs/run_roi_consistency_amil.sh)

- SEQUOIA runs:
  - [jobs/run_sequoia_end_to_end.sh](jobs/run_sequoia_end_to_end.sh)
  - [jobs/run_sequoia_brca_all_folds.sh](jobs/run_sequoia_brca_all_folds.sh)
  - [jobs/run_sequoia_smoke_test.sh](jobs/run_sequoia_smoke_test.sh)

---

## Known mismatches, sharp edges, and failure modes

This section captures “gotchas” that are either documented elsewhere or evidenced by failed runs.

### 1) Slide-content doc includes outdated dataset claims

[PRESENTATION_SLIDES_CONTENT.md](PRESENTATION_SLIDES_CONTENT.md) lists “lung, pancreas, prostate, skin” as organs; the actual raw IF data folders present are only:

- [data/if_data/](data/if_data): Colon, Kidney, Liver, Lymph Node

Treat the slide-content doc as narrative; treat the freeze YAML + archive reference CSVs as authoritative.

### 2) Offline compute nodes (HPC) require staged model weights

- UNI requires HuggingFace cache on login node: [docs/UNI_SETUP_HPC.md](docs/UNI_SETUP_HPC.md)
- SEQUOIA end-to-end pitfalls are enumerated in: [PRE_RUN_CHECKLIST.md](PRE_RUN_CHECKLIST.md)

### 3) Elastic net baseline instability / OOM

- Run status indicates OOM: [results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv](results/presentation_assets/2026-04-06/table_run_status_and_blockers.csv)
- One expected output folder is empty: [results/if2rna_models/phase2_elastic_net_baseline/](results/if2rna_models/phase2_elastic_net_baseline)

### 4) OOD generalization can be extremely weak

- Held-out organ OOD summary shows near-zero mean correlations for multiple models (see [results/phase3_summary/ood_aggregate_model_comparison.csv](results/phase3_summary/ood_aggregate_model_comparison.csv)).

This is not necessarily a bug; it’s a core scientific finding (domain shift is hard).

### 5) “Proposal coverage” status is tracked separately

There is an explicit proposal closure checklist:

- [docs/IF2RNA_PROPOSAL_TO_REPO_CHECKLIST.md](docs/IF2RNA_PROPOSAL_TO_REPO_CHECKLIST.md)

This document may lag behind actual artifacts (e.g., ROI-consistency outputs exist under `results/final_package/virtual_maps/` even if the checklist still says FAIL in earlier snapshots).

---

## Appendix: practical “how to rerun” command index

This is a consolidated command index (templates); always prefer job scripts under [jobs/](jobs/) on HPC.

### IF2RNA preprocessing (full pipeline)

Use the orchestrator:

```bash
python if2rna_scripts/run_if_preprocessing.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feat_type resnet \
  --model_dir models/resnet50 \
  --steps patches features kmeans
```

### IF2RNA training (main models)

```bash
python if2rna_scripts/train_if2rna.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_attention_mil \
  --model_type attention_mil \
  --log_transform \
  --train
```

(Replace `--model_type` with `vis` or `vit` for those baselines.)

### IF2RNA evaluation

```bash
python if2rna_scripts/evaluate_if2rna.py \
  --results_file results/if2rna_models/phase2_attention_mil/test_results.pkl \
  --reference_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --output_dir results/if2rna_models/phase2_attention_mil/evaluation
```

### OOD held-out slide/organ

Prefer the job scripts in [jobs/](jobs/). The underlying python entrypoints are in [if2rna_scripts/](if2rna_scripts/) with “heldout” in the name.

### Virtual maps

Prefer job scripts:

- [jobs/run_virtual_maps_amil.sh](jobs/run_virtual_maps_amil.sh)
- [jobs/run_virtual_maps_amil_stride28.sh](jobs/run_virtual_maps_amil_stride28.sh)

### ROI-consistency

Prefer:

- [jobs/run_roi_consistency_amil.sh](jobs/run_roi_consistency_amil.sh)

### Proposal/unified table aggregation

```bash
python scripts/aggregate_proposal_results_live.py
```

Outputs land in:

- [results/final_package/report/](results/final_package/report)

---

## Primary “already-written exhaustive log” (historical anchor)

If you want even more “execution log” narrative (including status of specific runs as of 2026-04-06), see:

- [docs/IF2RNA_COMPREHENSIVE_METHODS_MODELS_FINDINGS_2026-04-06.md](docs/IF2RNA_COMPREHENSIVE_METHODS_MODELS_FINDINGS_2026-04-06.md)
