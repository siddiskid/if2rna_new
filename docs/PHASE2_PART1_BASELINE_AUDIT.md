# Phase 2 Part 1 Baseline Audit

Date: 2026-04-04
Scope: Determine which baselines are already runnable in this repository for the frozen IF2RNA dataset.

## Summary
- Immediate runnable baseline exists: IF2RNA with model_type=vit (same training pipeline, different aggregator).
- Main model exists: IF2RNA with model_type=vis.
- HE2RNA baseline code exists in sequoia-pub, but is not directly compatible with current IF reference and feature layout.
- Handcrafted + elastic net baseline is not implemented yet.

## Baseline Availability Matrix

| Baseline | Status | Evidence | Blocker |
|---|---|---|---|
| IF2RNA ViS (main) | Ready now | if2rna_scripts/train_if2rna.py supports model_type=vis | None |
| IF2RNA ViT | Ready now | if2rna_scripts/train_if2rna.py supports model_type=vit | None |
| HE2RNA-style pooling | Partially available | sequoia-pub/src/he2rna.py | Data loader expects tcga_project paths |
| Handcrafted + elastic net | Missing | no dedicated baseline script found | needs new implementation |

## Compatibility Notes
- Frozen training reference uses organ_type and IF feature paths under data/if_features/<organ>/<sample>/<sample>.h5.
- sequoia-pub/src/read_data.py expects tcga_project and features_path/<tcga_project>/<wsi>/<wsi>.h5.
- Therefore HE2RNA from sequoia-pub cannot be dropped in directly without adapting data loading.

## Decision For 2-Day Sprint
Use the following baseline set for guaranteed progress:
1. Main model: ViS (model_type=vis).
2. Baseline A: ViT (model_type=vit).
3. Optional baseline B (if time): quick linear baseline on pooled cluster features.

## Small Next Steps (Execution Checklist)
1. Launch ViS run on frozen 941-sample training reference.
2. Launch ViT run with identical settings (seed, epochs, batch size).
3. Evaluate both runs with the same evaluation script and export comparable CSV metrics.
4. Create one table with mean/median gene-wise r, MAE, RMSE, and gene-threshold counts.
5. If time remains, implement simple pooled-feature linear baseline.

## Commands To Start (example)
ViS main run:
python if2rna_scripts/train_if2rna.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_vis_main \
  --model_type vis \
  --log_transform \
  --train

ViT baseline run:
python if2rna_scripts/train_if2rna.py \
  --ref_file data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv \
  --feature_dir data/if_features \
  --save_dir results/if2rna_models \
  --exp_name phase2_vit_baseline \
  --model_type vit \
  --log_transform \
  --train
