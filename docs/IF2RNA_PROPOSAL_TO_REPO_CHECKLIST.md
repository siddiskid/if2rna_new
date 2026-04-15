# IF2RNA Proposal-to-Repo Coverage Checklist

Date: 2026-04-10
Owner: Siddarth Chilukuri
Purpose: Track exactly what is needed for full (100%) coverage of the IF2RNA project proposal.

## How To Use This Checklist

- Mark each item as `PASS`, `PARTIAL`, or `FAIL`.
- A section is complete only when all required artifacts exist and validation checks pass.
- Keep one run per output folder and do not overwrite previous outputs.

## Current Coverage Snapshot (As Of 2026-04-10)

| Section | Status | Notes |
|---|---|---|
| ROI-level IF->RNA training pipeline | PASS | Implemented and run in current repo |
| Attention-MIL model | PASS | Implemented, trained, and summarized |
| Mean-pooling baseline | PASS | Implemented and benchmarked |
| Elastic-net handcrafted baseline | PARTIAL | Implemented but OOM in latest run |
| HE2RNA-style pooling adaptation baseline | PARTIAL | Mentioned in plan; no finalized comparable output table |
| Patient-level leakage control | PASS | Patient-level CV implemented |
| In-distribution benchmark metrics | PASS | r, MAE, RMSE, threshold counts reported |
| Held-out-study / held-out-slide OOD | PARTIAL | ViT + meanpool complete, ViS held-out-slide timed out |
| Held-out-organ OOD | PASS | Organ-level runs and summary tables exist |
| Ablations (pooling, bag cap, etc.) | PARTIAL | Some complete; full matrix not complete |
| Dense virtual expression maps | PARTIAL | Proxy script exists; full grid map pipeline not finalized |
| ROI-consistency checks for virtual maps | FAIL | Not complete as proposal-level deliverable |
| Final reproducible packaging | PARTIAL | Strong summaries exist; proposal-specific package checklist not fully closed |

---

## A) Data Curation And Inclusion Criteria

### A1. Dataset inclusion table (proposal Table 1 equivalent)
- Required fields: organ, study/cohort, slides, ROIs, IF channels, assay type.
- Required output:
  - `results/final_package/dataset/dataset_summary_table.csv`
  - `results/final_package/dataset/inclusion_exclusion_notes.md`
- Status: PARTIAL

Validation command:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('results/final_package/dataset/dataset_summary_table.csv')
required = {'organ','study','slides','rois','if_channels','assay_type'}
missing = required - set(df.columns)
print('missing_columns:', sorted(missing))
print('n_rows:', len(df))
PY
```

Pass criteria:
- No missing required columns.
- At least one row per included study.

### A2. MVP scale check
- Proposal target: >=300 ROIs across >=2 cohorts/studies.
- Status: PARTIAL (scale exists in data, but checklist output not formalized as a final package assertion).

Validation command:
```bash
python - <<'PY'
import pandas as pd
ref = pd.read_csv('data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv', low_memory=False)
print('n_rois:', len(ref))
study_col = 'study' if 'study' in ref.columns else ('cohort' if 'cohort' in ref.columns else None)
print('study_or_cohort_column:', study_col)
if study_col:
    print('n_studies_or_cohorts:', ref[study_col].nunique())
PY
```

Pass criteria:
- ROI count >= 300.
- At least 2 cohorts/studies (or explicit limitation note if unavailable).

---

## B) Preprocessing And Target Definition

### B1. HVG-on-train-only target set (top 2000 + markers)
- Required output:
  - `results/final_package/targets/hvg_train_only_top2000.csv`
  - `results/final_package/targets/marker_gene_list.csv`
- Status: FAIL (not currently locked as default training path).

Pass criteria:
- HVGs computed from training split only.
- No test leakage in target selection.

### B2. RNA normalization protocol lock
- Proposal default: log(CPM+1) (or documented fallback).
- Required output:
  - `results/final_package/targets/normalization_protocol.md`
- Status: PARTIAL (log transform exists, CPM protocol lock not yet formalized).

Pass criteria:
- One explicit normalization protocol used across all primary comparisons.
- Protocol and fallback documented with rationale.

### B3. Patch/bag construction defaults
- Proposal defaults: 224x224, effective ~10x, bag cap 256, mask-based sampling.
- Required output:
  - `results/final_package/preprocessing/preprocessing_config.yaml`
- Status: PARTIAL (bag cap support exists; full proposal defaults not consistently enforced/documented).

Pass criteria:
- Config file captures all defaults and is used by runs.

---

## C) Baselines And Main Model

### C1. Baseline 1: handcrafted + elastic net
- Script exists: `if2rna_scripts/train_if2rna_elastic_net.py`
- Required output:
  - `results/if2rna_models/phase2_elastic_net_baseline/summary_metrics.csv`
- Status: PARTIAL (OOM in latest run).

Validation command:
```bash
test -f results/if2rna_models/phase2_elastic_net_baseline/summary_metrics.csv && echo PASS || echo MISSING
```

### C2. Baseline 2: mean-pooling model
- Script: `if2rna_scripts/train_if2rna_mean_pool.py`
- Required output:
  - `results/if2rna_models/phase2_meanpool_baseline/summary_metrics.csv`
- Status: PASS

### C3. Baseline 3: HE2RNA-style pooling adaptation
- Required output:
  - `results/if2rna_models/phase2_he2rna_style_baseline/summary_metrics.csv`
  - clearly documented method note.
- Status: FAIL/PARTIAL (not finalized in benchmark outputs).

### C4. Main model: IF2RNA attention-MIL
- Script path: `if2rna_scripts/train_if2rna.py` with `--model_type attention_mil`
- Required output:
  - `results/if2rna_models/phase2_attention_mil/test_results.pkl`
- Status: PASS

---

## D) Evaluation And Generalization

### D1. In-distribution benchmark table
- Required metrics: mean/median gene-wise Pearson r, MAE, RMSE, counts above thresholds.
- Existing outputs:
  - `results/phase2_benchmark/phase2_model_comparison_with_deltas.csv`
  - `results/phase3_summary/phase2_model_comparison_snapshot.csv`
- Status: PASS

### D2. OOD: held-out study/slide
- Required output:
  - `results/phase3_ood/heldout_slide_*/heldout_slide_summary.csv`
  - aggregate comparison in `results/phase3_summary/ood_aggregate_model_comparison.csv`
- Status: PARTIAL (ViS held-out-slide incomplete due timeout).

### D3. OOD: held-out organ (if feasible)
- Existing outputs:
  - `results/phase3_summary/phase3_per_organ_model_comparison.csv`
  - `results/phase3_summary/phase3_per_organ_run_status.csv`
- Status: PASS

---

## E) Ablations

Required proposal ablations:
- Mean vs attention pooling
- Bag cap 128 vs 256
- Channels (nucleus-only vs all)
- Single-scale vs multi-scale
- MSE vs MSE+corr
- Mask vs box ROI geometry

Required output:
- `results/final_package/ablations/ablation_table.csv`

Status:
- Pooling: PASS
- Bag cap: PARTIAL
- Remaining ablations: FAIL/PARTIAL

Pass criteria:
- One compact table with deltas against chosen main model.

---

## F) Virtual Maps And ROI Consistency

### F1. Dense slide-wide virtual expression maps
- Current script: `if2rna_scripts/generate_virtual_expression_proxy.py` (patch contribution proxy).
- Required final output:
  - per-slide dense grid predictions for selected genes
  - saved map images/arrays under `results/final_package/virtual_maps/`
- Status: PARTIAL

### F2. ROI-consistency quantitative check
- Required output:
  - `results/final_package/virtual_maps/roi_consistency_metrics.csv`
  - metric definition doc.
- Status: FAIL

Pass criteria:
- For held-out set, aggregate tile predictions inside ROIs and correlate with measured ROI expression.

---

## G) Reproducibility And Packaging

### G1. One-command reproducibility list
- Required output:
  - `results/final_package/repro/reproduce_commands.sh`
  - `results/final_package/repro/environment.txt`
- Status: PARTIAL

### G2. Final report package
- Required outputs:
  - `results/final_package/report/main_benchmark_table.csv`
  - `results/final_package/report/ood_table.csv`
  - `results/final_package/report/ablation_table.csv`
  - `results/final_package/report/figures/`
  - `results/final_package/report/final_report.md`
- Status: PARTIAL (strong intermediate assets exist, final package naming not fully consolidated).

---

## Definition Of 100% Proposal Coverage

Mark project as 100% covered only when all of the following are true:

1. All items in Sections A-G are `PASS`.
2. Main benchmark includes all planned baseline families plus attention-MIL in one table.
3. OOD results include completed held-out-study/slide and held-out-organ (or explicit infeasibility note).
4. Virtual maps are dense grid outputs (not only proxy attribution) with ROI-consistency metrics.
5. A single final package folder can regenerate all proposal figures/tables from documented commands.

---

## Suggested Next Closure Sequence

1. Recover elastic-net baseline run without OOM and write summary output.
2. Close HE2RNA-style baseline adaptation into comparable metrics table.
3. Lock HVG-on-train-only target build and normalization protocol docs.
4. Complete dense virtual map generation plus ROI-consistency metrics.
5. Consolidate final package in `results/final_package/` with reproducibility scripts.
