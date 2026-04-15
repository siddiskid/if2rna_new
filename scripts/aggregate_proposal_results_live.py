#!/usr/bin/env python3
"""Aggregate live IF2RNA results into proposal-ready comparison tables."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "results" / "if2rna_models"
OOD_DIR = ROOT / "results" / "phase3_ood"
OUT_DIR = ROOT / "results" / "final_package" / "report"


def first_dir_with_summary(paths: list[Path]) -> Path | None:
    for p in paths:
        if (p / "summary_metrics.csv").exists() or (p / "test_results.pkl").exists():
            return p
    return None


def gene_corrs(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    corr = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    for i in range(y_true.shape[1]):
        t = y_true[:, i]
        p = y_pred[:, i]
        mask = ~(np.isnan(t) | np.isnan(p))
        if mask.sum() < 2:
            continue
        if np.std(t[mask]) == 0 or np.std(p[mask]) == 0:
            continue
        corr[i] = np.corrcoef(t[mask], p[mask])[0, 1]
    return corr


def summarize_test_results_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    split_keys = sorted([k for k in obj.keys() if str(k).startswith("split_")])
    preds_all, real_all = [], []
    for k in split_keys:
        d = obj[k]
        pred = np.asarray(d.get("preds", []))
        real = np.asarray(d.get("real", []))
        if pred.size == 0 or real.size == 0:
            continue
        preds_all.append(pred)
        real_all.append(real)
    if not preds_all:
        return None

    preds = np.vstack(preds_all)
    real = np.vstack(real_all)
    corr = gene_corrs(real, preds)
    return {
        "n_samples": int(real.shape[0]),
        "n_genes": int(real.shape[1]),
        "mean_gene_correlation": float(np.nanmean(corr)),
        "median_gene_correlation": float(np.nanmedian(corr)),
        "mae": float(np.mean(np.abs(preds - real))),
        "rmse": float(np.sqrt(np.mean((preds - real) ** 2))),
        "genes_r_gt_0_3": int(np.sum(corr > 0.3)),
        "genes_r_gt_0_5": int(np.sum(corr > 0.5)),
    }


def collect_id_models() -> pd.DataFrame:
    elastic_net_dir = first_dir_with_summary([
        MODELS_DIR / "phase2_elastic_net_rescue_20260413",
        MODELS_DIR / "phase2_elastic_net_safe_20260411",
        MODELS_DIR / "phase2_elastic_net_baseline",
    ])

    candidates = {
        "vis_main": MODELS_DIR / "phase2_vis_main",
        "vit_baseline": MODELS_DIR / "phase2_vit_baseline",
        "meanpool_ridge_baseline": MODELS_DIR / "phase2_meanpool_baseline",
        "attention_mil": MODELS_DIR / "phase2_attention_mil",
        "he2rna_style": MODELS_DIR / "phase2_he2rna_style_baseline_20260410",
        "vis_hvg2k": MODELS_DIR / "if2rna_vis_hvg2000_20260410",
        "amil_hvg2k": MODELS_DIR / "if2rna_amil_hvg2000_20260410",
        "meanpool_hvg2k": MODELS_DIR / "if2rna_meanpool_hvg2000_20260410",
    }
    if elastic_net_dir is not None:
        candidates["elastic_net"] = elastic_net_dir

    rows = []
    for name, d in candidates.items():
        sm = d / "summary_metrics.csv"
        tr = d / "test_results.pkl"
        if sm.exists():
            sdf = pd.read_csv(sm)
            if len(sdf) == 0:
                continue
            row = sdf.iloc[0].to_dict()
            mapped = {
                "model": name,
                "n_samples": int(row.get("n_samples_total", row.get("n_samples", np.nan))),
                "n_genes": int(row.get("n_genes", np.nan)),
                "mean_gene_correlation": float(row.get("mean_gene_pearson_r", row.get("mean_gene_correlation", np.nan))),
                "median_gene_correlation": float(row.get("median_gene_pearson_r", row.get("median_gene_correlation", np.nan))),
                "mae": float(row.get("mae", np.nan)),
                "rmse": float(row.get("rmse", np.nan)),
                "genes_r_gt_0_3": int(row.get("genes_r_gt_0_3", 0)),
                "genes_r_gt_0_5": int(row.get("genes_r_gt_0_5", 0)),
            }
            rows.append(mapped)
        elif tr.exists():
            stats = summarize_test_results_pkl(tr)
            if stats is None:
                continue
            stats["model"] = name
            rows.append(stats)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_gene_correlation", ascending=False).reset_index(drop=True)
    return df


def summarize_ood_table(summary_path: Path, key_col: str, model_name: str):
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if len(df) == 0:
        return None
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if len(df) == 0:
        return None
    return {
        "model": model_name,
        "ood_mean_of_means": float(df["mean_gene_correlation"].mean()),
        "ood_mean_of_medians": float(df["median_gene_correlation"].mean()),
        "ood_mean_mae": float(df["mae"].mean()),
        "ood_mean_rmse": float(df["rmse"].mean()),
        "n_splits": int(df[key_col].nunique()) if key_col in df.columns else int(len(df)),
    }


def collect_ood() -> tuple[pd.DataFrame, pd.DataFrame]:
    slide_targets = {
        "vit_ood_slide": OOD_DIR / "heldout_slide_vit_overnight_20260410" / "heldout_slide_summary.csv",
        "amil_ood_slide": OOD_DIR / "heldout_slide_amil_overnight_20260410" / "heldout_slide_summary.csv",
        "meanpool_ood_slide": OOD_DIR / "heldout_slide_meanpool_overnight_20260410" / "heldout_slide_summary.csv",
    }
    organ_targets = {
        "vis_ood_organ": OOD_DIR / "heldout_organ_vis_overnight_20260410" / "heldout_organ_summary.csv",
        "vit_ood_organ": OOD_DIR / "heldout_organ_vit_overnight_20260410" / "heldout_organ_summary.csv",
        "amil_ood_organ": OOD_DIR / "heldout_organ_amil_overnight_20260410" / "heldout_organ_summary.csv",
        "meanpool_ood_organ": OOD_DIR / "heldout_organ_meanpool_overnight_20260410" / "heldout_organ_summary.csv",
    }

    slide_rows = []
    for model, path in slide_targets.items():
        row = summarize_ood_table(path, key_col="heldout_slide", model_name=model)
        if row is not None:
            slide_rows.append(row)

    organ_rows = []
    for model, path in organ_targets.items():
        row = summarize_ood_table(path, key_col="heldout_organ", model_name=model)
        if row is not None:
            organ_rows.append(row)

    return pd.DataFrame(slide_rows), pd.DataFrame(organ_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    id_df = collect_id_models()
    slide_df, organ_df = collect_ood()

    if len(id_df):
        id_df.to_csv(OUT_DIR / "proposal_id_model_comparison_live.csv", index=False)
    if len(slide_df):
        slide_df.to_csv(OUT_DIR / "proposal_ood_slide_model_comparison_live.csv", index=False)
    if len(organ_df):
        organ_df.to_csv(OUT_DIR / "proposal_ood_organ_model_comparison_live.csv", index=False)

    # Unified compact report table.
    rows = []
    if len(id_df):
        for _, r in id_df.iterrows():
            rows.append({
                "table": "id",
                "model": r["model"],
                "mean_gene_correlation": r["mean_gene_correlation"],
                "median_gene_correlation": r["median_gene_correlation"],
                "mae": r["mae"],
                "rmse": r["rmse"],
            })
    if len(slide_df):
        for _, r in slide_df.iterrows():
            rows.append({
                "table": "ood_slide",
                "model": r["model"],
                "mean_gene_correlation": r["ood_mean_of_means"],
                "median_gene_correlation": r["ood_mean_of_medians"],
                "mae": r["ood_mean_mae"],
                "rmse": r["ood_mean_rmse"],
            })
    if len(organ_df):
        for _, r in organ_df.iterrows():
            rows.append({
                "table": "ood_organ",
                "model": r["model"],
                "mean_gene_correlation": r["ood_mean_of_means"],
                "median_gene_correlation": r["ood_mean_of_medians"],
                "mae": r["ood_mean_mae"],
                "rmse": r["ood_mean_rmse"],
            })

    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "proposal_unified_live_table.csv", index=False)

    # Include virtual-map and ROI-consistency summaries if available.
    vm_path = ROOT / "results" / "final_package" / "virtual_maps" / "amil_dense_maps_20260410" / "virtual_map_summary.csv"
    rc_path = ROOT / "results" / "final_package" / "virtual_maps" / "roi_consistency_amil_20260410" / "roi_consistency_summary.csv"

    extra_rows = []
    if vm_path.exists():
        vm = pd.read_csv(vm_path)
        if len(vm):
            extra_rows.append({
                "artifact": "virtual_maps",
                "status": "available",
                "n_rows": int(len(vm)),
                "n_ok": int((vm["status"] == "ok").sum()) if "status" in vm.columns else int(len(vm)),
            })
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        if len(rc):
            r = rc.iloc[0].to_dict()
            extra_rows.append({
                "artifact": "roi_consistency",
                "status": "available",
                "n_rows": 1,
                "n_ok": 1,
                "mean_gene_correlation": r.get("mean_gene_correlation", np.nan),
                "median_gene_correlation": r.get("median_gene_correlation", np.nan),
                "roi_mean_mae": r.get("roi_mean_mae", np.nan),
                "roi_mean_rmse": r.get("roi_mean_rmse", np.nan),
            })

    if extra_rows:
        pd.DataFrame(extra_rows).to_csv(OUT_DIR / "proposal_virtual_map_artifacts_live.csv", index=False)

    print(f"Wrote live proposal tables to: {OUT_DIR}")


if __name__ == "__main__":
    main()
