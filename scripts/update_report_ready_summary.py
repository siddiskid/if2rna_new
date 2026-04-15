#!/usr/bin/env python3
"""Build a unified report-ready model summary table from available result snapshots."""

from pathlib import Path

import pandas as pd


def main() -> None:
    out_dir = Path("results/phase3_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    id_path = out_dir / "phase2_model_comparison_snapshot.csv"
    ood_path = out_dir / "ood_aggregate_model_comparison.csv"
    org_path = out_dir / "phase3_per_organ_model_comparison.csv"

    if not id_path.exists():
        raise FileNotFoundError(id_path)

    id_df = pd.read_csv(id_path)
    ood_df = pd.read_csv(ood_path) if ood_path.exists() else pd.DataFrame()
    org_df = pd.read_csv(org_path) if org_path.exists() else pd.DataFrame()

    name_map = {
        "vis_main": "vis",
        "vit_baseline": "vit",
        "meanpool_ridge_baseline": "meanpool_ridge_baseline",
        "attention_mil": "attention_mil",
        "amil": "attention_mil",
        "vis": "vis",
        "vit": "vit",
    }

    id_df["model_key"] = id_df["model"].map(name_map).fillna(id_df["model"])

    id_keep = id_df[[
        "model_key", "model", "n_samples", "n_genes", "mean_gene_correlation", "median_gene_correlation",
        "mae", "rmse", "genes_r_gt_0_3", "genes_r_gt_0_5"
    ]].rename(columns={
        "model": "id_model_name",
        "n_samples": "id_n_samples",
        "n_genes": "id_n_genes",
        "mean_gene_correlation": "id_mean_gene_correlation",
        "median_gene_correlation": "id_median_gene_correlation",
        "mae": "id_mae",
        "rmse": "id_rmse",
        "genes_r_gt_0_3": "id_genes_r_gt_0_3",
        "genes_r_gt_0_5": "id_genes_r_gt_0_5",
    })

    combined = id_keep.copy()

    if not ood_df.empty:
        ood_df["model_key"] = ood_df["model"].map(name_map).fillna(ood_df["model"])
        ood_keep = ood_df[[
            "model_key", "model", "ood_mean_of_means", "ood_mean_of_medians", "ood_mean_mae", "ood_mean_rmse"
        ]].rename(columns={"model": "ood_model_name"})
        combined = combined.merge(ood_keep, on="model_key", how="outer")

    if not org_df.empty:
        org_df["model_key"] = org_df["model"].map(name_map).fillna(org_df["model"])
        org_agg = (
            org_df.groupby("model_key", as_index=False)
            .agg(
                per_organ_mean_corr=("mean_correlation", "mean"),
                per_organ_mean_mae=("mae", "mean"),
                per_organ_mean_rmse=("rmse", "mean"),
                per_organ_total_samples=("n_samples", "sum"),
            )
        )
        combined = combined.merge(org_agg, on="model_key", how="outer")

    if {"id_mean_gene_correlation", "ood_mean_of_means"}.issubset(set(combined.columns)):
        combined["delta_id_minus_ood_mean_corr"] = combined["id_mean_gene_correlation"] - combined["ood_mean_of_means"]

    combined = combined.sort_values("id_mean_gene_correlation", ascending=False, na_position="last")
    out_file = out_dir / "report_ready_model_summary.csv"
    combined.to_csv(out_file, index=False)
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
