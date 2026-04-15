#!/usr/bin/env python3
"""Build train-only HVG target lists and reduced IF2RNA reference CSV.

Outputs:
- train/val/test patient splits
- top-K HVG list computed on train only
- marker list (if provided)
- selected target list (HVG union markers)
- reduced reference CSV containing only selected rna_* targets
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build proposal-compliant HVG reference")
    p.add_argument("--ref_file", required=True)
    p.add_argument("--out_dir", default="results/final_package/targets")
    p.add_argument("--top_k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--log1p_for_hvg", action="store_true")
    p.add_argument("--marker_file", default=None, help="Optional text/csv with one gene symbol per line")
    return p.parse_args()


def load_markers(marker_file: Optional[str]) -> List[str]:
    if marker_file is None:
        return []
    p = Path(marker_file)
    if not p.exists():
        return []
    vals = []
    for line in p.read_text().splitlines():
        v = line.strip().split(",")[0].strip()
        if not v:
            continue
        vals.append(v)
    return sorted(set(vals))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ref_file, low_memory=False)
    if "patient_id" not in df.columns:
        raise ValueError("reference CSV must contain patient_id")

    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if not rna_cols:
        raise ValueError("No rna_* columns found")

    # Keep RNA cleanup aligned with training scripts.
    bad_genes = df[rna_cols].isna().sum(axis=0) > (len(df) * 0.5)
    if bad_genes.any():
        drop_cols = [c for c, b in zip(rna_cols, bad_genes) if b]
        df = df.drop(columns=drop_cols)
        rna_cols = [c for c in df.columns if c.startswith("rna_")]

    df[rna_cols] = df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    patients = np.array(sorted(df["patient_id"].dropna().unique().tolist()))
    p_train, p_tmp = train_test_split(
        patients,
        train_size=args.train_frac,
        random_state=args.seed,
        shuffle=True,
    )
    val_ratio_of_tmp = args.val_frac / (args.val_frac + args.test_frac)
    p_val, p_test = train_test_split(
        p_tmp,
        train_size=val_ratio_of_tmp,
        random_state=args.seed,
        shuffle=True,
    )

    train_df = df[df["patient_id"].isin(p_train)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(p_val)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(p_test)].reset_index(drop=True)

    x_train = train_df[rna_cols].to_numpy(dtype=np.float64)
    if args.log1p_for_hvg:
        x_train = np.log1p(x_train)

    var = np.nanvar(x_train, axis=0)
    order = np.argsort(var)[::-1]
    top_k = min(args.top_k, len(rna_cols))
    hvg_cols = [rna_cols[i] for i in order[:top_k]]
    hvg_genes = [c[4:] for c in hvg_cols]

    marker_genes = load_markers(args.marker_file)
    marker_cols = [f"rna_{g}" for g in marker_genes if f"rna_{g}" in rna_cols]

    selected_cols = sorted(set(hvg_cols) | set(marker_cols))
    meta_cols = [c for c in df.columns if not c.startswith("rna_")]

    reduced_df = df[meta_cols + selected_cols].copy()
    reduced_train = train_df[meta_cols + selected_cols].copy()
    reduced_val = val_df[meta_cols + selected_cols].copy()
    reduced_test = test_df[meta_cols + selected_cols].copy()

    pd.DataFrame({"patient_id": sorted(p_train)}).to_csv(out_dir / "split_train_patients.csv", index=False)
    pd.DataFrame({"patient_id": sorted(p_val)}).to_csv(out_dir / "split_val_patients.csv", index=False)
    pd.DataFrame({"patient_id": sorted(p_test)}).to_csv(out_dir / "split_test_patients.csv", index=False)

    pd.DataFrame({"gene": hvg_genes}).to_csv(out_dir / "hvg_train_only_top2000.csv", index=False)
    pd.DataFrame({"gene": marker_genes}).to_csv(out_dir / "marker_gene_list.csv", index=False)
    pd.DataFrame({"rna_col": selected_cols, "gene": [c[4:] for c in selected_cols]}).to_csv(
        out_dir / "selected_targets_hvg_plus_markers.csv", index=False
    )

    reduced_df.to_csv(out_dir / "if_reference_hvg2000_plus_markers.csv", index=False)
    reduced_train.to_csv(out_dir / "if_reference_hvg2000_train_split.csv", index=False)
    reduced_val.to_csv(out_dir / "if_reference_hvg2000_val_split.csv", index=False)
    reduced_test.to_csv(out_dir / "if_reference_hvg2000_test_split.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_samples_total": len(df),
                "n_train_samples": len(train_df),
                "n_val_samples": len(val_df),
                "n_test_samples": len(test_df),
                "n_patients_total": len(patients),
                "n_train_patients": len(p_train),
                "n_val_patients": len(p_val),
                "n_test_patients": len(p_test),
                "n_rna_before": len(rna_cols),
                "n_hvg": len(hvg_cols),
                "n_marker_present": len(marker_cols),
                "n_selected_targets": len(selected_cols),
                "log1p_for_hvg": bool(args.log1p_for_hvg),
            }
        ]
    )
    summary.to_csv(out_dir / "hvg_build_summary.csv", index=False)

    print(f"Wrote HVG package to: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
