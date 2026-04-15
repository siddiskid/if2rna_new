#!/usr/bin/env python3
"""Run held-out-organ OOD evaluation with a mean-pooling Ridge baseline.

For each organ O:
1) Train on all samples where organ_type != O
2) Test on samples where organ_type == O
3) Report gene-wise correlation and error metrics
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge


def load_mean_feature(feature_dir: Path, organ: str, sample_id: str):
    h5_path = feature_dir / organ / sample_id / f"{sample_id}.h5"
    if not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if "cluster_features" not in f:
                return None
            feats = f["cluster_features"][:]
        if feats.shape[0] == 0:
            return None
        return feats.mean(axis=0).astype(np.float32)
    except Exception:
        return None


def build_xy(df: pd.DataFrame, feature_dir: Path, rna_cols):
    x_list, y_list, keep_rows = [], [], []
    for _, row in df.iterrows():
        sample_id = str(row["wsi_file_name"])
        organ = str(row["organ_type"])
        x = load_mean_feature(feature_dir, organ, sample_id)
        if x is None:
            continue
        y = row[rna_cols].to_numpy(dtype=np.float32)
        x_list.append(x)
        y_list.append(y)
        keep_rows.append(row)

    if len(x_list) == 0:
        return None, None, pd.DataFrame(columns=df.columns)

    return np.stack(x_list), np.stack(y_list), pd.DataFrame(keep_rows).reset_index(drop=True)


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
        r, _ = pearsonr(t[mask], p[mask])
        corr[i] = r
    return corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out-organ OOD with mean-pooling Ridge")
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--feature_dir", default="data/if_features")
    parser.add_argument("--output_dir", default="results/phase3_ood/heldout_organ_meanpool")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_transform", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    ref_path = Path(args.ref_file)
    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ref_path, low_memory=False)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if len(rna_cols) == 0:
        raise ValueError("No rna_* columns found")

    # Match main pipeline cleaning behavior.
    bad_genes = df[rna_cols].isna().sum(axis=0) > (len(df) * 0.5)
    if bad_genes.any():
        bad_gene_cols = [col for col, is_bad in zip(rna_cols, bad_genes) if is_bad]
        df = df.drop(columns=bad_gene_cols)
        rna_cols = [c for c in df.columns if c.startswith("rna_")]

    df[rna_cols] = df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if args.log_transform:
        df[rna_cols] = np.log1p(df[rna_cols])

    organs = sorted(df["organ_type"].dropna().unique().tolist())
    summary_rows = []

    for heldout in organs:
        train_df = df[df["organ_type"] != heldout].reset_index(drop=True)
        test_df = df[df["organ_type"] == heldout].reset_index(drop=True)

        x_train, y_train, train_kept = build_xy(train_df, feature_dir, rna_cols)
        x_test, y_test, test_kept = build_xy(test_df, feature_dir, rna_cols)

        if x_train is None or x_test is None:
            summary_rows.append({
                "heldout_organ": heldout,
                "status": "skipped_missing_features",
                "n_train_samples": 0 if x_train is None else int(len(train_kept)),
                "n_test_samples": 0 if x_test is None else int(len(test_kept)),
            })
            continue

        model = Ridge(alpha=args.alpha, random_state=args.seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).astype(np.float32)

        corr = gene_corrs(y_test, y_pred)
        corr_df = pd.DataFrame({"gene": rna_cols, "correlation": corr})
        corr_df.to_csv(out_dir / f"heldout_{heldout.lower()}_gene_correlations.csv", index=False)

        row = {
            "heldout_organ": heldout,
            "status": "ok",
            "n_train_samples": int(len(train_kept)),
            "n_test_samples": int(len(test_kept)),
            "n_genes": int(len(rna_cols)),
            "mean_gene_correlation": float(np.nanmean(corr)),
            "median_gene_correlation": float(np.nanmedian(corr)),
            "mae": float(np.mean(np.abs(y_pred - y_test))),
            "rmse": float(np.sqrt(np.mean((y_pred - y_test) ** 2))),
            "genes_r_gt_0_3": int(np.sum(corr > 0.3)),
            "genes_r_gt_0_5": int(np.sum(corr > 0.5)),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "heldout_organ_summary.csv", index=False)

    if (summary_df["status"] == "ok").any():
        ok_df = summary_df[summary_df["status"] == "ok"].copy()
        agg = {
            "ood_mean_of_means": float(ok_df["mean_gene_correlation"].mean()),
            "ood_mean_of_medians": float(ok_df["median_gene_correlation"].mean()),
            "ood_mean_mae": float(ok_df["mae"].mean()),
            "ood_mean_rmse": float(ok_df["rmse"].mean()),
        }
        pd.DataFrame([agg]).to_csv(out_dir / "heldout_organ_aggregate.csv", index=False)

    print(f"Saved OOD summary: {out_dir / 'heldout_organ_summary.csv'}")
    if len(summary_df) > 0:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
