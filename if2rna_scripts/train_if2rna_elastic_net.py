#!/usr/bin/env python3
"""Train handcrafted-feature Elastic Net baseline for IF2RNA.

This baseline builds a per-ROI handcrafted vector from patch features, then fits
MultiTaskElasticNetCV on RNA targets with patient-level CV.
"""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import KFold, train_test_split


def patient_kfold(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42, valid_size: float = 0.1):
    indices = np.arange(len(df))
    patients_unique = df["patient_id"].unique()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_idx = []
    valid_idx = []
    test_idx = []

    for _, (ind_train, ind_test) in enumerate(skf.split(patients_unique)):
        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_mask = df["patient_id"].isin(patients_test)
        test_idx.append(indices[test_mask])

        patients_train, patients_valid = train_test_split(
            patients_train, test_size=valid_size, random_state=random_state
        )
        valid_mask = df["patient_id"].isin(patients_valid)
        valid_idx.append(indices[valid_mask])

        train_mask = df["patient_id"].isin(patients_train)
        train_idx.append(indices[train_mask])

    return train_idx, valid_idx, test_idx


def handcrafted_from_patch_features(feats: np.ndarray) -> np.ndarray:
    """Build a compact handcrafted vector from patch-level embeddings."""
    # Stats over patches per feature dimension.
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    q25 = np.quantile(feats, 0.25, axis=0)
    q75 = np.quantile(feats, 0.75, axis=0)
    # Global bag-level distribution descriptors.
    bag_size = np.array([feats.shape[0]], dtype=np.float32)
    global_mean = np.array([feats.mean()], dtype=np.float32)
    global_std = np.array([feats.std()], dtype=np.float32)
    out = np.concatenate([mean, std, q25, q75, bag_size, global_mean, global_std], axis=0)
    return out.astype(np.float32)


def load_handcrafted_feature(feature_dir: Path, organ: str, sample_id: str, feature_key: str):
    h5_path = feature_dir / organ / sample_id / f"{sample_id}.h5"
    if not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if feature_key not in f:
                return None
            feats = f[feature_key][:]
        if feats.shape[0] == 0:
            return None
        return handcrafted_from_patch_features(feats)
    except Exception:
        return None


def genewise_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    corrs = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    for i in range(y_true.shape[1]):
        t = y_true[:, i]
        p = y_pred[:, i]
        mask = ~(np.isnan(t) | np.isnan(p))
        if mask.sum() < 2:
            continue
        if np.std(t[mask]) == 0 or np.std(p[mask]) == 0:
            continue
        r, _ = pearsonr(t[mask], p[mask])
        corrs[i] = r
    return corrs


def build_xy(df: pd.DataFrame, feature_dir: Path, rna_cols, feature_key: str):
    x_list, y_list, kept_rows = [], [], []
    for _, row in df.iterrows():
        sample_id = str(row["wsi_file_name"])
        organ = str(row["organ_type"])
        x = load_handcrafted_feature(feature_dir, organ, sample_id, feature_key)
        if x is None:
            continue
        y = row[rna_cols].to_numpy(dtype=np.float32)
        x_list.append(x)
        y_list.append(y)
        kept_rows.append(row)

    if len(x_list) == 0:
        return None, None, pd.DataFrame(columns=df.columns)

    return np.stack(x_list), np.stack(y_list), pd.DataFrame(kept_rows).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train handcrafted Elastic Net baseline")
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--feature_dir", default="data/if_features")
    parser.add_argument("--feature_key", default="cluster_features", choices=["cluster_features", "resnet_features"])
    parser.add_argument("--save_dir", default="results/if2rna_models")
    parser.add_argument("--exp_name", default="phase2_elastic_net_baseline")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_transform", action="store_true")
    parser.add_argument("--n_alphas", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--cv_inner", type=int, default=5)
    parser.add_argument("--l1_ratio_grid", default="0.1,0.3,0.5,0.7,0.9")
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.save_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ref_file, low_memory=False)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if not rna_cols:
        raise ValueError("No rna_* columns found")

    bad_genes = df[rna_cols].isna().sum(axis=0) > (len(df) * 0.5)
    if bad_genes.any():
        bad_gene_cols = [col for col, is_bad in zip(rna_cols, bad_genes) if is_bad]
        df = df.drop(columns=bad_gene_cols)
        rna_cols = [c for c in df.columns if c.startswith("rna_")]

    df[rna_cols] = df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if args.log_transform:
        df[rna_cols] = np.log1p(df[rna_cols])

    l1_ratio_grid = [float(x.strip()) for x in str(args.l1_ratio_grid).split(",") if x.strip()]
    if not l1_ratio_grid:
        raise ValueError("l1_ratio_grid must contain at least one value")

    train_idxs, _, test_idxs = patient_kfold(df, n_splits=args.k, random_state=args.seed)

    all_test_real = []
    all_test_pred = []
    fold_metrics = []
    test_results = {}

    feature_dir = Path(args.feature_dir)

    for fold in range(args.k):
        train_df = df.iloc[train_idxs[fold]].reset_index(drop=True)
        test_df = df.iloc[test_idxs[fold]].reset_index(drop=True)

        X_train, y_train, train_kept = build_xy(train_df, feature_dir, rna_cols, args.feature_key)
        X_test, y_test, test_kept = build_xy(test_df, feature_dir, rna_cols, args.feature_key)

        if X_train is None or X_test is None:
            continue

        model = MultiTaskElasticNetCV(
            l1_ratio=l1_ratio_grid,
            n_alphas=args.n_alphas,
            cv=args.cv_inner,
            random_state=args.seed,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).astype(np.float32)

        corr_vec = genewise_corr(y_test, y_pred)
        fold_metrics.append({
            "fold": fold,
            "train_samples_used": int(len(train_kept)),
            "test_samples_used": int(len(test_kept)),
            "mean_gene_pearson_r": float(np.nanmean(corr_vec)),
            "median_gene_pearson_r": float(np.nanmedian(corr_vec)),
            "mae": float(np.mean(np.abs(y_pred - y_test))),
            "rmse": float(np.sqrt(np.mean((y_pred - y_test) ** 2))),
            "alpha": float(model.alpha_),
            "l1_ratio": float(model.l1_ratio_),
        })

        test_results[f"split_{fold}"] = {
            "real": y_test,
            "preds": y_pred,
            "sample_ids": test_kept["wsi_file_name"].tolist(),
            "wsi_file_name": test_kept["wsi_file_name"].to_numpy(),
            "patient_id": test_kept["patient_id"].to_numpy(),
            "organ_type": test_kept["organ_type"].to_numpy(),
        }

        all_test_real.append(y_test)
        all_test_pred.append(y_pred)

    if len(all_test_real) == 0:
        raise RuntimeError("No valid folds completed for elastic-net baseline")

    all_test_real = np.concatenate(all_test_real, axis=0)
    all_test_pred = np.concatenate(all_test_pred, axis=0)
    all_corr = genewise_corr(all_test_real, all_test_pred)

    summary = {
        "n_folds_completed": len(fold_metrics),
        "n_samples_total": int(all_test_real.shape[0]),
        "n_genes": int(all_test_real.shape[1]),
        "mean_gene_pearson_r": float(np.nanmean(all_corr)),
        "median_gene_pearson_r": float(np.nanmedian(all_corr)),
        "mae": float(np.mean(np.abs(all_test_pred - all_test_real))),
        "rmse": float(np.sqrt(np.mean((all_test_pred - all_test_real) ** 2))),
        "genes_r_gt_0_3": int(np.sum(all_corr > 0.3)),
        "genes_r_gt_0_5": int(np.sum(all_corr > 0.5)),
        "feature_key": args.feature_key,
    }

    pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame({"gene": rna_cols, "correlation": all_corr}).to_csv(out_dir / "gene_correlations.csv", index=False)

    test_results["genes"] = [g[4:] if g.startswith("rna_") else g for g in rna_cols]
    with open(out_dir / "test_results.pkl", "wb") as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved baseline outputs to:", out_dir)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
