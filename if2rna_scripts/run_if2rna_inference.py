#!/usr/bin/env python3
"""
Run IF2RNA inference on preprocessed features and evaluate against RNA.

This is designed for inference on new image cohorts (e.g., ROSIE-converted IF),
using a trained IF2RNA checkpoint.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sequoia-pub'))
sys.path.insert(0, str(ROOT / 'sequoia-pub' / 'src'))
from src.tformer_lin import ViS  # type: ignore[import]


def load_cluster_features(feature_h5: Path) -> Optional[np.ndarray]:
    if not feature_h5.exists():
        return None
    try:
        with h5py.File(feature_h5, "r") as f:
            if "cluster_features" not in f:
                return None
            return f["cluster_features"][:]
    except Exception:
        return None


def evaluate_correlations(pred_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(ref_df, on="wsi_file_name", how="inner", suffixes=("_pred", "_true"))

    rna_cols = [c for c in pred_df.columns if c.startswith("rna_")]
    out = []

    for gene_col in rna_cols:
        pred_col = f"{gene_col}_pred"
        true_col = f"{gene_col}_true"
        if pred_col not in merged.columns or true_col not in merged.columns:
            continue

        pred_vals = merged[pred_col].to_numpy(dtype=float)
        true_vals = merged[true_col].to_numpy(dtype=float)
        mask = ~(np.isnan(pred_vals) | np.isnan(true_vals))
        if mask.sum() < 2:
            continue

        p = pred_vals[mask]
        t = true_vals[mask]
        # Skip constant vectors to avoid undefined Pearson.
        if np.std(p) == 0 or np.std(t) == 0:
            continue

        r, pval = pearsonr(p, t)
        out.append({
            "gene": gene_col,
            "correlation": r,
            "p_value": pval,
            "n_samples": int(mask.sum()),
        })

    corr_df = pd.DataFrame(out)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values("correlation", ascending=False)
    return corr_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IF2RNA inference on feature HDF5 files")
    parser.add_argument("--reference_csv", required=True, help="Reference CSV with wsi_file_name, organ_type, rna_* columns")
    parser.add_argument("--feature_dir", required=True, help="Feature root directory containing <organ>/<wsi>/<wsi>.h5")
    parser.add_argument("--checkpoint", required=True, help="Path to trained IF2RNA checkpoint (.pt)")
    parser.add_argument("--output_predictions", required=True, help="Output predictions CSV")
    parser.add_argument("--output_correlations", required=False, help="Output per-gene correlations CSV")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth used during training")
    parser.add_argument("--num_heads", type=int, default=16, help="Transformer heads used during training")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_csv)
    rna_cols = [c for c in ref_df.columns if c.startswith("rna_")]
    if not rna_cols:
        raise ValueError("reference_csv contains no rna_* columns")

    first_feature = None
    for _, row in ref_df.iterrows():
        wsi = str(row["wsi_file_name"]).replace(".svs", "")
        organ = str(row.get("organ_type", ""))
        h5_path = Path(args.feature_dir) / organ / wsi / f"{wsi}.h5"
        first_feature = load_cluster_features(h5_path)
        if first_feature is not None:
            break
    if first_feature is None:
        raise RuntimeError("No cluster_features found in feature_dir for any sample")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = ViS(
        num_outputs=len(rna_cols),
        input_dim=first_feature.shape[1],
        depth=args.depth,
        nheads=args.num_heads,
        dimensions_f=64,
        dimensions_c=64,
        dimensions_s=64,
        device=device,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    pred_rows = []
    for _, row in ref_df.iterrows():
        wsi = str(row["wsi_file_name"]).replace(".svs", "")
        organ = str(row.get("organ_type", ""))
        h5_path = Path(args.feature_dir) / organ / wsi / f"{wsi}.h5"
        feats = load_cluster_features(h5_path)
        if feats is None:
            continue

        x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).cpu().numpy().squeeze()

        out_row = {
            "wsi_file_name": wsi,
            "patient_id": row.get("patient_id", ""),
            "organ_type": organ,
        }
        for gene_col, val in zip(rna_cols, y):
            out_row[gene_col] = float(val)
        pred_rows.append(out_row)

    pred_df = pd.DataFrame(pred_rows)
    Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_predictions, index=False)
    print(f"Saved predictions: {args.output_predictions}")
    print(f"Predicted samples: {len(pred_df)}")

    if args.output_correlations:
        corr_df = evaluate_correlations(pred_df, ref_df[["wsi_file_name", *rna_cols]])
        Path(args.output_correlations).parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(args.output_correlations, index=False)
        print(f"Saved correlations: {args.output_correlations}")
        if len(corr_df) > 0:
            print(f"Mean correlation: {corr_df['correlation'].mean():.4f}")
            print(f"Median correlation: {corr_df['correlation'].median():.4f}")


if __name__ == "__main__":
    main()
