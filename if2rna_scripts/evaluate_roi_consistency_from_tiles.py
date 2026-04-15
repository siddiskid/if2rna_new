#!/usr/bin/env python3
"""Evaluate ROI-consistency using tile-level predictions from IF feature bags.

For each ROI sample, predict expression per tile (bag instance), aggregate across
tiles (mean), and compare to measured ROI expression.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sequoia-pub"))
sys.path.insert(0, str(ROOT / "sequoia-pub" / "src"))
from src.tformer_lin import ViS  # type: ignore
from src.vit import ViT  # type: ignore


class AttentionMIL(nn.Module):
    def __init__(self, num_outputs, input_dim, hidden_dim=512, attn_dim=256, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_outputs))

    def forward(self, x):
        h = self.encoder(x)
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)
        a = torch.softmax(a, dim=1)
        z = torch.sum(h * a.unsqueeze(-1), dim=1)
        return self.decoder(z)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI consistency from tile predictions")
    p.add_argument("--ref_file", required=True)
    p.add_argument("--feature_dir", default="data/if_features")
    p.add_argument("--feature_key", default="resnet_features", choices=["resnet_features", "cluster_features"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_type", choices=["vis", "vit", "attention_mil"], default="attention_mil")
    p.add_argument("--output_dir", default="results/final_package/virtual_maps")
    p.add_argument("--log_transform", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def find_output_dim(state_dict: Dict[str, torch.Tensor], model_type: str) -> int:
    if model_type in ["vis", "vit"]:
        for k in ["linear_head.1.weight", "mlp_head.1.weight"]:
            if k in state_dict:
                return int(state_dict[k].shape[0])
    if model_type == "attention_mil":
        if "decoder.1.weight" in state_dict:
            return int(state_dict["decoder.1.weight"].shape[0])
    raise RuntimeError("Could not infer output dim from checkpoint")


def build_model(model_type: str, num_outputs: int, input_dim: int, device: torch.device) -> nn.Module:
    if model_type == "vis":
        model = ViS(
            num_outputs=num_outputs,
            input_dim=input_dim,
            depth=6,
            nheads=16,
            dimensions_f=64,
            dimensions_c=64,
            dimensions_s=64,
            device=device,
        )
    elif model_type == "vit":
        model = ViT(
            num_outputs=num_outputs,
            dim=input_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            device=device,
        )
    else:
        model = AttentionMIL(num_outputs=num_outputs, input_dim=input_dim, device=device)
    return model.to(device)


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
        corr[i] = pearsonr(t[mask], p[mask])[0]
    return corr


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ref_file, low_memory=False)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if len(rna_cols) == 0:
        raise ValueError("No rna_* columns found")

    df[rna_cols] = df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if args.log_transform:
        df[rna_cols] = np.log1p(df[rna_cols])

    if args.max_samples is not None and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    state = torch.load(args.checkpoint, map_location="cpu")
    out_dim = find_output_dim(state, args.model_type)
    rna_cols = rna_cols[:out_dim]

    # Infer input dim from first available feature file.
    input_dim = None
    for _, row in df.iterrows():
        sid = str(row["wsi_file_name"])
        org = str(row["organ_type"])
        p = Path(args.feature_dir) / org / sid / f"{sid}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            if args.feature_key in f:
                input_dim = int(f[args.feature_key].shape[1])
                break
    if input_dim is None:
        raise RuntimeError("Could not infer input dimension from feature files")

    model = build_model(args.model_type, out_dim, input_dim, device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            sid = str(row["wsi_file_name"])
            org = str(row["organ_type"])
            p = Path(args.feature_dir) / org / sid / f"{sid}.h5"
            if not p.exists():
                continue
            try:
                with h5py.File(p, "r") as f:
                    if args.feature_key not in f:
                        continue
                    feats = f[args.feature_key][:]
                if feats.shape[0] == 0:
                    continue

                x = torch.from_numpy(feats.astype(np.float32)).to(device).unsqueeze(1)  # [N,1,D]
                tile_preds = model(x).detach().cpu().numpy()  # [N,G]
                roi_pred = tile_preds.mean(axis=0)
                roi_true = row[rna_cols].to_numpy(dtype=np.float32)

                y_true_all.append(roi_true)
                y_pred_all.append(roi_pred)

                rows.append({
                    "wsi_file_name": sid,
                    "slide_name": str(row.get("slide_name", "")),
                    "organ_type": org,
                    "n_tiles": int(feats.shape[0]),
                    "mae": float(np.mean(np.abs(roi_pred - roi_true))),
                    "rmse": float(np.sqrt(np.mean((roi_pred - roi_true) ** 2))),
                })
            except Exception:
                continue

    if len(y_true_all) == 0:
        raise RuntimeError("No valid samples processed for ROI-consistency")

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    corr = gene_corrs(y_true, y_pred)

    gene_df = pd.DataFrame({"gene": [c.replace("rna_", "") for c in rna_cols], "correlation": corr})
    gene_df.to_csv(out_dir / "roi_consistency_gene_correlations.csv", index=False)

    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(out_dir / "roi_consistency_sample_metrics.csv", index=False)

    organ_df = (
        sample_df.groupby("organ_type", as_index=False)
        .agg(n_samples=("wsi_file_name", "count"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_tiles=("n_tiles", "mean"))
    )
    organ_df.to_csv(out_dir / "roi_consistency_per_organ.csv", index=False)

    summary = pd.DataFrame([
        {
            "n_samples": int(y_true.shape[0]),
            "n_genes": int(y_true.shape[1]),
            "mean_gene_correlation": float(np.nanmean(corr)),
            "median_gene_correlation": float(np.nanmedian(corr)),
            "roi_mean_mae": float(np.mean(np.abs(y_pred - y_true))),
            "roi_mean_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            "feature_key": args.feature_key,
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
        }
    ])
    summary.to_csv(out_dir / "roi_consistency_summary.csv", index=False)
    print(f"Saved ROI-consistency outputs to: {out_dir}")


if __name__ == "__main__":
    main()
