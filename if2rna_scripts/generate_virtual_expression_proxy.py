#!/usr/bin/env python3
"""Generate per-patch virtual-expression proxy scores from bag models.

Because cached IF features currently provide bag-level patch embeddings without x/y
coordinates, this script computes a patch contribution proxy by leave-one-out (LOO)
perturbation for selected genes.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def load_features(feature_dir: Path, organ: str, sample_id: str, feature_key: str):
    h5_path = feature_dir / organ / sample_id / f"{sample_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if feature_key not in f:
            raise KeyError(f"Missing key '{feature_key}' in {h5_path}")
        feats = f[feature_key][:]
    return feats.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual-expression proxy by leave-one-out patches")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--sample_id", required=True, help="wsi_file_name in reference CSV")
    parser.add_argument("--model_type", choices=["vis", "vit", "attention_mil"], default="vis")
    parser.add_argument("--feature_key", default="cluster_features", choices=["cluster_features", "resnet_features"])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--top_k_genes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_csv, low_memory=False)
    row = ref_df[ref_df["wsi_file_name"] == args.sample_id]
    if row.empty:
        raise ValueError(f"sample_id not found: {args.sample_id}")
    row = row.iloc[0]

    rna_cols = [c for c in ref_df.columns if c.startswith("rna_")]
    genes = [g[4:] if g.startswith("rna_") else g for g in rna_cols]

    organ = str(row["organ_type"])
    feats = load_features(Path(args.feature_dir), organ, args.sample_id, args.feature_key)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if args.model_type == "vis":
        model = ViS(
            num_outputs=len(rna_cols),
            input_dim=feats.shape[1],
            depth=args.depth,
            nheads=args.num_heads,
            dimensions_f=64,
            dimensions_c=64,
            dimensions_s=64,
            device=device,
        )
    elif args.model_type == "vit":
        model = ViT(
            num_outputs=len(rna_cols),
            dim=feats.shape[1],
            depth=args.depth,
            heads=args.num_heads,
            mlp_dim=2048,
            dim_head=64,
            device=device,
        )
    else:
        model = AttentionMIL(num_outputs=len(rna_cols), input_dim=feats.shape[1], device=device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    x_full = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        y_full = model(x_full).cpu().numpy().squeeze(0)

    top_idx = np.argsort(np.abs(y_full))[::-1][: args.top_k_genes]

    rows = []
    for patch_idx in range(feats.shape[0]):
        x_loo = feats.copy()
        x_loo[patch_idx] = 0.0
        x_loo = torch.from_numpy(x_loo).unsqueeze(0).to(device)
        with torch.no_grad():
            y_loo = model(x_loo).cpu().numpy().squeeze(0)
        delta = y_full - y_loo

        out = {
            "sample_id": args.sample_id,
            "organ_type": organ,
            "patch_index": patch_idx,
        }
        for gi in top_idx:
            out[f"delta_{genes[gi]}"] = float(delta[gi])
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
