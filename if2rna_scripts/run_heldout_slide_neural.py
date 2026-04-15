#!/usr/bin/env python3
"""Held-out-slide OOD evaluation for IF2RNA neural models (ViS/ViT/Attention-MIL)."""

import argparse
import os
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sequoia-pub"))
sys.path.insert(0, str(ROOT / "sequoia-pub" / "src"))
from src.tformer_lin import ViS  # type: ignore
from src.vit import ViT, train, evaluate  # type: ignore


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


class IFRNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_dir: str, rna_cols, feature_key='cluster_features', bag_cap=None):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.rna_cols = rna_cols
        self.feature_key = feature_key
        self.bag_cap = bag_cap

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["wsi_file_name"])
        organ = str(row["organ_type"])
        h5_path = os.path.join(self.feature_dir, organ, sample_id, f"{sample_id}.h5")

        try:
            with h5py.File(h5_path, "r") as f:
                if self.feature_key not in f:
                    return None, None, sample_id, organ
                features = f[self.feature_key][:]
                if self.bag_cap is not None and self.bag_cap > 0 and features.shape[0] > self.bag_cap:
                    sel = np.random.choice(features.shape[0], size=self.bag_cap, replace=False)
                    features = features[sel]
                features = torch.tensor(features, dtype=torch.float32)
        except Exception:
            return None, None, sample_id, organ

        rna = torch.tensor(row[self.rna_cols].values.astype(np.float32), dtype=torch.float32)
        return features, rna, sample_id, organ


def collate_skip_missing(batch):
    batch = [x for x in batch if x[0] is not None]
    if len(batch) == 0:
        return [], [], [], []
    return torch.utils.data.dataloader.default_collate(batch)


def split_train_val_by_patient(df: pd.DataFrame, valid_size: float, seed: int):
    patients = df["patient_id"].unique()
    p_train, p_val = train_test_split(patients, test_size=valid_size, random_state=seed)
    train_df = df[df["patient_id"].isin(p_train)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(p_val)].reset_index(drop=True)
    return train_df, val_df


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


def infer_feature_dim(df: pd.DataFrame, feature_dir: str, feature_key: str):
    for _, row in df.iterrows():
        sample_id = str(row["wsi_file_name"])
        organ = str(row["organ_type"])
        h5_path = Path(feature_dir) / organ / sample_id / f"{sample_id}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                if feature_key in f:
                    return int(f[feature_key].shape[1])
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out-slide OOD for ViS/ViT/Attention-MIL")
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--feature_dir", default="data/if_features")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", choices=["vis", "vit", "attention_mil"], default="vis")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_transform", action="store_true")
    parser.add_argument("--heldout_slides", nargs="*", default=None)
    parser.add_argument("--min_test_samples", type=int, default=10)
    parser.add_argument("--feature_key", default="cluster_features", choices=["cluster_features", "resnet_features"])
    parser.add_argument("--bag_cap", type=int, default=None)
    parser.add_argument("--mil_hidden_dim", type=int, default=512)
    parser.add_argument("--mil_attn_dim", type=int, default=256)
    parser.add_argument("--mil_dropout", type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ref_file, low_memory=False)
    if "slide_name" not in df.columns:
        raise ValueError("reference CSV missing slide_name column")
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

    slides = sorted(df["slide_name"].dropna().unique().tolist())
    if args.heldout_slides:
        slides = [s for s in slides if s in args.heldout_slides]

    feature_dim = infer_feature_dim(df, args.feature_dir, args.feature_key)
    if feature_dim is None:
        raise RuntimeError("Could not infer feature dimension")

    summary_rows = []
    all_results = {"genes": [g[4:] if g.startswith("rna_") else g for g in rna_cols]}

    for heldout in slides:
        train_pool = df[df["slide_name"] != heldout].reset_index(drop=True)
        test_df = df[df["slide_name"] == heldout].reset_index(drop=True)

        if len(test_df) < args.min_test_samples:
            summary_rows.append({"heldout_slide": heldout, "status": "skipped_small_test", "n_test_samples": int(len(test_df))})
            continue
        if len(train_pool) == 0 or len(test_df) == 0:
            summary_rows.append({"heldout_slide": heldout, "status": "skipped_empty_split"})
            continue

        train_df, val_df = split_train_val_by_patient(train_pool, valid_size=args.valid_size, seed=args.seed)

        train_ds = IFRNADataset(train_df, args.feature_dir, rna_cols, feature_key=args.feature_key, bag_cap=args.bag_cap)
        val_ds = IFRNADataset(val_df, args.feature_dir, rna_cols, feature_key=args.feature_key, bag_cap=args.bag_cap)
        test_ds = IFRNADataset(test_df, args.feature_dir, rna_cols, feature_key=args.feature_key, bag_cap=args.bag_cap)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_skip_missing)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_missing)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_missing)

        if args.model_type == "vis":
            model = ViS(
                num_outputs=len(rna_cols),
                input_dim=feature_dim,
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
                dim=feature_dim,
                depth=args.depth,
                heads=args.num_heads,
                mlp_dim=2048,
                dim_head=64,
                device=device,
            )
        else:
            model = AttentionMIL(
                num_outputs=len(rna_cols),
                input_dim=feature_dim,
                hidden_dim=args.mil_hidden_dim,
                attn_dim=args.mil_attn_dim,
                dropout=args.mil_dropout,
                device=device,
            )

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=False, weight_decay=0.0)

        slide_out = out_root / f"heldout_{str(heldout).replace(' ', '_').lower()}"
        slide_out.mkdir(parents=True, exist_ok=True)

        dataloaders = {"train": train_loader, "val": val_loader}
        model = train(
            model,
            dataloaders,
            optimizer,
            num_epochs=args.num_epochs,
            save_dir=str(slide_out),
            patience=20,
            phases=["train", "val"],
            split=None,
            save_on="loss",
            stop_on="loss",
            delta=0.5,
        )

        best_path = slide_out / "model_best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

        preds, real, wsis, projs = evaluate(model, test_loader, run=None, verbose=True, suff="")
        corr = gene_corrs(real, preds)

        summary_rows.append({
            "heldout_slide": heldout,
            "status": "ok",
            "model_type": args.model_type,
            "n_train_samples": int(len(train_df)),
            "n_val_samples": int(len(val_df)),
            "n_test_samples": int(len(test_df)),
            "n_genes": int(len(rna_cols)),
            "mean_gene_correlation": float(np.nanmean(corr)),
            "median_gene_correlation": float(np.nanmedian(corr)),
            "mae": float(np.mean(np.abs(preds - real))),
            "rmse": float(np.sqrt(np.mean((preds - real) ** 2))),
            "genes_r_gt_0_3": int(np.sum(corr > 0.3)),
            "genes_r_gt_0_5": int(np.sum(corr > 0.5)),
        })

        all_results[f"heldout_{str(heldout).replace(' ', '_').lower()}"] = {
            "preds": preds,
            "real": real,
            "wsi_file_name": wsis,
            "organ_type": projs,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "heldout_slide_summary.csv", index=False)

    if len(summary_df) and (summary_df["status"] == "ok").any():
        ok_df = summary_df[summary_df["status"] == "ok"]
        pd.DataFrame([
            {
                "model_type": args.model_type,
                "ood_type": "heldout_slide",
                "ood_mean_of_means": float(ok_df["mean_gene_correlation"].mean()),
                "ood_mean_of_medians": float(ok_df["median_gene_correlation"].mean()),
                "ood_mean_mae": float(ok_df["mae"].mean()),
                "ood_mean_rmse": float(ok_df["rmse"].mean()),
            }
        ]).to_csv(out_root / "heldout_slide_aggregate.csv", index=False)

    with open(out_root / "heldout_slide_predictions.pkl", "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {out_root / 'heldout_slide_summary.csv'}")
    if len(summary_df):
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
