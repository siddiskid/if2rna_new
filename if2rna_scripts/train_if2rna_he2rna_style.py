#!/usr/bin/env python3
"""Train HE2RNA-style top-k pooling baseline on IF features.

This adapts the HE2RNA idea (tile-level scoring + top-k aggregation) to the
current IF2RNA feature layout under data/if_features/{organ}/{sample}/{sample}.h5.
"""

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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset


class HE2RNAStyle(nn.Module):
    """Conv1d-on-instances + top-k pooling across bag instances."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, ks=None, dropout: float = 0.2):
        super().__init__()
        if ks is None:
            ks = [5, 10, 20]
        self.ks = np.array(sorted(set(int(k) for k in ks if int(k) > 0)))
        if len(self.ks) == 0:
            raise ValueError("ks must contain at least one positive integer")

        self.layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, bias=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=True),
        ])
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _conv_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> [B, D, N]
        h = x.permute(0, 2, 1)
        h = self.dropout(self.nonlin(self.layers[0](h)))
        h = self.layers[1](h)  # [B, G, N]
        return h

    def _forward_fixed_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        scores = self._conv_scores(x)  # [B, G, N]
        n_tiles = scores.shape[-1]
        k_eff = min(max(1, int(k)), n_tiles)
        topk_vals, _ = torch.topk(scores, k=k_eff, dim=2, largest=True, sorted=False)
        return topk_vals.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            k = int(np.random.choice(self.ks))
            return self._forward_fixed_k(x, k)
        preds = 0.0
        for k in self.ks:
            preds = preds + self._forward_fixed_k(x, int(k))
        return preds / float(len(self.ks))


class IFRNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_dir: str, rna_cols, feature_key: str = "cluster_features"):
        self.df = df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.rna_cols = rna_cols
        self.feature_key = feature_key

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
                x = torch.tensor(f[self.feature_key][:], dtype=torch.float32)
        except Exception:
            return None, None, sample_id, organ

        y = torch.tensor(row[self.rna_cols].values.astype(np.float32), dtype=torch.float32)
        return x, y, sample_id, organ


def collate_skip_missing(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return [], [], [], []
    return torch.utils.data.dataloader.default_collate(batch)


def patient_kfold(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42, valid_size: float = 0.1):
    indices = np.arange(len(df))
    patients_unique = df["patient_id"].unique()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_idx, valid_idx, test_idx = [], [], []
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


def run_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    for x, y, _, _ in loader:
        if len(x) == 0:
            continue
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    preds, reals, sample_ids, organs = [], [], [], []
    for x, y, ids, org in loader:
        if len(x) == 0:
            continue
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(float(loss.detach().cpu().item()))

        preds.append(pred.detach().cpu().numpy())
        reals.append(y.detach().cpu().numpy())
        sample_ids.extend(list(ids))
        organs.extend(list(org))

    if len(preds) == 0:
        return np.nan, None, None, [], []
    preds = np.concatenate(preds, axis=0)
    reals = np.concatenate(reals, axis=0)
    return float(np.mean(losses)), preds, reals, sample_ids, organs


def main() -> None:
    p = argparse.ArgumentParser(description="Train IF2RNA HE2RNA-style pooling baseline")
    p.add_argument("--ref_file", required=True)
    p.add_argument("--feature_dir", default="data/if_features")
    p.add_argument("--feature_key", default="cluster_features", choices=["cluster_features", "resnet_features"])
    p.add_argument("--save_dir", default="results/if2rna_models")
    p.add_argument("--exp_name", default="phase2_he2rna_style_baseline")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_transform", action="store_true")
    p.add_argument("--num_epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.save_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ref_file, low_memory=False)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if len(rna_cols) == 0:
        raise ValueError("No rna_* columns found")

    bad_genes = df[rna_cols].isna().sum(axis=0) > (len(df) * 0.5)
    if bad_genes.any():
        bad_gene_cols = [col for col, is_bad in zip(rna_cols, bad_genes) if is_bad]
        df = df.drop(columns=bad_gene_cols)
        rna_cols = [c for c in df.columns if c.startswith("rna_")]

    df[rna_cols] = df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if args.log_transform:
        df[rna_cols] = np.log1p(df[rna_cols])

    tr_idx, va_idx, te_idx = patient_kfold(df, n_splits=args.k, random_state=args.seed)

    all_real, all_pred = [], []
    fold_rows = []
    test_results = {"genes": [g[4:] if g.startswith("rna_") else g for g in rna_cols]}

    for fold in range(args.k):
        train_df = df.iloc[tr_idx[fold]].reset_index(drop=True)
        val_df = df.iloc[va_idx[fold]].reset_index(drop=True)
        test_df = df.iloc[te_idx[fold]].reset_index(drop=True)

        train_ds = IFRNADataset(train_df, args.feature_dir, rna_cols, feature_key=args.feature_key)
        val_ds = IFRNADataset(val_df, args.feature_dir, rna_cols, feature_key=args.feature_key)
        test_ds = IFRNADataset(test_df, args.feature_dir, rna_cols, feature_key=args.feature_key)

        tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_skip_missing)
        va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_missing)
        te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_missing)

        # Infer feature dimension from first valid item.
        feat_dim = None
        for i in range(len(train_ds)):
            x, _, _, _ = train_ds[i]
            if x is not None:
                feat_dim = int(x.shape[1])
                break
        if feat_dim is None:
            raise RuntimeError(f"No valid features found for fold {fold}")

        model = HE2RNAStyle(
            input_dim=feat_dim,
            output_dim=len(rna_cols),
            hidden_dim=args.hidden_dim,
            ks=args.ks,
            dropout=args.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val = np.inf
        best_state = None
        no_improve = 0
        for _epoch in range(args.num_epochs):
            _ = run_epoch(model, tr_loader, opt, device)
            val_loss, _, _, _, _ = eval_epoch(model, va_loader, device)
            if np.isnan(val_loss):
                continue
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, preds, real, ids, organs = eval_epoch(model, te_loader, device)
        if preds is None:
            continue

        all_real.append(real)
        all_pred.append(preds)

        c = gene_corrs(real, preds)
        fold_rows.append(
            {
                "fold": fold,
                "n_test_samples": int(real.shape[0]),
                "n_genes": int(real.shape[1]),
                "val_loss_best": float(best_val),
                "test_loss": float(test_loss),
                "mean_gene_pearson_r": float(np.nanmean(c)),
                "median_gene_pearson_r": float(np.nanmedian(c)),
                "mae": float(np.mean(np.abs(preds - real))),
                "rmse": float(np.sqrt(np.mean((preds - real) ** 2))),
            }
        )

        test_results[f"split_{fold}"] = {
            "real": real,
            "preds": preds,
            "sample_ids": ids,
            "organ_type": np.array(organs),
        }

    if len(all_real) == 0:
        raise RuntimeError("No valid folds completed")

    all_real = np.concatenate(all_real, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_corr = gene_corrs(all_real, all_pred)

    summary = {
        "n_folds_completed": len(fold_rows),
        "n_samples_total": int(all_real.shape[0]),
        "n_genes": int(all_real.shape[1]),
        "mean_gene_pearson_r": float(np.nanmean(all_corr)),
        "median_gene_pearson_r": float(np.nanmedian(all_corr)),
        "mae": float(np.mean(np.abs(all_pred - all_real))),
        "rmse": float(np.sqrt(np.mean((all_pred - all_real) ** 2))),
        "genes_r_gt_0_3": int(np.sum(all_corr > 0.3)),
        "genes_r_gt_0_5": int(np.sum(all_corr > 0.5)),
        "feature_key": args.feature_key,
        "ks": "|".join(str(x) for x in args.ks),
    }

    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame({"gene": [g[4:] if g.startswith("rna_") else g for g in rna_cols], "correlation": all_corr}).to_csv(
        out_dir / "gene_correlations.csv", index=False
    )
    with open(out_dir / "test_results.pkl", "wb") as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved HE2RNA-style outputs to:", out_dir)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
