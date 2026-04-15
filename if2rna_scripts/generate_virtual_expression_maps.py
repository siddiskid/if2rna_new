#!/usr/bin/env python3
"""Generate dense slide-level virtual expression maps from IF images.

For each selected slide, this script tiles ROI PNG images on a regular grid,
extracts per-tile ResNet features, predicts expression per tile with a trained
IF2RNA model, and writes per-gene heatmaps.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sequoia-pub"))
sys.path.insert(0, str(ROOT / "sequoia-pub" / "src"))
from src.resnet import resnet50  # type: ignore
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
    p = argparse.ArgumentParser(description="Generate dense virtual expression maps")
    p.add_argument("--ref_file", required=True)
    p.add_argument("--image_root", default="data/if_images")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_type", choices=["vis", "vit", "attention_mil"], default="attention_mil")
    p.add_argument("--resnet_weights", default="models/resnet50/resnet50_imagenet.pth")
    p.add_argument("--output_dir", default="results/final_package/virtual_maps")
    p.add_argument("--slide_names", nargs="*", default=None)
    p.add_argument("--max_slides", type=int, default=4)
    p.add_argument("--max_rois_per_slide", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--stride", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--top_k_genes", type=int, default=8)
    p.add_argument("--smooth_sigma", type=float, default=6.0)
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


def load_resnet(weights_path: str, device: torch.device):
    model = resnet50(pretrained=False)
    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"ResNet weights not found: {weights_path}")
    model.load_state_dict(torch.load(wp, map_location="cpu"))
    model = model.to(device)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, mean, std


def tile_image(img: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    h, w = img.shape[:2]
    tiles = []
    coords = []
    if h < patch_size or w < patch_size:
        return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            tile = img[y:y+patch_size, x:x+patch_size]
            if tile.mean() < 2.0:
                continue
            tiles.append(tile)
            coords.append((x, y))
    if not tiles:
        return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), []
    return np.stack(tiles, axis=0), coords


@torch.no_grad()
def extract_features(tiles: np.ndarray, resnet_model, mean, std, batch_size: int, device: torch.device) -> np.ndarray:
    feats = []
    for i in range(0, len(tiles), batch_size):
        b = tiles[i:i+batch_size]
        x = torch.from_numpy(b).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = (x - mean) / std
        f = resnet_model.forward_extract(x)
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def predict_tiles(model: nn.Module, tile_features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(tile_features).float().to(device).unsqueeze(1)  # [N, 1, D]
    p = model(x)
    return p.detach().cpu().numpy()


def choose_genes(ref_df: pd.DataFrame, n_out: int, top_k: int) -> List[int]:
    rna_cols = [c for c in ref_df.columns if c.startswith("rna_")]
    rna_cols = rna_cols[:n_out]
    var = ref_df[rna_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).var(axis=0).to_numpy()
    idx = np.argsort(var)[::-1][:max(1, top_k)]
    return [int(i) for i in idx]


def heatmap_from_points(
    coords: List[Tuple[int, int]],
    vals: np.ndarray,
    shape_hw: Tuple[int, int],
    patch_size: int,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    h, w = shape_hw
    sum_grid = np.zeros((h, w), dtype=np.float32)
    cnt_grid = np.zeros((h, w), dtype=np.float32)
    for (x, y), v in zip(coords, vals):
        sum_grid[y:y+patch_size, x:x+patch_size] += float(v)
        cnt_grid[y:y+patch_size, x:x+patch_size] += 1.0

    grid = np.full((h, w), np.nan, dtype=np.float32)
    valid = cnt_grid > 0
    if np.any(valid):
        grid[valid] = sum_grid[valid] / cnt_grid[valid]

    if smooth_sigma > 0 and np.any(valid):
        fill_val = float(np.nanmean(grid[valid]))
        dense = np.where(valid, grid, fill_val)
        smoothed = gaussian_filter(dense, sigma=smooth_sigma)
        grid[valid] = smoothed[valid]
    return grid


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(args.ref_file, low_memory=False)

    state = torch.load(args.checkpoint, map_location="cpu")
    out_dim = find_output_dim(state, args.model_type)
    model = build_model(args.model_type, out_dim, input_dim=2048, device=device)
    model.load_state_dict(state)
    model.eval()

    resnet_model, mean, std = load_resnet(args.resnet_weights, device)

    if args.slide_names:
        slides = list(args.slide_names)
    else:
        slides = sorted(ref_df["slide_name"].dropna().unique().tolist())[: args.max_slides]

    gene_ids = choose_genes(ref_df, out_dim, args.top_k_genes)
    rna_cols = [c for c in ref_df.columns if c.startswith("rna_")][:out_dim]
    gene_names = [rna_cols[i].replace("rna_", "") for i in gene_ids]

    summary_rows = []

    for slide in slides:
        s_df = ref_df[ref_df["slide_name"] == slide]
        if len(s_df) == 0:
            continue
        organ = str(s_df["organ_type"].mode().iloc[0])
        slide_dir = Path(args.image_root) / organ / slide
        if not slide_dir.exists():
            summary_rows.append({"slide_name": slide, "status": "missing_slide_dir"})
            continue

        imgs = sorted([p for p in slide_dir.glob("*.png")])[: args.max_rois_per_slide]
        if len(imgs) == 0:
            summary_rows.append({"slide_name": slide, "status": "no_roi_images"})
            continue

        slide_out = out_dir / f"slide_{slide}"
        slide_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            img = np.array(Image.open(img_path).convert("RGB"))
            tiles, coords = tile_image(img, args.patch_size, args.stride)
            if len(tiles) == 0:
                continue

            feats = extract_features(tiles, resnet_model, mean, std, args.batch_size, device)
            preds = predict_tiles(model, feats, device)

            roi_name = img_path.stem
            roi_out = slide_out / roi_name
            roi_out.mkdir(parents=True, exist_ok=True)

            np.save(roi_out / "tile_coords.npy", np.asarray(coords, dtype=np.int32))
            np.save(roi_out / "tile_preds.npy", preds.astype(np.float32))

            for gi, gname in zip(gene_ids, gene_names):
                hm = heatmap_from_points(
                    coords,
                    preds[:, gi],
                    img.shape[:2],
                    args.patch_size,
                    smooth_sigma=args.smooth_sigma,
                )
                np.save(roi_out / f"heatmap_{gname}.npy", hm.astype(np.float32))

                fig = plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.imshow(hm, cmap="inferno", alpha=0.55, interpolation="bilinear")
                plt.title(f"{slide} | {roi_name} | {gname}")
                plt.axis("off")
                fig.savefig(roi_out / f"heatmap_{gname}.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

            summary_rows.append(
                {
                    "slide_name": slide,
                    "roi_image": roi_name,
                    "status": "ok",
                    "n_tiles": int(len(tiles)),
                    "height": int(img.shape[0]),
                    "width": int(img.shape[1]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "virtual_map_summary.csv", index=False)
    pd.DataFrame({"gene_index": gene_ids, "gene_name": gene_names}).to_csv(out_dir / "selected_map_genes.csv", index=False)
    print(f"Saved virtual maps to: {out_dir}")


if __name__ == "__main__":
    main()
