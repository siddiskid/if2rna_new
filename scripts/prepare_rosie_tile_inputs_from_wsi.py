#!/usr/bin/env python3
"""Prepare ROSIE inputs as tissue-rich tiles sampled at a target mpp.

This avoids whole-slide preview conversion and preserves local morphology.
Output tile names follow:
  <slide_id>__tile_<idx>.png
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from openslide import OpenSlide
from PIL import Image


def normalize_slide_name(name: str) -> str:
    name = str(name)
    for suffix in (".svs", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_native_mpp(slide: OpenSlide) -> float:
    props = slide.properties
    for key in ("openslide.mpp-x", "aperio.MPP"):
        value = props.get(key)
        if value is not None:
            try:
                v = float(value)
                if v > 0:
                    return v
            except ValueError:
                pass

    obj_power = props.get("openslide.objective-power") or props.get("aperio.AppMag")
    if obj_power is not None:
        try:
            p = float(obj_power)
            if p >= 40:
                return 0.25
            if p >= 20:
                return 0.5
            if p >= 10:
                return 1.0
        except ValueError:
            pass

    return 0.5


def choose_level(slide: OpenSlide, target_mpp: float) -> Tuple[int, float]:
    native_mpp = infer_native_mpp(slide)
    target_downsample = max(1.0, target_mpp / native_mpp)
    downsamples = [float(d) for d in slide.level_downsamples]
    level = min(range(len(downsamples)), key=lambda i: abs(downsamples[i] - target_downsample))
    return level, native_mpp * downsamples[level]


def tissue_mask_from_level(slide: OpenSlide, level: int) -> np.ndarray:
    dims = slide.level_dimensions[level]
    rgb = np.array(slide.read_region((0, 0), level, dims).convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Keep non-background regions; robust for H&E tissue/background separation.
    tissue = ((sat > 20) & (val < 245)) | (val < 220)
    return tissue.astype(np.uint8)


def score_tile(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    h_max, w_max = mask.shape
    x2 = min(w_max, x + w)
    y2 = min(h_max, y + h)
    if x >= w_max or y >= h_max or x2 <= x or y2 <= y:
        return 0.0
    patch = mask[y:y2, x:x2]
    return float(patch.mean())


def select_tile_coords(
    slide: OpenSlide,
    extract_level: int,
    score_level: int,
    tile_size: int,
    max_tiles: int,
    min_tissue_ratio: float,
) -> List[Tuple[int, int]]:
    mask = tissue_mask_from_level(slide, score_level)
    ds_extract = float(slide.level_downsamples[extract_level])
    ds_score = float(slide.level_downsamples[score_level])

    w0, h0 = slide.level_dimensions[0]
    step0 = int(tile_size * ds_extract)
    if step0 <= 0:
        step0 = 1

    candidates = []
    for y0 in range(0, max(1, h0 - step0), step0):
        for x0 in range(0, max(1, w0 - step0), step0):
            x_score = int(x0 / ds_score)
            y_score = int(y0 / ds_score)
            wh_score = max(1, int((tile_size * ds_extract) / ds_score))
            ratio = score_tile(mask, x_score, y_score, wh_score, wh_score)
            if ratio >= min_tissue_ratio:
                candidates.append((ratio, x0, y0))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    top = candidates[:max_tiles]
    return [(x0, y0) for _, x0, y0 in top]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ROSIE tile PNG inputs from WSI")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--max_tiles_per_slide", type=int, default=64)
    parser.add_argument("--min_tissue_ratio", type=float, default=0.2)
    parser.add_argument("--score_level_offset", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_dir = Path(args.wsi_dir)
    df = pd.read_csv(args.reference_csv)

    slides_done = 0
    slides_missing = 0
    slides_failed = 0
    tiles_written = 0

    for _, row in df.iterrows():
        slide = normalize_slide_name(row["wsi_file_name"])
        candidates = [
            wsi_dir / f"{slide}.svs",
            wsi_dir / f"{slide}.tiff",
            wsi_dir / f"{slide}.tif",
        ]
        wsi_path = next((p for p in candidates if p.exists()), None)
        if wsi_path is None:
            slides_missing += 1
            continue

        try:
            oslide = OpenSlide(str(wsi_path))
            extract_level, effective_mpp = choose_level(oslide, args.target_mpp)
            score_level = min(len(oslide.level_dimensions) - 1, extract_level + max(0, args.score_level_offset))

            coords = select_tile_coords(
                oslide,
                extract_level=extract_level,
                score_level=score_level,
                tile_size=args.tile_size,
                max_tiles=args.max_tiles_per_slide,
                min_tissue_ratio=args.min_tissue_ratio,
            )

            if not coords:
                # Fall back to a center tile so each slide can still proceed.
                w0, h0 = oslide.level_dimensions[0]
                ds = float(oslide.level_downsamples[extract_level])
                step0 = int(args.tile_size * ds)
                coords = [(max(0, (w0 - step0) // 2), max(0, (h0 - step0) // 2))]

            for idx, (x0, y0) in enumerate(coords):
                tile = oslide.read_region((int(x0), int(y0)), extract_level, (args.tile_size, args.tile_size)).convert("RGB")
                tile_name = f"{slide}__tile_{idx:03d}.png"
                tile.save(out_dir / tile_name)
                tiles_written += 1

            slides_done += 1
            print(
                f"Prepared {len(coords)} tiles for {slide} "
                f"(level={extract_level}, mpp~{effective_mpp:.3f})"
            )
        except Exception as exc:
            slides_failed += 1
            print(f"Failed {slide}: {exc}")

    print(f"Slides prepared: {slides_done}")
    print(f"Slides missing: {slides_missing}")
    print(f"Slides failed: {slides_failed}")
    print(f"Total tiles written: {tiles_written}")


if __name__ == "__main__":
    main()
