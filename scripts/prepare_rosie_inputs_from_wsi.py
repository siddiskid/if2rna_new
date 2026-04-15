#!/usr/bin/env python3
"""Create ROSIE-compatible PNG inputs from WSI slides.

Reads slide names from a reference CSV and exports one RGB PNG per slide using
an intermediate WSI pyramid level (instead of the lowest thumbnail level) to
preserve more morphology for downstream conversion.
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from openslide import OpenSlide


def normalize_name(name: str) -> str:
    if name.endswith(".svs"):
        return name[:-4]
    if name.endswith(".tiff"):
        return name[:-5]
    return name


def select_level(slide_obj: OpenSlide, requested_level: int, target_downsample: float) -> int:
    """Pick a pyramid level for preview export.

    Priority:
    1) Explicit level if provided and valid
    2) Level with downsample closest to target_downsample
    """
    max_level = len(slide_obj.level_dimensions) - 1

    if requested_level >= 0:
        if requested_level > max_level:
            raise ValueError(f"Requested level {requested_level} exceeds max level {max_level}")
        return requested_level

    downsamples = list(slide_obj.level_downsamples)
    best_idx = min(range(len(downsamples)), key=lambda i: abs(float(downsamples[i]) - target_downsample))
    return int(best_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ROSIE PNG inputs from WSI")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--max_side",
        type=int,
        default=4096,
        help="Resize longest side to this value (0 disables resize)",
    )
    parser.add_argument(
        "--wsi_level",
        type=int,
        default=-1,
        help="Explicit OpenSlide pyramid level to read (default: auto by --target_downsample)",
    )
    parser.add_argument(
        "--target_downsample",
        type=float,
        default=16.0,
        help="Auto-select level whose OpenSlide downsample is closest to this value",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.reference_csv)
    wsi_dir = Path(args.wsi_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    missing = 0
    failed = 0

    for _, row in df.iterrows():
        slide = normalize_name(str(row["wsi_file_name"]))

        candidates = [
            wsi_dir / f"{slide}.svs",
            wsi_dir / f"{slide}.tiff",
            wsi_dir / f"{slide}.tif",
        ]
        wsi_path = next((p for p in candidates if p.exists()), None)
        if wsi_path is None:
            missing += 1
            continue

        out_png = out_dir / f"{slide}.png"
        if out_png.exists():
            done += 1
            continue

        try:
            slide_obj = OpenSlide(str(wsi_path))
            level = select_level(slide_obj, args.wsi_level, args.target_downsample)
            dims = slide_obj.level_dimensions[level]
            img = slide_obj.read_region((0, 0), level, dims).convert("RGB")

            if args.max_side and max(img.size) > args.max_side:
                scale = args.max_side / float(max(img.size))
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                img = img.resize(new_size, Image.Resampling.BILINEAR)

            img.save(out_png)
            done += 1
        except Exception:
            failed += 1

    print(f"Prepared PNGs: {done}")
    print(f"Missing WSIs: {missing}")
    print(f"Failed WSIs: {failed}")


if __name__ == "__main__":
    main()
