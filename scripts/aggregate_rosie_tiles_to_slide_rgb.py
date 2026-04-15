#!/usr/bin/env python3
"""Aggregate ROSIE-converted tile RGB images into one mosaic per slide.

Input expected tile naming:
  <slide_id>__tile_<idx>_ROSIE.png
Output naming:
  <slide_id>.png
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image


def parse_slide_id(stem: str) -> str:
    base = stem
    if base.endswith("_ROSIE"):
        base = base[:-6]
    marker = "__tile_"
    pos = base.find(marker)
    if pos == -1:
        return ""
    return base[:pos]


def make_mosaic(images, tile_size: int, cols: int) -> Image.Image:
    n = len(images)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), (0, 0, 0))

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        if img.size != (tile_size, tile_size):
            img = img.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        canvas.paste(img, (c * tile_size, r * tile_size))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ROSIE tile RGB outputs per slide")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tiles_per_slide", type=int, default=64)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=512)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in sorted(in_dir.glob("*.png")):
        slide_id = parse_slide_id(p.stem)
        if slide_id:
            groups[slide_id].append(p)

    saved = 0
    for slide_id, paths in groups.items():
        paths = paths[: args.max_tiles_per_slide]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        mosaic = make_mosaic(imgs, tile_size=args.tile_size, cols=args.grid_cols)
        mosaic.save(out_dir / f"{slide_id}.png")
        saved += 1

    print(f"Slides aggregated: {saved}")
    print(f"Input tile images: {sum(len(v) for v in groups.values())}")


if __name__ == "__main__":
    main()
