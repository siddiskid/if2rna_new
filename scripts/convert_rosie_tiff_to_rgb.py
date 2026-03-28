#!/usr/bin/env python3
"""Convert ROSIE multi-channel TIFF outputs to RGB PNG composites for IF2RNA pipeline.

Default channel mapping uses ROSIE README examples:
- R: channel 8  (PanCK)
- G: channel 1  (CD45)
- B: channel 0  (DAPI)
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def normalize_channel(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip((x - a) / (b - a), 0.0, 1.0)
    return (255.0 * y).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ROSIE TIFF to RGB PNG")
    parser.add_argument("--input_dir", required=True, help="Directory with ROSIE output TIFF files")
    parser.add_argument("--output_dir", required=True, help="Output directory for RGB PNG files")
    parser.add_argument("--r_channel", type=int, default=8)
    parser.add_argument("--g_channel", type=int, default=1)
    parser.add_argument("--b_channel", type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiffs = sorted(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")))
    if not tiffs:
        print("No TIFF files found")
        return

    converted = 0
    for tiff_path in tiffs:
        arr = tifffile.imread(tiff_path)

        # Accept either CxHxW or HxWxC.
        if arr.ndim != 3:
            continue
        if arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
            # CxHxW
            c, h, w = arr.shape
            get_ch = lambda i: arr[i]
            n_channels = c
        else:
            # HxWxC
            h, w, c = arr.shape
            get_ch = lambda i: arr[:, :, i]
            n_channels = c

        if max(args.r_channel, args.g_channel, args.b_channel) >= n_channels:
            continue

        r = normalize_channel(get_ch(args.r_channel))
        g = normalize_channel(get_ch(args.g_channel))
        b = normalize_channel(get_ch(args.b_channel))
        rgb = np.stack([r, g, b], axis=-1)

        out_png = out_dir / f"{tiff_path.stem}.png"
        Image.fromarray(rgb).save(out_png)
        converted += 1

    print(f"Converted PNG files: {converted}")


if __name__ == "__main__":
    main()
