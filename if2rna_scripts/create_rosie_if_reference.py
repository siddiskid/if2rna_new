#!/usr/bin/env python3
"""
Create IF2RNA reference CSV for ROSIE-converted IF images.

This script matches converted IF images back to the H&E reference rows and
carries over RNA ground truth columns, so converted images can be evaluated
with the IF2RNA model.

Input reference expected columns:
- wsi_file_name
- patient_id
- rna_* columns

Output columns include:
- wsi_file_name
- patient_id
- organ_type
- slide_name
- image_path
- rna_* columns

Usage:
  python if2rna_scripts/create_rosie_if_reference.py \
      --source_reference data/hne_data/metadata/tcga_reference_brca_88.csv \
      --rosie_image_dir data/rosie_if/images \
      --output_reference data/hne_data/metadata/rosie_if_reference_brca_88.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def canonical_slide_key(name: str) -> str:
    key = str(name)
    lower = key.lower()

    for suffix in (".svs", ".tiff", ".tif", ".png", ".jpg", ".jpeg"):
        if lower.endswith(suffix):
            key = key[: -len(suffix)]
            lower = key.lower()
            break

    if lower.endswith("_rosie"):
        key = key[:-6]

    return key


def build_image_index(image_dir: Path, extensions: List[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for ext in extensions:
        for p in image_dir.rglob(f"*{ext}"):
            stem = canonical_slide_key(p.stem)
            # Keep the first match for determinism if duplicates exist.
            if stem not in index:
                index[stem] = p
    return index


def normalize_slide_name(name: str) -> str:
    return canonical_slide_key(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ROSIE IF reference for IF2RNA")
    parser.add_argument("--source_reference", required=True, help="Source H&E reference CSV with RNA columns")
    parser.add_argument("--rosie_image_dir", required=True, help="Directory with ROSIE-converted IF images")
    parser.add_argument("--output_reference", required=True, help="Output IF reference CSV path")
    parser.add_argument("--organ_type", default="BRCA", help="Organ/project tag stored in organ_type column")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        help="Image extensions to search",
    )
    args = parser.parse_args()

    source_reference = Path(args.source_reference)
    rosie_image_dir = Path(args.rosie_image_dir)
    output_reference = Path(args.output_reference)

    if not source_reference.exists():
        raise FileNotFoundError(f"Source reference not found: {source_reference}")
    if not rosie_image_dir.exists():
        raise FileNotFoundError(f"ROSIE image directory not found: {rosie_image_dir}")

    df = pd.read_csv(source_reference)
    if "wsi_file_name" not in df.columns:
        raise ValueError("source_reference must contain wsi_file_name")

    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    if not rna_cols:
        raise ValueError("source_reference contains no rna_* columns")

    image_index = build_image_index(rosie_image_dir, args.extensions)
    print(f"Indexed converted images: {len(image_index)}")

    rows = []
    missing = []

    for _, row in df.iterrows():
        slide = normalize_slide_name(str(row["wsi_file_name"]))
        image_path = image_index.get(slide)
        if image_path is None:
            missing.append(slide)
            continue

        out_row = {
            "wsi_file_name": slide,
            "patient_id": row.get("patient_id", slide),
            "organ_type": args.organ_type,
            "slide_name": slide,
            "image_path": str(image_path),
        }

        # Preserve project metadata if present.
        if "tcga_project" in df.columns:
            out_row["tcga_project"] = row["tcga_project"]

        for c in rna_cols:
            out_row[c] = row[c]

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    output_reference.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_reference, index=False)

    print(f"Saved: {output_reference}")
    print(f"Matched slides: {len(out_df)}")
    print(f"Missing slides: {len(missing)}")

    if missing:
        missing_path = output_reference.with_suffix(".missing.txt")
        missing_path.write_text("\n".join(sorted(set(missing))) + "\n")
        print(f"Missing list: {missing_path}")


if __name__ == "__main__":
    main()
