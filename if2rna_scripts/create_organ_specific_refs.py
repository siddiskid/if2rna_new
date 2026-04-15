#!/usr/bin/env python3
"""Create organ-specific reference CSVs from the frozen train-ready reference."""

from pathlib import Path
import pandas as pd


def main() -> None:
    src = Path("data/hne_data_archive_20260403_153424/metadata/if_reference_phase1_train_ready.csv")
    out_dir = Path("data/hne_data_archive_20260403_153424/metadata/organ_refs")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)
    if "organ_type" not in df.columns:
        raise ValueError("organ_type column not found in reference CSV")

    for organ in sorted(df["organ_type"].dropna().unique()):
        sub = df[df["organ_type"] == organ].reset_index(drop=True)
        out = out_dir / f"if_reference_{organ.lower()}.csv"
        sub.to_csv(out, index=False)
        print(f"Wrote {out} (n={len(sub)})")


if __name__ == "__main__":
    main()
