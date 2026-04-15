#!/usr/bin/env python3
"""Generate Phase 1 Part 3 dataset summary artifacts for the frozen IF2RNA dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def main() -> None:
    ref = Path("data/hne_data_archive_20260403_153424/metadata/if_reference.csv")
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ref.exists():
        raise FileNotFoundError(f"Reference file not found: {ref}")

    df = pd.read_csv(ref, low_memory=False)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]

    overall_rows = [
        {"metric": "reference_csv", "value": str(ref)},
        {"metric": "n_samples", "value": int(len(df))},
        {"metric": "n_patients", "value": int(df["patient_id"].nunique())},
        {"metric": "n_organs", "value": int(df["organ_type"].nunique())},
        {"metric": "n_genes_rna_prefix", "value": int(len(rna_cols))},
        {"metric": "split_protocol", "value": "patient-level 5-fold CV"},
        {"metric": "random_seed", "value": 42},
    ]

    overall_df = pd.DataFrame(overall_rows)
    overall_path = out_dir / "PHASE1_PART3_DATASET_SUMMARY.csv"
    overall_df.to_csv(overall_path, index=False)

    organ_df = (
        df["organ_type"]
        .value_counts()
        .rename_axis("organ_type")
        .reset_index(name="n_samples")
    )
    organ_df["percent"] = (organ_df["n_samples"] / len(df) * 100).round(2)
    organ_path = out_dir / "PHASE1_PART3_ORGAN_DISTRIBUTION.csv"
    organ_df.to_csv(organ_path, index=False)

    seed = 42
    n_splits = 5
    valid_size = 0.1

    indices = pd.Series(range(len(df)), index=df.index)
    patients_unique = df["patient_id"].unique()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    split_rows = []
    for fold, (ind_train, ind_test) in enumerate(skf.split(patients_unique)):
        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_mask = df["patient_id"].isin(patients_test)
        test_idx = indices[test_mask].values

        p_train, p_valid = train_test_split(
            patients_train,
            test_size=valid_size,
            random_state=seed,
        )
        valid_mask = df["patient_id"].isin(p_valid)
        train_mask = df["patient_id"].isin(p_train)

        train_idx = indices[train_mask].values
        valid_idx = indices[valid_mask].values

        split_rows.append(
            {
                "fold": fold,
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(valid_idx)),
                "test_samples": int(len(test_idx)),
                "train_patients": int(df.iloc[train_idx]["patient_id"].nunique()),
                "val_patients": int(df.iloc[valid_idx]["patient_id"].nunique()),
                "test_patients": int(df.iloc[test_idx]["patient_id"].nunique()),
            }
        )

    split_df = pd.DataFrame(split_rows)
    split_path = out_dir / "PHASE1_PART3_SPLIT_SUMMARY.csv"
    split_df.to_csv(split_path, index=False)

    print(f"Wrote: {overall_path}")
    print(f"Wrote: {organ_path}")
    print(f"Wrote: {split_path}")


if __name__ == "__main__":
    main()
