#!/usr/bin/env python3
"""
01_build_screening_library.py  (SIMPLE pipeline)
==================================================
Stage 1 of the sEH QSAR screening pipeline -- the small-library version.

Takes ONE spreadsheet of candidate molecules (an ID column + a SMILES
column -- that's the only requirement), standardizes every molecule with
RDKit, removes duplicates and anything already in your training set, then
builds an RDKit-descriptor + Morgan-fingerprint feature matrix aligned to
your training set's exact feature schema.

This version loads everything into memory and writes plain .xlsx files --
simple to read, simple to debug. It's a good fit for libraries up to
roughly a few hundred thousand compounds. For millions to hundreds of
millions of compounds (e.g. the full ZINC database), use the large_scale/
pipeline instead -- see the top-level README for which one to pick.

Works identically on Windows, macOS, and Linux -- every path is handled
with pathlib.

Default paths (override any of them with flags -- see --help):
    --input-file  data/my_library.xlsx
    --train-file  data/data_for_classification.xlsx
    --output-dir  results/

Example (using all defaults, run from the project root)
---------------------------------------------------------
    python src/01_build_screening_library.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem

sys.path.append(str(Path(__file__).resolve().parent))
from qsar_utils import (  # noqa: E402
    DESCRIPTOR_NAMES,
    compute_descriptors,
    find_column,
    get_morgan_generator,
    standardize_mol,
)

warnings.filterwarnings("ignore")

TRAIN_META = [
    "ChEMBL_ID", "Smiles", "smiles", "canonical_smiles", "pChEMBL",
    "Label_binary", "Label_low_high", "Label_3class", "Regression",
]

ID_COL_CANDIDATES = ["id", "ID", "Id", "compound_id", "molecule_id", "drugId", "drugbank_id"]
SMILES_COL_CANDIDATES = ["smiles", "SMILES", "Smiles", "canonical_smiles"]
NAME_COL_CANDIDATES = ["name", "Name", "compound_name", "molecule_name", "drugName"]

PRINT_EVERY = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standardize a candidate-molecule library and align its features to a training set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-file", type=Path, default=Path("data/my_library.xlsx"),
                    help="Your library spreadsheet (.xlsx or .csv) with one row per molecule: "
                         "an ID column and a SMILES column. Column names are auto-detected "
                         "or set them explicitly with --id-col / --smiles-col.")
    p.add_argument("--id-col", type=str, default=None, help="Name of the ID column (auto-detected if omitted).")
    p.add_argument("--smiles-col", type=str, default=None, help="Name of the SMILES column (auto-detected if omitted).")
    p.add_argument("--name-col", type=str, default=None,
                    help="Optional name column. If omitted, the ID column doubles as the name.")

    p.add_argument("--train-file", type=Path, default=Path("data/data_for_classification.xlsx"),
                    help="Your labeled training spreadsheet -- used for the overlap check and the feature schema.")

    p.add_argument("--output-dir", type=Path, default=Path("results"))

    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--fp-no-chirality", action="store_true")
    p.add_argument("--fp-col-prefix", type=str, default="MF")

    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / "screening_library_for_ML.xlsx"

    fp_chirality = not args.fp_no_chirality
    generator = get_morgan_generator(args.fp_radius, args.fp_size, fp_chirality)

    # ----------------------------------------------------------------
    # STEP 1: load and normalize your library file
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: LOADING YOUR LIBRARY FILE")
    print("=" * 60)

    raw = read_table(args.input_file)
    print(f"Loaded: {len(raw)} rows | columns: {list(raw.columns)}")

    id_col = args.id_col or find_column(raw.columns, ID_COL_CANDIDATES)
    smiles_col = args.smiles_col or find_column(raw.columns, SMILES_COL_CANDIDATES)
    name_col = args.name_col or find_column(raw.columns, NAME_COL_CANDIDATES)

    if smiles_col is None or id_col is None:
        raise ValueError(
            f"Could not auto-detect id/smiles columns. Columns present: {list(raw.columns)}. "
            "Pass --id-col / --smiles-col explicitly."
        )
    print(f"Using ID column: '{id_col}' | SMILES column: '{smiles_col}'"
          + (f" | name column: '{name_col}'" if name_col else " | no name column found, using ID as name"))

    merged = pd.DataFrame({
        "name": raw[name_col] if name_col else raw[id_col].astype(str),
        "id": raw[id_col].astype(str),
        "smiles": raw[smiles_col],
    })
    merged = merged[merged["smiles"].notna()].reset_index(drop=True)
    merged = merged[merged["smiles"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"After removing missing SMILES: {len(merged)}")

    before = len(merged)
    merged = merged.drop_duplicates(subset="smiles").reset_index(drop=True)
    print(f"After raw SMILES dedup:        {len(merged)} (removed {before - len(merged)})")

    # ----------------------------------------------------------------
    # STEP 2: load training set (overlap check + feature schema)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: LOADING TRAINING SET")
    print("=" * 60)

    train_df = pd.read_excel(args.train_file)
    print(f"Training set loaded: {len(train_df)} compounds")

    train_smiles_col = find_column(train_df.columns, ["canonical_smiles", "smiles", "Smiles", "SMILES"])
    if train_smiles_col is None:
        raise ValueError(f"No SMILES column found in training file. Columns: {list(train_df.columns)}")

    print("Standardizing training SMILES for the overlap check...")
    train_canonical = set()
    for smi in train_df[train_smiles_col]:
        if pd.isna(smi):
            continue
        _, canon, _ = standardize_mol(str(smi))
        if canon is not None:
            train_canonical.add(canon)
    print(f"Unique canonical training SMILES: {len(train_canonical)}")

    # ----------------------------------------------------------------
    # STEP 3: standardize your library
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: STANDARDIZING YOUR LIBRARY")
    print("=" * 60)
    print(f"Processing {len(merged)} compounds...")

    canonical_smiles_list, error_reasons = [], []
    n_failed = 0
    for i, smi in enumerate(merged["smiles"]):
        if (i + 1) % PRINT_EVERY == 0 or i == 0:
            print(f"  {i + 1}/{len(merged)} (failed so far: {n_failed})")
        _, canon, error = standardize_mol(str(smi))
        canonical_smiles_list.append(canon)
        error_reasons.append(error)
        if error is not None:
            n_failed += 1

    merged["canonical_smiles"] = canonical_smiles_list
    merged["standardization_error"] = error_reasons
    merged["standardization_ok"] = [e is None for e in error_reasons]
    print(f"\nStandardization complete. Successful: {len(merged) - n_failed} | Failed: {n_failed}")

    failed_df = merged[~merged["standardization_ok"]].copy()
    if len(failed_df) > 0:
        failed_path = args.output_dir / "failed_standardization.xlsx"
        failed_df.to_excel(failed_path, index=False)
        print(f"Failed compounds saved to: {failed_path}")

    merged_ok = merged[merged["standardization_ok"]].copy().reset_index(drop=True)

    before = len(merged_ok)
    merged_ok = merged_ok.drop_duplicates(subset="canonical_smiles").reset_index(drop=True)
    print(f"Canonical SMILES dedup: {before} -> {len(merged_ok)}")

    # ----------------------------------------------------------------
    # STEP 4: overlap check
    # ----------------------------------------------------------------
    is_overlap = merged_ok["canonical_smiles"].isin(train_canonical)
    overlap_df = merged_ok[is_overlap].copy()
    clean_df = merged_ok[~is_overlap].copy().reset_index(drop=True)
    print(f"\nOverlapping with training set: {len(overlap_df)} | Remaining: {len(clean_df)}")
    if len(overlap_df) > 0:
        overlap_path = args.output_dir / "compounds_in_training_set.xlsx"
        overlap_df.to_excel(overlap_path, index=False)
        print(f"Overlap saved to: {overlap_path}")

    # ----------------------------------------------------------------
    # STEP 5: feature schema from the training file
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: BUILDING FEATURE MATRIX")
    print("=" * 60)

    train_feat_df = pd.read_excel(args.train_file, nrows=5)
    train_features = [c for c in train_feat_df.columns if c not in TRAIN_META]
    print(f"Training feature columns: {len(train_features)}")

    fps_clean, descs_clean = [], []
    for i, canon in enumerate(clean_df["canonical_smiles"]):
        if (i + 1) % PRINT_EVERY == 0 or i == 0:
            print(f"  Feature extraction: {i + 1}/{len(clean_df)}")
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            fps_clean.append([0] * args.fp_size)
            descs_clean.append([0] * len(DESCRIPTOR_NAMES))
            continue
        fps_clean.append(list(generator.GetFingerprint(mol)))
        descs_clean.append(compute_descriptors(mol))

    fp_df = pd.DataFrame(fps_clean, columns=[f"{args.fp_col_prefix}{i + 1}" for i in range(args.fp_size)])
    desc_df = pd.DataFrame(descs_clean, columns=DESCRIPTOR_NAMES)
    X_new = pd.concat([desc_df, fp_df], axis=1).reindex(columns=train_features, fill_value=0)
    print(f"Final feature matrix: {X_new.shape}")

    final_df = pd.concat(
        [clean_df[["name", "id", "smiles", "canonical_smiles"]].reset_index(drop=True), X_new.reset_index(drop=True)],
        axis=1,
    )
    final_df.to_excel(output_file, index=False)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Final screening library: {len(final_df)} compounds, {X_new.shape[1]} features")
    print(f"Output: {output_file}")
    print("===== DONE =====")


if __name__ == "__main__":
    main()
