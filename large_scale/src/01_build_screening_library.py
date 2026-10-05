#!/usr/bin/env python3
"""
01_build_screening_library.py
==============================
Stage 1 of the sEH QSAR screening pipeline.

Takes your candidate-molecule library (one file, or a DIRECTORY of files --
e.g. ZINC's tranche files), standardizes every molecule with RDKit in
parallel, removes duplicates and anything already in your training set, and
writes a compact Parquet file of (name, id, smiles, canonical_smiles).

This script deliberately does NOT compute descriptors/fingerprints or write
a feature matrix -- for very large libraries that matrix wouldn't fit on
disk. Features are computed on the fly, per chunk, in stage 2, and
discarded immediately after use. See the README section
"Screening very large libraries" for why.

Works on libraries of any size: a few hundred rows or a few hundred
million. For anything beyond a few million rows, use .csv/.tsv/.smi/.parquet
(not .xlsx -- Excel caps out at ~1,048,576 rows per sheet, .xlsx is only
supported here for convenience with small libraries).

Default paths (override any of them with flags -- see --help):
    --input       data/my_library.xlsx        (file OR a directory of files)
    --train-file  data/data_for_classification.xlsx
    --output-dir  results/

Example (using all defaults, run from the project root)
---------------------------------------------------------
    python src/01_build_screening_library.py

Example (a directory of ZINC tranche .smi files, 8 worker processes)
-----------------------------------------------------------------------
    python src/01_build_screening_library.py \\
        --input data/zinc_tranches/ \\
        --chunk-size 200000 \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from qsar_utils import (  # noqa: E402
    ParquetChunkWriter,
    find_column,
    iter_library_chunks,
    map_chunksize,
    standardize_mol,
    standardize_one,
)

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standardize a candidate-molecule library (any size) and check it against a training set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=Path("data/my_library.xlsx"),
                    help="Your library: a single file, or a directory of files (.csv/.tsv/.smi/.txt/.parquet/.xlsx). "
                         "Column names are auto-detected (id/ID/zinc_id/... and smiles/SMILES/... , "
                         "or plain 'SMILES ID' with no header for .smi/.txt) -- "
                         "or set them explicitly with --id-col / --smiles-col.")
    p.add_argument("--id-col", type=str, default=None, help="Name of the ID column (auto-detected if omitted).")
    p.add_argument("--smiles-col", type=str, default=None, help="Name of the SMILES column (auto-detected if omitted).")
    p.add_argument("--name-col", type=str, default=None,
                    help="Optional name column. If omitted, the ID column doubles as the name.")

    p.add_argument("--train-file", type=Path, default=Path("data/data_for_classification.xlsx"),
                    help="Your labeled training spreadsheet -- used here only for the training-overlap check.")

    p.add_argument("--output-dir", type=Path, default=Path("results"),
                    help="Where output Parquet files are written.")
    p.add_argument("--chunk-size", type=int, default=100_000,
                    help="Rows processed at a time. Lower this if you run out of memory; "
                         "raise it (with more --n-workers) for more throughput on a big machine.")
    p.add_argument("--n-workers", type=int, default=None,
                    help="Parallel worker processes for standardization. Default: all available CPU cores.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = args.output_dir / "screening_library_standardized.parquet"
    overlap_path = args.output_dir / "compounds_in_training_set.parquet"
    failed_path = args.output_dir / "failed_standardization.parquet"

    # ----------------------------------------------------------------
    # STEP 1: build the training-set canonical SMILES set (for overlap check)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: LOADING TRAINING SET (for the overlap check)")
    print("=" * 60)

    train_df = pd.read_excel(args.train_file)
    print(f"Training set loaded: {len(train_df)} compounds")

    train_smiles_col = find_column(train_df.columns, ["canonical_smiles", "smiles", "Smiles", "SMILES"])
    if train_smiles_col is None:
        raise ValueError(f"No SMILES column found in training file. Columns: {list(train_df.columns)}")

    train_canonical = set()
    for smi in train_df[train_smiles_col]:
        if pd.isna(smi):
            continue
        _, canon, _ = standardize_mol(str(smi))
        if canon is not None:
            train_canonical.add(canon)
    print(f"Unique canonical training SMILES: {len(train_canonical)}")

    # ----------------------------------------------------------------
    # STEP 2: stream, standardize (in parallel), dedupe per chunk, split
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: STREAMING + STANDARDIZING YOUR LIBRARY")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Chunk size: {args.chunk_size:,} rows | Workers: {args.n_workers or 'all CPU cores'}")
    print(
        "\nNote: duplicate removal is done WITHIN each chunk, not across the whole library. "
        "For a library this size, an exact global dedup would need a database-backed approach -- "
        "see the README. Most large libraries (ZINC included) are already deduplicated at the source."
    )

    n_total = n_failed = n_overlap = n_clean = 0
    t_start = time.time()

    with ParquetChunkWriter(clean_path) as clean_writer, \
            ParquetChunkWriter(overlap_path) as overlap_writer, \
            ParquetChunkWriter(failed_path) as failed_writer, \
            ProcessPoolExecutor(max_workers=args.n_workers) as executor:

        for chunk_idx, chunk in enumerate(
            iter_library_chunks(args.input, args.chunk_size, args.id_col, args.smiles_col, args.name_col)
        ):
            chunk = chunk[chunk["smiles"].notna()]
            chunk = chunk[chunk["smiles"].astype(str).str.strip() != ""]
            if len(chunk) == 0:
                continue
            n_total += len(chunk)

            results = list(executor.map(
                standardize_one, chunk["smiles"].astype(str),
                chunksize=map_chunksize(len(chunk), args.n_workers),
            ))
            canon_list = [r[0] for r in results]
            err_list = [r[1] for r in results]

            chunk = chunk.assign(canonical_smiles=canon_list, standardization_error=err_list)
            ok_mask = np.array([e is None for e in err_list])

            failed_chunk = chunk[~ok_mask]
            n_failed += len(failed_chunk)
            if len(failed_chunk) > 0:
                failed_writer.write(failed_chunk[["name", "id", "smiles", "standardization_error"]])

            ok_chunk = chunk[ok_mask].drop_duplicates(subset="canonical_smiles")

            is_overlap = ok_chunk["canonical_smiles"].isin(train_canonical)
            overlap_chunk = ok_chunk[is_overlap]
            clean_chunk = ok_chunk[~is_overlap]

            n_overlap += len(overlap_chunk)
            n_clean += len(clean_chunk)

            if len(overlap_chunk) > 0:
                overlap_writer.write(overlap_chunk[["name", "id", "smiles", "canonical_smiles"]])
            clean_writer.write(clean_chunk[["name", "id", "smiles", "canonical_smiles"]])

            elapsed = time.time() - t_start
            rate = n_total / elapsed if elapsed > 0 else 0
            print(
                f"  chunk {chunk_idx + 1}: processed {n_total:,} total | "
                f"clean {n_clean:,} | failed {n_failed:,} | overlap {n_overlap:,} | "
                f"{rate:,.0f} compounds/sec"
            )

    elapsed = time.time() - t_start

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total processed:          {n_total:,}")
    print(f"Failed standardization:   {n_failed:,} ({n_failed / n_total * 100:.2f}%)" if n_total else "Failed standardization: 0")
    print(f"Already in training set:  {n_overlap:,}")
    print(f"Final clean library:      {n_clean:,}")
    print(f"Time elapsed:             {elapsed / 60:.1f} min ({n_total / elapsed:,.0f} compounds/sec)" if elapsed else "")
    print("\nOutput files:")
    print(f"  Clean, ready for stage 2:  {clean_path}")
    print(f"  Overlap with training:     {overlap_path}")
    print(f"  Failed standardization:    {failed_path}")
    print("===== DONE =====")


if __name__ == "__main__":
    main()
