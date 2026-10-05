# `data/` folder (simple pipeline)

Expected files, exact names (or override the path with a flag when running
a script):

```
data/
├── data_for_classification.xlsx   <- training data for stage 2 (classifier)
├── data_for_regression.xlsx       <- training data for stage 3 (regressor)
└── my_library.xlsx                <- YOUR molecules, the ones you want screened
```

## `data_for_classification.xlsx`

One row per training compound:

| Column | Description |
|---|---|
| `ChEMBL_ID` (or `chembl_id` / `ID`) | Compound identifier |
| `canonical_smiles` (or `smiles` / `Smiles`) | SMILES string |
| `Label_low_high` | Binary label used by the classifier (0 = inactive, 1 = active) |
| *(everything else)* | Feature columns -- RDKit descriptors + `MF1..MFn` Morgan fingerprint bits |

## `data_for_regression.xlsx`

Same shape, with a continuous `pChEMBL` column instead of a binary label.

## `my_library.xlsx`

Any spreadsheet (`.xlsx` or `.csv`) with an ID column and a SMILES column
(names auto-detected: `id`/`ID`/`compound_id`/... and `smiles`/`SMILES`/...,
or set explicitly with `--id-col` / `--smiles-col`).

## How the feature schema stays consistent

Stage 1 recomputes RDKit descriptors and Morgan fingerprints from your
library's SMILES, then reindexes those columns to exactly match
`data_for_classification.xlsx`'s feature columns (same names, same order;
anything missing is filled with 0).

## Is this the right pipeline for my library size?

This pipeline loads everything into memory and writes `.xlsx` files --
simple to read, but Excel itself caps out at ~1,048,576 rows per sheet, and
in practice things get slow well before that. As a rough guide:

- **Up to ~50,000 compounds:** comfortable.
- **~50,000-300,000:** works, but stage 2's Tanimoto applicability-domain
  check and plotting will take noticeably longer.
- **Beyond that (and especially anything ZINC-sized, i.e. millions to
  hundreds of millions):** switch to the `large_scale/` pipeline instead --
  see the top-level README for why and how.
