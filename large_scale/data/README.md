# `data/` folder (large-scale pipeline)

```
data/
├── data_for_classification.xlsx   <- training data for stage 2 (small -- stays as .xlsx)
├── data_for_regression.xlsx       <- training data for stage 3 (small -- stays as .xlsx)
└── my_library/  (or my_library.csv, my_library.smi, ...)  <- YOUR molecules to screen
```

Training data stays small (a few thousand ChEMBL compounds) and stays as
`.xlsx` regardless of pipeline -- only the *screening library* needs
big-data handling, since that's the part that can be hundreds of millions
of rows.

## `data_for_classification.xlsx` / `data_for_regression.xlsx`

Same schema as the simple pipeline -- see that pipeline's `data/README.md`
for the exact column requirements. Nothing changes here regardless of how
big your screening library is.

## Your screening library

Point `--input` at **a single file OR a directory** (searched recursively,
subfolders included) containing any mix of:

- `.csv`, `.tsv`
- `.smi`, `.txt` (the common ZINC tranche format: `smiles<TAB>zinc_id`,
  with or without a header row -- both are auto-detected)
- gzip-compressed versions of any of the above (`.csv.gz`, `.smi.gz`, etc.)
- `.parquet`
- `.xlsx`/`.xls` (works, but only advisable for smaller inputs -- pandas
  has to load the whole file for these regardless of pipeline)

**Using ZINC as downloaded, no pre-processing needed:** ZINC ships as a
top-level folder containing many subfolders, each containing many small
tranche files. Just point `--input` at that top-level folder:

```bash
python src/01_build_screening_library.py --input /path/to/zinc_download/
```

The pipeline finds every matching file underneath (however many subfolders
deep), sniffs each file's delimiter and whether it has a header row, and
streams through all of them as one logical library. You do not need to
merge, reformat, or flatten anything yourself.

If your files use unusual column names, pass `--id-col` / `--smiles-col`
explicitly and they'll be used for every file.

