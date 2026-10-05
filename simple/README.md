# sEH QSAR Screening Pipeline -- Simple version

For screening libraries of a few hundred up to roughly a few hundred
thousand candidate molecules against a soluble epoxide hydrolase (sEH)
QSAR model. Everything here loads into memory and reads/writes plain
`.xlsx` files -- easy to open, easy to read, easy to debug.

**Screening something ZINC-sized (millions to hundreds of millions of
compounds)?** Use [`../large_scale/`](../large_scale/) instead -- see the
top-level README's
["Which pipeline should I use?"](../README.md#which-pipeline-should-i-use).

## Pipeline overview

```
01_build_screening_library.py   ──►  02_xgb_classification_screening.py  ──►  03_xgb_regression_ranking.py
 standardize your molecules,          predict active / inactive,                predict pIC50, rank,
 align features to training set       flag applicability domain                 build a diverse shortlist
```

| Stage | Script | What it needs | What it produces |
|---|---|---|---|
| 1. Build library | `01_build_screening_library.py` | Your molecules (`data/my_library.xlsx`) + training data | A standardized, feature-aligned screening library |
| 2. Classify | `02_xgb_classification_screening.py` | Stage 1's output + training data | Active (pIC50>8)/inactive (pIC50<6) predictions, confidence, applicability-domain flag, top-N shortlist |
| 3. Rank | `03_xgb_regression_ranking.py` | Stage 2's predicted actives + training data | Predicted pIC50 with uncertainty, a top-N shortlist, and a scaffold-diverse shortlist |

## Project layout

```
simple/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                   <- schema details for the files below
│   ├── data_for_classification.xlsx   <- my labeled training data if you need to check them (classifier)
│   ├── data_for_regression.xlsx       <- my labeled training data (regressor)
│   └── my_library.xlsx                <- the molecules YOU want to screen
├── results/                        <- created automatically when you run the scripts
└── src/
    ├── qsar_utils.py
    ├── 01_build_screening_library.py
    ├── 02_xgb_classification_screening.py
    └── 03_xgb_regression_ranking.py
```

Every script defaults to these exact paths, so once your three data files
are in `data/`, run all three stages with **no extra flags at all**. Run
any script with `-h` to see every option.

## Installing dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

RDKit installs via `pip` on recent Python on Windows/macOS/Linux. If that
gives you trouble: `conda create -n seh-qsar python=3.11 && conda activate
seh-qsar && conda install -c conda-forge rdkit`, then `pip install -r
requirements.txt` for the rest.

## Running it

```bash
python src/01_build_screening_library.py
python src/02_xgb_classification_screening.py
python src/03_xgb_regression_ranking.py
```

Everything lands in `results/` (and `results/regression/` for stage 3):
Excel files ranking your compounds, plus PNG plots. Stage 2 also saves
`results/top_predicted_actives.xlsx` -- the top-scoring predicted actives
(500 by default; change with `--top-n-actives`) in one file for a quick
manual look, separate from the full results table.

## Methods

- **Featurization**: RDKit physicochemical descriptors + Morgan (ECFP-style)
  fingerprints, computed from standardized SMILES.
- **Model selection**: XGBoost, Random Forest and SVM were both evaluated across
  random,Tanimoto similarity Butina, and Murcko-Bemis scaffold train/test splits during development.
  XGBoost was chosen as the final classifier for its better specificity,
  and as the final regressor for its better point-estimate accuracy
  (Random Forest gave better-calibrated uncertainty but slightly worse
  point predictions).
- **Classification**: validated with 5-fold stratified cross-validation.
- **Regression**: pIC50 prediction with a 90% prediction interval from
  XGBoost quantile regression (5th/50th/95th percentile models).
- **Applicability domain (AD)**: max Tanimoto similarity to the nearest
  training compound. Stage 2 prints a plain-language warning (and saves a
  plot) if a large fraction of your library falls outside the AD.
- **Composite scoring & diversification**: final ranking blends normalized
  predicted pIC50 and normalized prediction certainty
  (`--weight-pred`/`--weight-uncertainty`). A greedy Murcko-scaffold
  selection then produces a diverse shortlist.

## Reusing a trained model instead of retraining every run

```bash
python src/02_xgb_classification_screening.py --skip-training
python src/03_xgb_regression_ranking.py --skip-training
```

## Limitations

- Trained on a modest ChEMBL-derived dataset for a single target;
  predictions on scaffolds far from the training distribution (flagged by
  the AD check) should be treated as hypotheses, not conclusions.
- Internal validation is cross-validation on the training set; no wet-lab
  confirmation is included in this repository.
- Composite score weighting is a modeling choice, not ground truth.

## License

MIT -- see [`../LICENSE`](../LICENSE).
