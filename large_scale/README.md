# sEH QSAR Screening Pipeline -- Large-Scale (streaming) version

This is the version for screening libraries too big to fit in memory or in
an Excel file -- ZINC-scale (tens to hundreds of millions of compounds).
If your library is a few hundred to a few hundred thousand compounds, use
the [`simple/`](../simple/) pipeline instead -- it's plainer to read and
easier to debug. See the top-level README's
["Which pipeline should I use?"](../README.md#which-pipeline-should-i-use)
if you're not sure.

## How this differs from the simple pipeline

| | simple/ | large_scale/ (this one) |
|---|---|---|
| Screening input | one `.xlsx`/`.csv` file | a file **or a directory** (searched recursively) -- `.csv`, `.tsv`, `.smi`, `.txt`, `.parquet`, `.xlsx`, gzip-compressed variants |
| Processing | loads everything into memory | streams through in chunks, always |
| Output format | `.xlsx` | `.parquet` for anything that scales with library size; small shortlists are still `.xlsx` for convenience |
| Parallelism | single process | multiple worker processes (`--n-workers`) |
| Feature matrix | built once, kept on disk | computed per chunk, used, and discarded -- never accumulated for the full library |

The core reason for all of this: a full RDKit-descriptor + fingerprint
feature matrix for 700 million compounds would be on the order of **7 TB**.
Even setting memory aside, Excel itself caps out at 1,048,576 rows per
sheet -- so past a few hundred thousand compounds, `.xlsx` isn't just slow,
it's impossible. This pipeline computes features for one chunk at a time,
predicts, writes out compact result columns, and throws the feature matrix
away before touching the next chunk.

## Using ZINC (or any big library) with no pre-processing

Point `--input` at ZINC's top-level download folder as-is:

```bash
python src/01_build_screening_library.py --input /path/to/zinc_download/
```

Subfolders are searched recursively, delimiter and header presence are
auto-detected per file (ZINC's tab-separated `smiles<TAB>zinc_id` tranche
files work whether or not they have a header row), and gzip-compressed
files are read transparently. See [`data/README.md`](data/README.md) for
the full format list.

## Pipeline overview

```
01_build_screening_library.py   ──►  02_xgb_classification_screening.py  ──►  03_xgb_regression_ranking.py
 stream + standardize                 stream + featurize + classify              stream + featurize + rank
 (Parquet, no features stored)        (features computed & discarded per chunk)  (same, on the much smaller active set)
```

| Stage | Reads | Writes |
|---|---|---|
| 1 | your library (any size/format above) | `results/screening_library_standardized.parquet` |
| 2 | stage 1's output + training data | `results/screening_predictions_all.parquet`, `results/actives_for_regression.parquet`, `results/top_predicted_actives.xlsx` |
| 3 | stage 2's actives + training data | `results/regression/all_actives_regression_scored.parquet`, plus small `.xlsx` shortlists |

Default paths need no flags if your files match the names in
[`data/README.md`](data/README.md); run any script with `-h` for every
option.

## Opening Parquet files

Parquet isn't something you double-click open like an Excel file, but it's
easy to work with:

**In Python (pandas)** -- works for files that fit in memory:
```python
import pandas as pd
df = pd.read_parquet("results/screening_predictions_all.parquet")
```

**In Python, for files too big to load whole** -- read only the columns you
need, or read it in batches:
```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("results/screening_predictions_all.parquet")
for batch in pf.iter_batches(batch_size=100_000, columns=["name", "active_probability"]):
    df_chunk = batch.to_pandas()
    ...
```

**Without writing any code** -- [DuckDB](https://duckdb.org/) can query a
Parquet file directly, out-of-core, from its command-line shell or a
one-line Python call, and is the easiest way to poke around a huge results
file:
```sql
duckdb -c "SELECT * FROM 'results/screening_predictions_all.parquet' ORDER BY active_probability DESC LIMIT 50"
```

**Converting to Excel for a quick look** -- fine for anything you've
already filtered down to a reasonable size (the top-N shortlists are
already saved as `.xlsx` for exactly this reason):
```python
import pandas as pd
pd.read_parquet("results/regression/all_actives_regression_scored.parquet").to_excel("shortlist.xlsx", index=False)
```
Don't do this on the full `screening_predictions_all.parquet` if your
library was ZINC-sized -- that's exactly the file that won't fit in Excel.

## Screening very large libraries: what's actually achievable

**Real, measured throughput** (standardize + all RDKit descriptors +
1024-bit Morgan fingerprint, no multiprocessing overhead): **~200
molecules/sec on one CPU core.** This is the number that matters --
descriptor computation, not I/O, is the bottleneck, and it's fixed by what
the trained model needs as input (you can't skip descriptors the model was
trained on without retraining it on fewer features, which is out of scope
of a screening run).

The pipeline parallelizes this across CPU cores (`--n-workers`, default:
all available cores).


**Benchmark your own machine before committing to a full run.** Don't
trust a test on a few dozen or a few hundred compounds -- fixed startup
costs (spawning workers, training the model) dominate at that scale and
will make the reported compounds/sec look far worse than reality. Instead:

```bash
python src/02_xgb_classification_screening.py --n-workers 8 --no-plots
# (on a 100,000-500,000 compound subset)
```
and read the rate from one of the LATER chunk lines, not the first. Then:
`estimated_hours = 700_000_000 / (measured_rate * 3600)`.


### Speed levers in the code, if you still want to trim runtime

- **`--skip-ad`** (stage 2): skips the Tanimoto applicability-domain check
  entirely. You lose the "is this prediction on a novel scaffold"
  flag, but it removes a real per-compound cost.
- **`--ad-max-train-refs N`** (stage 2, default 1000): instead of skipping
  the AD check, cap how many training compounds each screening compound is
  compared against (a random subsample). Lower this further (e.g. 200) to
  trade some AD precision for speed without losing the check entirely.
- **`--single-point-estimate`** (stage 3): fits one regressor instead of
  three quantile models, ~3x faster prediction on the active set. You lose
  the 90% prediction interval / uncertainty term (ranking becomes pure
  predicted-pIC50 order).
- **`--n-workers`**: set this to your actual core count, not more --
  over-subscribing doesn't help and can hurt.
- **`--chunk-size`**: larger chunks amortize per-chunk overhead better on
  a machine with a lot of RAM; lower it if you're memory-constrained.

None of these change what the model was trained on, so they're all safe to
toggle per-run without retraining anything.

## A note on exact deduplication at this scale

Stage 1 deduplicates canonical SMILES **within each chunk**, not across the
whole library -- an exact global dedup across hundreds of millions of rows
would need a database-backed approach (e.g. a persistent hash index),
which is outside this pipeline's scope. In practice this rarely matters:
ZINC (and most curated libraries) are already deduplicated at the source,
so residual duplicates after per-chunk dedup should be rare and harmless
(a compound appearing twice just gets scored twice).

## Limitations

Same caveats as the simple pipeline (see its README) apply here too:
training data size, applicability domain, and composite-score weighting
are all modeling choices, not ground truth. At this scale, additionally:
predictions on compounds far outside the applicability domain are
extrapolations on chemistry the model has essentially no information
about -- the AD warning at the end of stage 2 is worth reading, not just
skimming past.

## License

MIT -- see [`../LICENSE`](../LICENSE).
