# sEH QSAR Screening Pipeline

A QSAR pipeline for identifying novel inhibitors of **soluble epoxide
hydrolase (sEH)**. Give it a spreadsheet (or a folder of files, at scale)
of candidate molecules and it predicts which are likely active, ranks them
by predicted potency and confidence, and produces a scaffold-diverse
shortlist for follow-up.

This repository has **two pipelines** that do the same science with two
very different implementations for smaller and larger libraries:

- [`simple/`](simple/) -- loads everything into memory, reads/writes
  `.xlsx`. Easy to read, easy to debug. Good for a few hundred up to
  roughly a few hundred thousand compounds.
- [`large_scale/`](large_scale/) -- streams through the library in
  chunks, computes features per chunk and discards them immediately,
  writes `.parquet`, parallelizes across CPU cores. Built for ZINC-scale
  libraries (tens to hundreds of millions of compounds, or more).

## Which pipeline should I use?

| Your library size | Use |
|---|---|
| Up to ~50,000 compounds | `simple/` |
| ~50,000-300,000 | `simple/` still works; `large_scale/` if it feels slow |
| Millions+ (e.g. a ZINC subset or the full database) | `large_scale/` |

Both pipelines produce the same kinds of outputs (active/inactive
predictions, predicted pIC50, a scaffold-diverse shortlist) using the same
underlying model logic -- the difference is entirely in how much data each
one can handle and how it manages memory while doing so.

## Using it after cloning

```bash
git clone https://github.com/MarSKara/sEH-QSAR.git
cd sEH-QSAR/simple        # or large_scale/, depending on your library size
```
then follow that pipeline's own `README.md` -- both are self-contained
(their own `requirements.txt`, their own `data/` and `src/`).

## Cross-platform

Both pipelines work unchanged on Windows, macOS, and Linux -- every file
path is built with Python's `pathlib`, and plotting uses a headless-safe
backend. The large-scale pipeline's multiprocessing also works on all
three, though Windows has somewhat higher per-process startup overhead
than Linux/macOS (see `large_scale/README.md`'s performance section for
why that matters when benchmarking).

## License

MIT -- see [`LICENSE`](LICENSE). Applies to both pipelines.
