#!/usr/bin/env python3
"""
03_xgb_regression_ranking.py
=============================
Stage 3 (final stage) of the sEH QSAR screening pipeline.

Trains XGBoost quantile-regression models on your labeled training data
(small, loaded fully in memory as usual), then STREAMS the predicted
actives from stage 2 through in chunks: computes descriptors + Morgan
fingerprints on the fly, predicts pIC50 with a 90% prediction interval,
scores each compound, and writes compact results -- again, the feature
matrix is never accumulated for the full active set.

A note on the composite score at scale: with a streamed, unbounded-length
input, there's no "whole batch" to normalize predicted pIC50 / uncertainty
against (the usual min-max approach needs to see everything first). Instead,
this version normalizes against the TRAINING data's own prediction range,
computed once up front. That range doesn't depend on how big your screening
run is or what happens to be in it, so composite scores are stable and
comparable across runs -- a genuine improvement over batch-relative
normalization, not just a workaround.

Default paths (override any of them with flags -- see --help):
    --train-file      data/data_for_regression.xlsx
    --screening-file  results/actives_for_regression.parquet   (output of stage 2)
    --output-dir      results/regression/

Example (using all defaults, run from the project root, after stage 2)
-------------------------------------------------------------------------
    python src/03_xgb_regression_ranking.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from qsar_utils import (  # noqa: E402
    DESCRIPTOR_NAMES,
    OnlineHistogram,
    ParquetChunkWriter,
    TopKTracker,
    activity_category,
    featurize_for_regression_one,
    init_featurize_only_worker,
    iter_parquet_chunks,
    map_chunksize,
)

import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

TRAIN_META = [
    "ChEMBL_ID", "Smiles", "smiles", "canonical_smiles", "pChEMBL",
    "Label_binary", "Label_low_high", "Label_3class", "Regression",
]

BEST_PARAMS = {
    "subsample": 0.7,
    "reg_lambda": 2,
    "reg_alpha": 0,
    "n_estimators": 200,
    "min_child_weight": 3,
    "max_depth": 7,
    "learning_rate": 0.05,
    "gamma": 0.1,
    "colsample_bytree": 0.6,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8,
    "verbosity": 0,
}

Q_LOW = 0.05
Q_HIGH = 0.95

CAT_COLORS = {
    "Highly active (pIC50>=8)": "#1a7a4a",
    "Active (7<=pIC50<8)": "#2ecc71",
    "Moderately active (6<=pIC50<7)": "#f39c12",
    "Weakly active (pIC50<6)": "#e74c3c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/apply the XGBoost pIC50 regressor, streaming over stage 2's predicted actives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file", type=Path, default=Path("data/data_for_regression.xlsx"))
    p.add_argument("--screening-file", type=Path, default=Path("results/actives_for_regression.parquet"),
                    help="Output of 02_xgb_classification_screening.py.")
    p.add_argument("--output-dir", type=Path, default=Path("results/regression"))
    p.add_argument("--target-col", type=str, default="pChEMBL")

    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--n-workers", type=int, default=None,
                    help="Parallel worker processes for featurization. Default: all available CPU cores.")
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--fp-no-chirality", action="store_true")

    p.add_argument("--top-n-final", type=int, default=200, help="Size of the top composite-score shortlist.")
    p.add_argument("--top-n-diverse", type=int, default=100, help="Size of the scaffold-diverse shortlist.")
    p.add_argument("--candidate-pool-size", type=int, default=None,
                    help="How many top-scoring compounds to keep as candidates for the two shortlists above "
                         "(memory-bounded -- the shortlists are built from this pool, not the full stream). "
                         "Default: max(5000, 5x top-n-final, 20x top-n-diverse).")
    p.add_argument("--weight-pred", type=float, default=0.80, help="Composite score weight on predicted pIC50.")
    p.add_argument("--weight-uncertainty", type=float, default=0.20,
                    help="Composite score weight on prediction certainty (1 - normalized PI width).")
    p.add_argument("--single-point-estimate", action="store_true",
                    help="Fit ONE regressor instead of three quantile models (lower/median/upper) -- roughly 3x "
                         "faster prediction on the active set, at the cost of losing the 90%% prediction interval "
                         "(ranking becomes pure predicted-pIC50 order, since there's no uncertainty term left).")

    p.add_argument("--model-dir", type=Path, default=None,
                    help="Directory to save/load the quantile models. Default: <output-dir>/models")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--no-plots", action="store_true")

    args = p.parse_args()
    if args.model_dir is None:
        args.model_dir = args.output_dir / "models"
    if args.candidate_pool_size is None:
        args.candidate_pool_size = max(5000, args.top_n_final * 5, args.top_n_diverse * 20)
    if abs(args.weight_pred + args.weight_uncertainty - 1.0) > 1e-6:
        raise ValueError("--weight-pred and --weight-uncertainty must sum to 1.0")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    fp_chirality = not args.fp_no_chirality

    all_scored_path = args.output_dir / "all_actives_regression_scored.parquet"

    # ----------------------------------------------------------------
    # STEP 1: load training data
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 60)

    train_df = pd.read_excel(args.train_file)
    X_cols = [c for c in train_df.columns if c not in TRAIN_META]
    X_train = train_df[X_cols].values
    y_train = train_df[args.target_col].values
    print(f"Training data: {train_df.shape} | pIC50 range: {y_train.min():.2f}-{y_train.max():.2f}")

    # ----------------------------------------------------------------
    # STEP 2: train (or load) quantile models
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN / LOAD QUANTILE REGRESSION MODELS")
    print("=" * 60)

    if args.single_point_estimate:
        print("--single-point-estimate: fitting ONE regressor (no quantile lower/upper) for ~3x faster prediction.")
        model_path = args.model_dir / "xgb_regressor_point.json"
        models = {}
        if args.skip_training and model_path.exists():
            m = xgb.XGBRegressor()
            m.load_model(model_path)
            models["median"] = m
            print(f"Loaded existing model from {model_path}")
        else:
            m = xgb.XGBRegressor(**BEST_PARAMS)
            m.fit(X_train, y_train)
            m.save_model(model_path)
            models["median"] = m
            print(f"Model fitted and saved to {model_path}")
        models["lower"] = models["upper"] = models["median"]
    else:
        quantile_paths = {q: args.model_dir / f"xgb_regressor_q_{q}.json" for q in ("lower", "median", "upper")}
        xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
        obj_name = "reg:quantileerror" if xgb_version >= (1, 7) else "reg:quantilereg"
        print(f"XGBoost {xgb.__version__} -> using objective='{obj_name}'")

        models = {}
        if args.skip_training and all(p.exists() for p in quantile_paths.values()):
            print("Loading existing quantile models...")
            for qname, path in quantile_paths.items():
                m = xgb.XGBRegressor()
                m.load_model(path)
                models[qname] = m
        else:
            print("Fitting quantile regression models (5th / 50th / 95th percentile)...")
            for q, qname in [(Q_LOW, "lower"), (0.5, "median"), (Q_HIGH, "upper")]:
                base = {k: v for k, v in BEST_PARAMS.items() if k not in ["objective", "reg_alpha", "reg_lambda"]}
                m = xgb.XGBRegressor(**{**base, "objective": obj_name, "quantile_alpha": q, "verbosity": 0})
                m.fit(X_train, y_train)
                m.save_model(quantile_paths[qname])
                models[qname] = m
                print(f"  Quantile q={q:.2f} fitted and saved to {quantile_paths[qname]}")

    # ----------------------------------------------------------------
    # STEP 3: establish composite-score normalization from the TRAINING
    # distribution (not the screening batch -- see module docstring)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: COMPOSITE SCORE NORMALIZATION RANGE")
    print("=" * 60)

    train_median = models["median"].predict(X_train)
    if args.single_point_estimate:
        train_lower = train_upper = train_median
    else:
        train_lower = models["lower"].predict(X_train)
        train_upper = models["upper"].predict(X_train)
    train_pi_width = np.maximum(train_upper, train_lower) - np.minimum(train_upper, train_lower)

    pred_min, pred_max = float(train_median.min()), float(train_median.max())
    pi_min, pi_max = float(train_pi_width.min()), float(train_pi_width.max())
    print(f"Predicted-pIC50 normalization range (from training): {pred_min:.2f}-{pred_max:.2f}")
    print(f"PI-width normalization range (from training):        {pi_min:.3f}-{pi_max:.3f}")
    print("Screening compounds outside these ranges are clipped to [0, 1] rather than allowed to skew the scale.")

    # ----------------------------------------------------------------
    # STEP 4: stream the predicted actives -- featurize, predict, score
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: STREAMING REGRESSION + SCORING")
    print("=" * 60)
    print(f"Screening file: {args.screening_file}")
    print(f"Candidate pool size for shortlists: {args.candidate_pool_size:,}")

    n_total = n_feature_failed = 0
    pic50_hist = OnlineHistogram(bins=30, value_range=(pred_min - 1, pred_max + 1))
    score_hist = OnlineHistogram(bins=30, value_range=(0.0, 1.0))
    candidate_pool = TopKTracker(args.candidate_pool_size)
    scaffold_seen_counts: dict = {}
    t_start = time.time()

    with ParquetChunkWriter(all_scored_path) as scored_writer, \
            ProcessPoolExecutor(
                max_workers=args.n_workers,
                initializer=init_featurize_only_worker,
                initargs=(args.fp_radius, args.fp_size, fp_chirality),
            ) as executor:

        for chunk_idx, chunk in enumerate(iter_parquet_chunks(args.screening_file, args.chunk_size)):
            canonical_list = chunk["canonical_smiles"].tolist()
            feat_results = list(executor.map(
                featurize_for_regression_one, canonical_list, chunksize=map_chunksize(len(canonical_list), args.n_workers)
            ))

            keep_mask = np.array([r is not None for r in feat_results])
            n_feature_failed += int((~keep_mask).sum())
            if not keep_mask.any():
                continue
            chunk = chunk[keep_mask].reset_index(drop=True)
            feat_results = [r for r in feat_results if r is not None]

            desc_df = pd.DataFrame([r[0] for r in feat_results], columns=DESCRIPTOR_NAMES)
            fp_df = pd.DataFrame([r[1] for r in feat_results], columns=[f"MF{i + 1}" for i in range(args.fp_size)])
            scaffolds = [r[2] for r in feat_results]

            X_chunk = pd.concat([desc_df, fp_df], axis=1).reindex(columns=X_cols, fill_value=0)

            median = models["median"].predict(X_chunk.values)
            if args.single_point_estimate:
                pred_lower = pred_upper = median
            else:
                lower = models["lower"].predict(X_chunk.values)
                upper = models["upper"].predict(X_chunk.values)
                pred_lower = np.minimum(lower, upper)
                pred_upper = np.maximum(lower, upper)
            pi_width = pred_upper - pred_lower

            pred_norm = np.clip((median - pred_min) / (pred_max - pred_min + 1e-10), 0, 1)
            uncert_norm = np.clip(1 - (pi_width - pi_min) / (pi_max - pi_min + 1e-10), 0, 1)
            composite_score = args.weight_pred * pred_norm + args.weight_uncertainty * uncert_norm

            result_df = pd.DataFrame({
                "name": chunk["name"].values,
                "id": chunk["id"].values,
                "canonical_smiles": chunk["canonical_smiles"].values,
                "active_probability": chunk.get("active_probability", pd.Series([None] * len(chunk))).values,
                "max_tanimoto_to_train": chunk.get("max_tanimoto_to_train", pd.Series([None] * len(chunk))).values,
                "inside_AD": chunk.get("inside_AD", pd.Series([None] * len(chunk))).values,
                "predicted_pIC50": np.round(median, 4),
                "pred_pIC50_lower": np.round(pred_lower, 4),
                "pred_pIC50_upper": np.round(pred_upper, 4),
                "PI_width": np.round(pi_width, 4),
                "composite_score": np.round(composite_score, 4),
                "murcko_scaffold": scaffolds,
                "activity_category": [activity_category(p) for p in median],
            })

            scored_writer.write(result_df)

            n_total += len(result_df)
            pic50_hist.update(median)
            score_hist.update(composite_score)
            for _, row in result_df.iterrows():
                candidate_pool.offer(row["composite_score"], row.to_dict())
                scaffold_seen_counts[row["murcko_scaffold"]] = scaffold_seen_counts.get(row["murcko_scaffold"], 0) + 1

            elapsed = time.time() - t_start
            rate = n_total / elapsed if elapsed > 0 else 0
            print(f"  chunk {chunk_idx + 1}: {n_total:,} scored | {rate:,.0f} compounds/sec")

    elapsed = time.time() - t_start

    # ----------------------------------------------------------------
    # STEP 5: build shortlists from the bounded candidate pool
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: BUILDING SHORTLISTS")
    print("=" * 60)

    pool_sorted = candidate_pool.get_sorted()
    if len(pool_sorted) < args.candidate_pool_size and len(pool_sorted) > 0:
        print(f"Note: only {len(pool_sorted):,} scored compounds total -- using all of them as the candidate pool.")

    top_final = pd.DataFrame(pool_sorted[: args.top_n_final])
    top_final_path = args.output_dir / f"top{args.top_n_final}_regression_candidates.xlsx"
    if len(top_final) > 0:
        top_final.to_excel(top_final_path, index=False)
    print(f"Top {len(top_final)} (composite score) saved: {top_final_path}")

    seen_scaffolds = set()
    diverse_rows = []
    for row in pool_sorted:
        scaffold = row["murcko_scaffold"]
        if scaffold == "no_scaffold":
            diverse_rows.append(row)
        elif scaffold not in seen_scaffolds:
            seen_scaffolds.add(scaffold)
            diverse_rows.append(row)
        if len(diverse_rows) >= args.top_n_diverse:
            break

    top_diverse = pd.DataFrame(diverse_rows)
    if len(top_diverse) > 0:
        top_diverse.insert(0, "diverse_rank", range(1, len(top_diverse) + 1))
    top_diverse_path = args.output_dir / f"top{args.top_n_diverse}_scaffold_diverse_for_docking.xlsx"
    if len(top_diverse) > 0:
        top_diverse.to_excel(top_diverse_path, index=False)
    print(f"Scaffold-diverse top {len(top_diverse)} saved: {top_diverse_path} "
          f"(from a candidate pool of {len(pool_sorted):,} top-scoring compounds -- "
          f"see --candidate-pool-size to search deeper)")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total actives scored:    {n_total:,}")
    print(f"Feature extraction failures: {n_feature_failed:,}")
    print(f"Unique scaffolds seen:   {len(scaffold_seen_counts):,}")
    print(f"Time elapsed:            {elapsed / 60:.1f} min ({n_total / elapsed:,.0f} compounds/sec)" if elapsed else "")
    print("\nOutput files:")
    print(f"  All actives scored (Parquet): {all_scored_path}")
    print(f"  Top {args.top_n_final} shortlist (Excel):        {top_final_path}")
    print(f"  Scaffold-diverse shortlist (Excel): {top_diverse_path}")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        _plot_overview(args.output_dir, pic50_hist, score_hist)
        _plot_top50_ranked(args.output_dir, top_final)
        _plot_scaffold_summary(args.output_dir, scaffold_seen_counts, top_diverse, n_total)

    print("===== DONE =====")


# ----------------------------------------------------------------------
# Plotting helpers -- built from bounded aggregates / small shortlists
# ----------------------------------------------------------------------

def _plot_overview(out_dir, pic50_hist: OnlineHistogram, score_hist: OnlineHistogram):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    centers = pic50_hist.bin_centers()
    width = (pic50_hist.edges[1] - pic50_hist.edges[0]) * 0.9
    ax.bar(centers, pic50_hist.counts, width=width, color="#3498db", alpha=0.8)
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Count")
    ax.set_title("Predicted pIC50 distribution (all actives)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    centers = score_hist.bin_centers()
    width = (score_hist.edges[1] - score_hist.edges[0]) * 0.9
    ax.bar(centers, score_hist.counts, width=width, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("Composite score")
    ax.set_ylabel("Count")
    ax.set_title("Composite score distribution (all actives)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Regression Screening Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "regression_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Overview plot saved.")


def _plot_top50_ranked(out_dir, top_final: pd.DataFrame):
    top50_plot = top_final.head(50).copy()
    if len(top50_plot) == 0:
        print("  No candidates -- skipping top-50 plot.")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(top50_plot) * 0.2)))
    y_pos = np.arange(len(top50_plot))
    pred_rev = top50_plot["predicted_pIC50"].values[::-1]
    low_rev = top50_plot["pred_pIC50_lower"].values[::-1]
    high_rev = top50_plot["pred_pIC50_upper"].values[::-1]
    names_rev = top50_plot["name"].values[::-1]
    cat_rev = top50_plot["activity_category"].values[::-1]
    bar_cols = [CAT_COLORS.get(c, "#3498db") for c in cat_rev]

    ax.barh(y_pos, pred_rev, color=bar_cols, alpha=0.85, height=0.7)
    xerr_lo = np.maximum(0, pred_rev - low_rev)
    xerr_hi = np.maximum(0, high_rev - pred_rev)
    ax.errorbar(pred_rev, y_pos, xerr=[xerr_lo, xerr_hi], fmt="none", color="black", capsize=3, alpha=0.6,
                 label="90% prediction interval")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(n)[:35] for n in names_rev], fontsize=7)
    ax.set_xlabel("Predicted pIC50")
    ax.set_title("Top candidates -- predicted pIC50 with uncertainty", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(max(0, low_rev.min() - 0.3), high_rev.max() + 0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "top_candidates_ranked.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Top candidates plot saved.")


def _plot_scaffold_summary(out_dir, scaffold_seen_counts: dict, top_diverse: pd.DataFrame, n_total: int):
    if len(top_diverse) == 0:
        print("  No diverse shortlist -- skipping scaffold summary plot.")
        return

    counts = pd.Series(scaffold_seen_counts)
    freq_bins = [1, 2, 3, 4, 6, 11, 51, 10**9]  # 8 edges -> 7 bins, matching the 7 labels below
    freq_labels = ["1", "2", "3", "4-5", "6-10", "11-50", ">50"]
    freq_vals = [((counts >= freq_bins[i]) & (counts < freq_bins[i + 1])).sum() for i in range(len(freq_bins) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(freq_labels, freq_vals, color="#3498db", alpha=0.8)
    for bar, val in zip(bars, freq_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Compounds sharing a scaffold")
    ax.set_ylabel("Number of scaffolds")
    ax.set_title(f"Scaffold frequency ({len(counts):,} unique scaffolds among {n_total:,} actives)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    cat_counts = top_diverse["activity_category"].value_counts()
    colors_pie = [CAT_COLORS.get(c, "#95a5a6") for c in cat_counts.index]
    ax.pie(cat_counts.values, labels=[c.split(" (")[0] for c in cat_counts.index], colors=colors_pie,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Activity category (scaffold-diverse shortlist)", fontweight="bold")

    plt.suptitle("Scaffold Diversity Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "scaffold_diversity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Scaffold diversity plot saved.")


if __name__ == "__main__":
    main()
