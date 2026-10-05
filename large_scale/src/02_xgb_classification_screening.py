#!/usr/bin/env python3
"""
02_xgb_classification_screening.py
===================================
Stage 2 of the sEH QSAR screening pipeline.

Trains an XGBoost classifier on your labeled training data (small -- a few
thousand rows, loaded fully in memory as usual), then STREAMS the screening
library from stage 1 through in chunks: for each chunk, it computes
descriptors + Morgan fingerprints on the fly, predicts, checks the
applicability domain, writes out compact result columns, and immediately
discards the feature matrix before moving to the next chunk. This is what
makes screening hundreds of millions of compounds possible -- the full
feature matrix for a library that size would be many terabytes; the
per-chunk matrix is a few hundred MB and is never accumulated.

Default paths (override any of them with flags -- see --help):
    --train-file      data/data_for_classification.xlsx
    --screening-file  results/screening_library_standardized.parquet   (output of stage 1)
    --output-dir      results/

Example (using all defaults, run from the project root, after stage 1)
-------------------------------------------------------------------------
    python src/02_xgb_classification_screening.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.append(str(Path(__file__).resolve().parent))
from qsar_utils import (  # noqa: E402
    DESCRIPTOR_NAMES,
    OnlineHistogram,
    ParquetChunkWriter,
    TopKTracker,
    applicability_domain_summary,
    confidence_category,
    featurize_and_score_one,
    featurize_only_one,
    find_column,
    init_featurize_only_worker,
    init_featurize_worker,
    iter_parquet_chunks,
    map_chunksize,
    standardize_mol,
)

warnings.filterwarnings("ignore")

TRAIN_META = [
    "ChEMBL_ID", "Smiles", "smiles", "canonical_smiles", "pChEMBL",
    "Label_binary", "Label_low_high", "Label_3class", "Regression",
]

BEST_PARAMS = {
    "subsample": 0.9,
    "n_estimators": 500,
    "min_child_weight": 1,
    "max_depth": 3,
    "learning_rate": 0.1,
    "gamma": 0.2,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8,
    "verbosity": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/apply the XGBoost sEH activity classifier, streaming over the screening library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file", type=Path, default=Path("data/data_for_classification.xlsx"))
    p.add_argument("--screening-file", type=Path, default=Path("results/screening_library_standardized.parquet"),
                    help="Output of 01_build_screening_library.py.")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--label-col", type=str, default="Label_low_high")

    p.add_argument("--chunk-size", type=int, default=100_000,
                    help="Compounds featurized/predicted at a time. Lower this if you run out of memory.")
    p.add_argument("--n-workers", type=int, default=None,
                    help="Parallel worker processes for featurization. Default: all available CPU cores.")

    p.add_argument("--tanimoto-threshold", type=float, default=0.4,
                    help="Below this max-Tanimoto-to-training similarity, a compound is flagged as outside the AD.")
    p.add_argument("--ad-max-train-refs", type=int, default=1000,
                    help="Cap on how many training compounds are used as the applicability-domain reference set. "
                         "At huge screening scale, comparing every compound against thousands of training "
                         "fingerprints dominates runtime; a random subsample of this size is used instead if the "
                         "training set is bigger. Set to 0 to always use the full training set.")
    p.add_argument("--skip-ad", action="store_true",
                    help="Skip the applicability-domain (Tanimoto) check entirely -- fastest option, but you lose "
                         "the flag for which predictions are on chemically novel scaffolds. Worth trying if AD "
                         "computation is a meaningful share of your runtime at very large scale.")
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--fp-no-chirality", action="store_true")
    p.add_argument("--fp-col-prefix", type=str, default="MF")

    p.add_argument("--top-n-report", type=int, default=500,
                    help="How many top predicted actives to keep for the summary plot/list (memory-bounded).")

    p.add_argument("--model-path", type=Path, default=None,
                    help="Path to save/load the trained model (JSON). Default: <output-dir>/xgb_classifier.json")
    p.add_argument("--skip-training", action="store_true",
                    help="Load an existing model from --model-path instead of retraining.")
    p.add_argument("--no-plots", action="store_true")

    args = p.parse_args()
    if args.model_path is None:
        args.model_path = args.output_dir / "xgb_classifier.json"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp_chirality = not args.fp_no_chirality

    predictions_path = args.output_dir / "screening_predictions_all.parquet"
    actives_path = args.output_dir / "actives_for_regression.parquet"

    # ----------------------------------------------------------------
    # STEP 1: load training data + train (or load) the model
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: TRAINING DATA + MODEL")
    print("=" * 60)

    train_df = pd.read_excel(args.train_file)
    print(f"Training data: {train_df.shape}")

    X_cols = [c for c in train_df.columns if c not in TRAIN_META]
    X_train = train_df[X_cols].values
    y_train = train_df[args.label_col].values

    neg_count, pos_count = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"Class balance: {neg_count} inactive / {pos_count} active")
    BEST_PARAMS["scale_pos_weight"] = neg_count / pos_count

    model = xgb.XGBClassifier(**BEST_PARAMS)
    if args.skip_training and args.model_path.exists():
        model.load_model(args.model_path)
        print(f"Loaded existing model from {args.model_path}")
    else:
        model.fit(X_train, y_train)
        model.save_model(args.model_path)
        print(f"Model trained and saved to {args.model_path}")

    print("\nRunning 5-fold stratified cross-validation (honest performance estimate)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(xgb.XGBClassifier(**BEST_PARAMS), X_train, y_train, cv=cv)
    y_proba_cv = cross_val_predict(xgb.XGBClassifier(**BEST_PARAMS), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    cv_acc = accuracy_score(y_train, y_pred_cv)
    cv_auc = roc_auc_score(y_train, y_proba_cv)
    cv_mcc = matthews_corrcoef(y_train, y_pred_cv)
    cv_f1 = f1_score(y_train, y_pred_cv)
    cv_bal = balanced_accuracy_score(y_train, y_pred_cv)
    print(f"  AUC={cv_auc:.3f}  MCC={cv_mcc:.3f}  Acc={cv_acc:.3f}  F1={cv_f1:.3f}  BalAcc={cv_bal:.3f}")

    # ----------------------------------------------------------------
    # STEP 2: build the applicability-domain reference set (unless skipped)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: APPLICABILITY DOMAIN REFERENCE SET")
    print("=" * 60)

    if args.skip_ad:
        ad_refs = []
        print("Skipping the applicability-domain check (--skip-ad) -- no reference fingerprints needed.")
    else:
        train_smiles_col = find_column(train_df.columns, ["canonical_smiles", "smiles", "Smiles"])
        train_canonical = []
        for smi in train_df[train_smiles_col]:
            if pd.isna(smi):
                continue
            _, canon, _ = standardize_mol(str(smi))
            if canon is not None:
                train_canonical.append(canon)

        if args.ad_max_train_refs and len(train_canonical) > args.ad_max_train_refs:
            rng = random.Random(42)
            ad_refs = rng.sample(train_canonical, args.ad_max_train_refs)
            print(f"Training set has {len(train_canonical)} compounds; using a random {args.ad_max_train_refs} "
                  f"as the AD reference set for speed (override with --ad-max-train-refs).")
        else:
            ad_refs = train_canonical
            print(f"Using all {len(ad_refs)} training compounds as the AD reference set.")

    # ----------------------------------------------------------------
    # STEP 3: stream the screening library -- featurize, predict, discard
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: STREAMING SCREENING + PREDICTION")
    print("=" * 60)
    print(f"Screening file: {args.screening_file}")
    print(f"Chunk size: {args.chunk_size:,} | Workers: {args.n_workers or 'all CPU cores'}")

    n_total = n_active = n_inactive = n_inside_ad = n_feature_failed = 0
    conf_high = conf_medium = conf_low = 0
    tanimoto_hist = OnlineHistogram(bins=30, value_range=(0.0, 1.0))
    top_actives = TopKTracker(args.top_n_report)
    t_start = time.time()

    if args.skip_ad:
        worker_fn = featurize_only_one
        worker_init = init_featurize_only_worker
        worker_initargs = (args.fp_radius, args.fp_size, fp_chirality)
    else:
        worker_fn = featurize_and_score_one
        worker_init = init_featurize_worker
        worker_initargs = (ad_refs, args.fp_radius, args.fp_size, fp_chirality)

    with ParquetChunkWriter(predictions_path) as pred_writer, \
            ParquetChunkWriter(actives_path) as actives_writer, \
            ProcessPoolExecutor(
                max_workers=args.n_workers,
                initializer=worker_init,
                initargs=worker_initargs,
            ) as executor:

        for chunk_idx, chunk in enumerate(iter_parquet_chunks(args.screening_file, args.chunk_size)):
            canonical_list = chunk["canonical_smiles"].tolist()
            feat_results = list(executor.map(
                worker_fn, canonical_list, chunksize=map_chunksize(len(canonical_list), args.n_workers)
            ))

            keep_mask = np.array([r is not None for r in feat_results])
            n_feature_failed += int((~keep_mask).sum())
            if not keep_mask.any():
                continue
            chunk = chunk[keep_mask].reset_index(drop=True)
            feat_results = [r for r in feat_results if r is not None]

            desc_df = pd.DataFrame([r[0] for r in feat_results], columns=DESCRIPTOR_NAMES)
            fp_df = pd.DataFrame([r[1] for r in feat_results],
                                  columns=[f"{args.fp_col_prefix}{i + 1}" for i in range(args.fp_size)])
            if args.skip_ad:
                max_tanimoto = np.full(len(feat_results), np.nan)
                inside_ad = np.full(len(feat_results), None)  # AD wasn't computed; column carried through as empty
            else:
                max_tanimoto = np.array([r[2] for r in feat_results])
                inside_ad = max_tanimoto >= args.tanimoto_threshold

            X_chunk = pd.concat([desc_df, fp_df], axis=1).reindex(columns=X_cols, fill_value=0)
            y_proba = model.predict_proba(X_chunk.values)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            confidence = np.maximum(y_proba, 1 - y_proba)

            result_df = pd.DataFrame({
                "name": chunk["name"].values,
                "id": chunk["id"].values,
                "smiles": chunk["smiles"].values,
                "canonical_smiles": chunk["canonical_smiles"].values,
                "predicted_class": y_pred,
                "predicted_label": np.where(y_pred == 1, "Active", "Inactive"),
                "active_probability": np.round(y_proba, 4),
                "confidence_score": np.round(confidence, 4),
                "confidence_category": [confidence_category(c) for c in confidence],
                "max_tanimoto_to_train": np.round(max_tanimoto, 4),
                "inside_AD": inside_ad,
            })

            pred_writer.write(result_df)

            actives_chunk = result_df[result_df["predicted_class"] == 1]
            if len(actives_chunk) > 0:
                actives_writer.write(actives_chunk[[
                    "name", "id", "smiles", "canonical_smiles",
                    "active_probability", "confidence_score", "max_tanimoto_to_train", "inside_AD",
                ]])

            # ---- update running aggregates (bounded memory regardless of library size) ----
            n_total += len(result_df)
            n_active += int((y_pred == 1).sum())
            n_inactive += int((y_pred == 0).sum())
            conf_high += int((confidence >= 0.8).sum())
            conf_medium += int(((confidence >= 0.5) & (confidence < 0.8)).sum())
            conf_low += int((confidence < 0.5).sum())
            if not args.skip_ad:
                n_inside_ad += int(inside_ad.sum())
                tanimoto_hist.update(max_tanimoto)
            for _, row in actives_chunk.iterrows():
                top_actives.offer(row["active_probability"], row.to_dict())

            elapsed = time.time() - t_start
            rate = n_total / elapsed if elapsed > 0 else 0
            ad_note = f"{n_inside_ad:,} inside AD | " if not args.skip_ad else ""
            print(f"  chunk {chunk_idx + 1}: {n_total:,} processed | {n_active:,} active | "
                  f"{ad_note}{rate:,.0f} compounds/sec")

    elapsed = time.time() - t_start

    # ----------------------------------------------------------------
    # Summary + AD warning
    # ----------------------------------------------------------------
    ad = None if args.skip_ad else applicability_domain_summary(n_total, n_inside_ad, args.tanimoto_threshold)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total screened:        {n_total:,}")
    print(f"Predicted active:      {n_active:,} ({n_active / n_total * 100:.2f}%)" if n_total else "")
    print(f"Predicted inactive:    {n_inactive:,}")
    print(f"Feature extraction failures: {n_feature_failed:,}")
    print(f"Time elapsed:          {elapsed / 60:.1f} min ({n_total / elapsed:,.0f} compounds/sec)" if elapsed else "")
    if ad is not None:
        print(f"\nApplicability domain (threshold={args.tanimoto_threshold}):")
        print(f"  Inside AD:  {ad['n_inside']:,} ({(1 - ad['frac_outside']) * 100:.1f}%)")
        print(f"  Outside AD: {ad['n_outside']:,} ({ad['frac_outside'] * 100:.1f}%)")
        if ad["warning"]:
            print(f"\n  *** {ad['warning']} ***")
    else:
        print("\nApplicability domain check was skipped (--skip-ad).")
    print(f"\nInternal CV performance (from training set): AUC={cv_auc:.3f}  MCC={cv_mcc:.3f}  Acc={cv_acc:.3f}  F1={cv_f1:.3f}")

    # ----------------------------------------------------------------
    # Save the top predicted actives to Excel for a quick manual look
    # ----------------------------------------------------------------
    top_rows = top_actives.get_sorted()
    top_actives_xlsx_path = args.output_dir / "top_predicted_actives.xlsx"
    if top_rows:
        top_df_full = pd.DataFrame(top_rows)
        if args.skip_ad:
            top_df_full = top_df_full.drop(columns=["max_tanimoto_to_train", "inside_AD"], errors="ignore")
        top_df_full.to_excel(top_actives_xlsx_path, index=False)
        print(f"\nTop {len(top_df_full)} predicted actives (of {args.top_n_report:,} tracked) saved to Excel: "
              f"{top_actives_xlsx_path}")
    else:
        print("\nNo predicted actives -- no Excel shortlist to save.")

    print("\nOutput files:")
    print(f"  All predictions (Parquet):        {predictions_path}")
    print(f"  Actives only, for stage 3 (Parquet): {actives_path}")
    print(f"  Top predicted actives (Excel):    {top_actives_xlsx_path}")

    # ----------------------------------------------------------------
    # Plots (built from the bounded aggregates above -- no full-size arrays)
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        _plot_overview(args.output_dir, n_active, n_inactive, conf_high, conf_medium, conf_low)
        if ad is not None:
            _plot_applicability_domain(args.output_dir, tanimoto_hist, args.tanimoto_threshold, ad)
        _plot_top_actives(args.output_dir, top_actives, args.skip_ad)

    print("===== DONE =====")


# ----------------------------------------------------------------------
# Plotting helpers -- all built from bounded aggregates, safe at any scale
# ----------------------------------------------------------------------

def _plot_overview(out_dir, n_active, n_inactive, conf_high, conf_medium, conf_low):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.pie([n_inactive, n_active],
           labels=[f"Inactive\n(n={n_inactive:,})", f"Active\n(n={n_active:,})"],
           colors=["#5B9BD5", "#E05C5C"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Predicted class distribution", fontweight="bold")

    ax = axes[1]
    total_conf = max(conf_high + conf_medium + conf_low, 1)
    bars = ax.bar(["High\n(>=0.8)", "Medium\n(0.5-0.8)", "Low\n(<0.5)"],
                   [conf_high, conf_medium, conf_low],
                   color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.85)
    for bar, val in zip(bars, [conf_high, conf_medium, conf_low]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f"n={val:,}\n({val / total_conf * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Prediction confidence", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Screening Results Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "screening_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Overview plot saved.")


def _plot_applicability_domain(out_dir, tanimoto_hist: OnlineHistogram, threshold, ad):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    centers = tanimoto_hist.bin_centers()
    width = (tanimoto_hist.edges[1] - tanimoto_hist.edges[0]) * 0.9
    ax.bar(centers, tanimoto_hist.counts, width=width, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5, label=f"AD threshold ({threshold})")
    ax.set_xlabel("Max Tanimoto similarity to nearest training compound")
    ax.set_ylabel("Number of compounds")
    ax.set_title(
        f"Applicability Domain\n"
        f"{ad['n_inside']:,} inside AD ({(1 - ad['frac_outside']) * 100:.1f}%) | "
        f"{ad['n_outside']:,} outside AD ({ad['frac_outside'] * 100:.1f}%)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if ad["warning"]:
        ax.text(0.5, -0.22, "\n".join(_wrap(ad["warning"], 90)), transform=ax.transAxes,
                 ha="center", va="top", fontsize=8.5, color="darkred")

    plt.tight_layout()
    plt.savefig(out_dir / "applicability_domain.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Applicability domain plot saved.")


def _plot_top_actives(out_dir, top_actives: TopKTracker, skip_ad: bool = False):
    rows = top_actives.get_sorted()
    if not rows:
        print("  No predicted actives -- skipping top-actives plot.")
        return
    top_df = pd.DataFrame(rows).head(50)

    fig, ax = plt.subplots(figsize=(12, max(6, len(top_df) * 0.18)))
    color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
    bar_colors = [color_map.get(c, "#3498db") for c in top_df["confidence_category"]]

    ax.barh(range(len(top_df)), top_df["active_probability"].values[::-1], color=bar_colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_df)))
    if skip_ad:
        labels = [str(row["name"])[:30] for _, row in top_df.iloc[::-1].iterrows()]
    else:
        labels = [f"{str(row['name'])[:30]} ({'in-AD' if row['inside_AD'] else 'out-AD'})"
                  for _, row in top_df.iloc[::-1].iterrows()]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Active probability")
    ax.set_title(f"Top {len(top_df)} predicted actives (of top {top_actives.k:,} tracked)", fontweight="bold")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    patches = [mpatches.Patch(color=v, label=f"{k} confidence") for k, v in color_map.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "top_predicted_actives.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Top predicted actives plot saved.")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    main()
