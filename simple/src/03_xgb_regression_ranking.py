#!/usr/bin/env python3
"""
03_xgb_regression_ranking.py  (SIMPLE pipeline)
=================================================
Stage 3 (final stage) of the sEH QSAR screening pipeline -- the
small-library version.

Trains an XGBoost regressor to predict pIC50 for the compounds the
classifier (stage 2) flagged as active, estimates per-compound uncertainty
with XGBoost quantile regression, combines predicted potency and certainty
into a composite score, and produces both a top-N ranked list and a
scaffold-diverse subset (one compound per Murcko scaffold) suitable for
docking follow-up. Everything is done in memory -- fine for the active
subset a normal-sized screen produces. For a very large active set (e.g.
screening ZINC-scale libraries), use the large_scale/ pipeline instead.

Default paths (override any of them with flags -- see --help):
    --train-file      data/data_for_regression.xlsx
    --screening-file  results/actives_with_features_for_regression.xlsx
    --output-dir      results/regression/

Example (using all defaults, run from the project root, after stage 2)
-------------------------------------------------------------------------
    python src/03_xgb_regression_ranking.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from qsar_utils import activity_category, find_column, get_murcko_scaffold  # noqa: E402

warnings.filterwarnings("ignore")

TRAIN_META = [
    "ChEMBL_ID", "Smiles", "smiles", "canonical_smiles", "pChEMBL",
    "Label_binary", "Label_low_high", "Label_3class", "Regression",
]
SCREEN_META = [
    "name", "id", "smiles", "canonical_smiles",
    "active_probability", "confidence_score", "max_tanimoto_to_train", "inside_AD",
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
        description="Train/apply the XGBoost pIC50 regressor and rank screening actives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file", type=Path, default=Path("data/data_for_regression.xlsx"))
    p.add_argument("--screening-file", type=Path,
                    default=Path("results/actives_with_features_for_regression.xlsx"))
    p.add_argument("--output-dir", type=Path, default=Path("results/regression"))
    p.add_argument("--target-col", type=str, default="pChEMBL")

    p.add_argument("--top-n-final", type=int, default=200, help="Size of the top composite-score shortlist.")
    p.add_argument("--top-n-diverse", type=int, default=100, help="Size of the scaffold-diverse shortlist.")
    p.add_argument("--weight-pred", type=float, default=0.80, help="Composite score weight on predicted pIC50.")
    p.add_argument("--weight-uncertainty", type=float, default=0.20,
                    help="Composite score weight on prediction certainty (1 - normalized PI width).")

    p.add_argument("--model-dir", type=Path, default=None,
                    help="Directory to save/load the quantile models. Default: <output-dir>/models")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--no-plots", action="store_true")

    args = p.parse_args()
    if args.model_dir is None:
        args.model_dir = args.output_dir / "models"
    if abs(args.weight_pred + args.weight_uncertainty - 1.0) > 1e-6:
        raise ValueError("--weight-pred and --weight-uncertainty must sum to 1.0")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

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
    # STEP 2: load screening data (predicted actives from stage 2)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: LOADING SCREENING DATA (predicted actives)")
    print("=" * 60)

    screen_df = pd.read_excel(args.screening_file)
    print(f"Screening data: {screen_df.shape}")

    meta_cols_present = [c for c in SCREEN_META if c in screen_df.columns]
    meta_df = screen_df[meta_cols_present].copy()
    X_screen = screen_df[[c for c in X_cols if c in screen_df.columns]].copy()
    X_screen = X_screen.reindex(columns=X_cols, fill_value=0)
    assert X_screen.shape[1] == len(X_cols), "Feature mismatch between training and screening!"
    print("Feature alignment: OK")

    # ----------------------------------------------------------------
    # STEP 3: internal CV (printed only)
    # ----------------------------------------------------------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict

    print("\n" + "=" * 60)
    print("STEP 3: INTERNAL CV VALIDATION (5-fold)")
    print("=" * 60)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(xgb.XGBRegressor(**BEST_PARAMS), X_train, y_train, cv=cv)
    cv_r2 = r2_score(y_train, y_pred_cv)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    cv_mae = mean_absolute_error(y_train, y_pred_cv)
    print(f"R2={cv_r2:.4f}  RMSE={cv_rmse:.4f}  MAE={cv_mae:.4f}")

    # ----------------------------------------------------------------
    # STEP 4: train/load quantile models
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN / LOAD REGRESSION MODELS")
    print("=" * 60)

    quantile_paths = {q: args.model_dir / f"xgb_regressor_q_{q}.json" for q in ("lower", "median", "upper")}
    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    obj_name = "reg:quantileerror" if xgb_version >= (1, 7) else "reg:quantilereg"
    print(f"XGBoost {xgb.__version__} -> using objective='{obj_name}'")

    quantile_preds = {}
    if args.skip_training and all(p.exists() for p in quantile_paths.values()):
        print("Loading existing quantile models...")
        for qname, path in quantile_paths.items():
            m = xgb.XGBRegressor()
            m.load_model(path)
            quantile_preds[qname] = m.predict(X_screen.values)
    else:
        print("Fitting quantile regression models (5th / 50th / 95th percentile)...")
        for q, qname in [(Q_LOW, "lower"), (0.5, "median"), (Q_HIGH, "upper")]:
            base = {k: v for k, v in BEST_PARAMS.items() if k not in ["objective", "reg_alpha", "reg_lambda"]}
            m = xgb.XGBRegressor(**{**base, "objective": obj_name, "quantile_alpha": q, "verbosity": 0})
            m.fit(X_train, y_train)
            m.save_model(quantile_paths[qname])
            quantile_preds[qname] = m.predict(X_screen.values)
            print(f"  Quantile q={q:.2f} fitted and saved to {quantile_paths[qname]}")

    y_pred_lower = np.minimum(quantile_preds["lower"], quantile_preds["upper"])
    y_pred_upper = np.maximum(quantile_preds["lower"], quantile_preds["upper"])
    y_pred_screen = quantile_preds["median"]
    pi_width = y_pred_upper - y_pred_lower
    print(f"\nScreening predictions: pIC50 {y_pred_screen.min():.2f}-{y_pred_screen.max():.2f} | "
          f"PI width mean {pi_width.mean():.3f}")

    # ----------------------------------------------------------------
    # STEP 5: composite scoring
    # ----------------------------------------------------------------
    pred_min, pred_max = y_pred_screen.min(), y_pred_screen.max()
    pred_norm = (y_pred_screen - pred_min) / (pred_max - pred_min + 1e-10)
    pi_min, pi_max = pi_width.min(), pi_width.max()
    uncert_norm = 1 - ((pi_width - pi_min) / (pi_max - pi_min + 1e-10))
    composite_score = args.weight_pred * pred_norm + args.weight_uncertainty * uncert_norm

    # ----------------------------------------------------------------
    # STEP 6: Murcko scaffolds
    # ----------------------------------------------------------------
    smiles_col = find_column(meta_df.columns, ["canonical_smiles", "smiles"])
    scaffolds = [get_murcko_scaffold(smi) for smi in meta_df[smiles_col]]
    print(f"\nUnique Murcko scaffolds: {len(set(scaffolds))} from {len(scaffolds)} compounds")

    # ----------------------------------------------------------------
    # STEP 7: build results
    # ----------------------------------------------------------------
    results_df = meta_df.copy()
    results_df["predicted_pIC50"] = np.round(y_pred_screen, 4)
    results_df["pred_pIC50_lower"] = np.round(y_pred_lower, 4)
    results_df["pred_pIC50_upper"] = np.round(y_pred_upper, 4)
    results_df["PI_width"] = np.round(pi_width, 4)
    results_df["composite_score"] = np.round(composite_score, 4)
    results_df["murcko_scaffold"] = scaffolds
    results_df["activity_category"] = [activity_category(p) for p in y_pred_screen]
    results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    top_final = results_df.head(args.top_n_final).copy()
    top_final_path = args.output_dir / f"top{args.top_n_final}_regression_candidates.xlsx"
    top_final.to_excel(top_final_path, index=False)

    seen_scaffolds, diverse_indices = set(), []
    for idx, row in results_df.iterrows():
        scaffold = row["murcko_scaffold"]
        if scaffold == "no_scaffold":
            diverse_indices.append(idx)
        elif scaffold not in seen_scaffolds:
            seen_scaffolds.add(scaffold)
            diverse_indices.append(idx)
        if len(diverse_indices) >= args.top_n_diverse:
            break
    top_diverse = results_df.loc[diverse_indices].reset_index(drop=True)
    top_diverse.insert(0, "diverse_rank", range(1, len(top_diverse) + 1))
    top_diverse_path = args.output_dir / f"top{args.top_n_diverse}_scaffold_diverse_for_docking.xlsx"
    top_diverse.to_excel(top_diverse_path, index=False)

    all_path = args.output_dir / "all_actives_regression_scored.xlsx"
    results_df.to_excel(all_path, index=False)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        _plot_overview(args.output_dir, results_df, top_final, top_diverse)
        _plot_top50_ranked(args.output_dir, top_final)
        _plot_scaffold_diversity(args.output_dir, results_df, top_final, top_diverse)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Actives scored: {len(results_df)}")
    print(f"Internal CV: R2={cv_r2:.3f}  RMSE={cv_rmse:.3f}  MAE={cv_mae:.3f}")
    print(f"Top {args.top_n_final}: pIC50 {top_final['predicted_pIC50'].min():.2f}-{top_final['predicted_pIC50'].max():.2f}")
    print(f"Diverse {args.top_n_diverse}: {top_diverse['murcko_scaffold'].nunique()} unique scaffolds")
    print("\nOutput files:")
    print(f"  All actives scored:  {all_path}")
    print(f"  Top {args.top_n_final}:             {top_final_path}")
    print(f"  Scaffold-diverse:    {top_diverse_path}")
    print("===== DONE =====")


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def _plot_overview(out_dir, results_df, top_final, top_diverse):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for cat, col in CAT_COLORS.items():
        mask = results_df["activity_category"] == cat
        if mask.sum() > 0:
            ax.hist(results_df.loc[mask, "predicted_pIC50"], bins=20, alpha=0.75, color=col,
                     label=f"{cat.split(' (')[0]} (n={mask.sum()})")
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Count")
    ax.set_title("Predicted pIC50 distribution", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(results_df["composite_score"], bins=30, color="#3498db", alpha=0.8)
    ax.axvline(x=top_final["composite_score"].min(), color="red", linestyle="--",
                label=f"Top-N cutoff ({top_final['composite_score'].min():.3f})")
    ax.axvline(x=top_diverse["composite_score"].min(), color="orange", linestyle=":",
                label=f"Diverse-subset min ({top_diverse['composite_score'].min():.3f})")
    ax.set_xlabel("Composite score")
    ax.set_ylabel("Count")
    ax.set_title("Composite score distribution", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Regression Screening Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "regression_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Overview plot saved.")


def _plot_top50_ranked(out_dir, top_final):
    top50_plot = top_final.head(50).copy()
    if len(top50_plot) == 0:
        print("No candidates -- skipping top-50 plot.")
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
    print("Top candidates plot saved.")


def _plot_scaffold_diversity(out_dir, results_df, top_final, top_diverse):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(results_df["predicted_pIC50"], bins=25, alpha=0.6, color="#3498db",
            label=f"All actives (n={len(results_df)})", density=True)
    ax.hist(top_diverse["predicted_pIC50"], bins=20, alpha=0.7, color="#e74c3c",
            label=f"Scaffold-diverse subset (n={len(top_diverse)})", density=True)
    ax.hist(top_final["predicted_pIC50"], bins=20, alpha=0.5, color="#2ecc71",
            label=f"Top-N (n={len(top_final)})", density=True)
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Scaffold diversity check\n{results_df['murcko_scaffold'].nunique()} unique scaffolds among all actives",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scaffold_diversity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Scaffold diversity plot saved.")


if __name__ == "__main__":
    main()
