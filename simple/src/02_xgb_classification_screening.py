#!/usr/bin/env python3
"""
02_xgb_classification_screening.py  (SIMPLE pipeline)
=======================================================
Stage 2 of the sEH QSAR screening pipeline -- the small-library version.

Trains an XGBoost classifier on your labeled training data, validates it
with 5-fold stratified CV, checks each screening compound's applicability
domain (max Tanimoto similarity to the training set), then scores and
ranks the screening library built by 01_build_screening_library.py.
Everything is done in memory -- fine up to roughly a few hundred thousand
compounds. For millions+ (e.g. ZINC), use the large_scale/ pipeline instead.

Default paths (override any of them with flags -- see --help):
    --train-file      data/data_for_classification.xlsx
    --screening-file  results/screening_library_for_ML.xlsx
    --output-dir      results/

Example (using all defaults, run from the project root, after stage 1)
-------------------------------------------------------------------------
    python src/02_xgb_classification_screening.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import DataStructs
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
    applicability_domain_summary,
    confidence_category,
    find_column,
    get_morgan_generator,
    smiles_to_fp,
)

warnings.filterwarnings("ignore")

TRAIN_META = [
    "ChEMBL_ID", "Smiles", "smiles", "canonical_smiles", "pChEMBL",
    "Label_binary", "Label_low_high", "Label_3class", "Regression",
]
SCREEN_META = ["name", "id", "smiles", "canonical_smiles"]

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
        description="Train/apply the XGBoost sEH activity classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file", type=Path, default=Path("data/data_for_classification.xlsx"))
    p.add_argument("--screening-file", type=Path, default=Path("results/screening_library_for_ML.xlsx"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--label-col", type=str, default="Label_low_high")

    p.add_argument("--tanimoto-threshold", type=float, default=0.4)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=1024)
    p.add_argument("--fp-no-chirality", action="store_true")

    p.add_argument("--top-n-actives", type=int, default=500,
                    help="How many top predicted actives to save to a separate Excel file for a quick manual look.")

    p.add_argument("--model-path", type=Path, default=None,
                    help="Path to save/load the trained model (JSON). Default: <output-dir>/xgb_classifier.json")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--no-plots", action="store_true")

    args = p.parse_args()
    if args.model_path is None:
        args.model_path = args.output_dir / "xgb_classifier.json"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp_chirality = not args.fp_no_chirality

    # ----------------------------------------------------------------
    # STEP 1: load training data
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 60)

    train_df = pd.read_excel(args.train_file)
    print(f"Training data: {train_df.shape}")

    X_cols = [c for c in train_df.columns if c not in TRAIN_META]
    X_train = train_df[X_cols].values
    y_train = train_df[args.label_col].values

    print(f"Features:      {len(X_cols)}")
    print(f"Class balance: {(y_train == 0).sum()} inactive / {(y_train == 1).sum()} active")

    neg_count, pos_count = (y_train == 0).sum(), (y_train == 1).sum()
    BEST_PARAMS["scale_pos_weight"] = neg_count / pos_count

    # ----------------------------------------------------------------
    # STEP 2: load screening library
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: LOADING SCREENING LIBRARY")
    print("=" * 60)

    screen_df = pd.read_excel(args.screening_file)
    print(f"Screening library: {screen_df.shape}")

    meta_df = screen_df[SCREEN_META].copy()
    X_screen = screen_df[[c for c in X_cols if c in screen_df.columns]].copy()
    X_screen = X_screen.reindex(columns=X_cols, fill_value=0)
    assert X_screen.shape[1] == len(X_cols), "Feature count mismatch between training and screening!"
    print("Feature alignment: OK")

    # ----------------------------------------------------------------
    # STEP 3: applicability domain
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: APPLICABILITY DOMAIN CHECK")
    print("=" * 60)

    generator = get_morgan_generator(args.fp_radius, args.fp_size, fp_chirality)
    train_smiles_col = find_column(train_df.columns, ["canonical_smiles", "smiles", "Smiles"])
    train_fps_valid = [
        fp for fp in (smiles_to_fp(smi, generator) for smi in train_df[train_smiles_col]) if fp is not None
    ]

    max_tanimoto = []
    for i, smi in enumerate(screen_df["canonical_smiles"]):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(screen_df)}")
        fp = smiles_to_fp(smi, generator)
        if fp is None:
            max_tanimoto.append(0.0)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps_valid)
        max_tanimoto.append(round(max(sims), 4))

    max_tanimoto = np.array(max_tanimoto)
    inside_ad = max_tanimoto >= args.tanimoto_threshold
    ad = applicability_domain_summary(len(max_tanimoto), int(inside_ad.sum()), args.tanimoto_threshold)

    print(f"\nApplicability domain (threshold={args.tanimoto_threshold}):")
    print(f"  Inside AD:  {ad['n_inside']} ({(1 - ad['frac_outside']) * 100:.1f}%)")
    print(f"  Outside AD: {ad['n_outside']} ({ad['frac_outside'] * 100:.1f}%)")
    if ad["warning"]:
        print(f"\n  *** {ad['warning']} ***")

    # ----------------------------------------------------------------
    # STEP 4: train/load model + internal CV
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN / LOAD MODEL + INTERNAL VALIDATION")
    print("=" * 60)

    model = xgb.XGBClassifier(**BEST_PARAMS)
    if args.skip_training and args.model_path.exists():
        model.load_model(args.model_path)
        print(f"Loaded existing model from {args.model_path}")
    else:
        model.fit(X_train, y_train)
        model.save_model(args.model_path)
        print(f"Model trained and saved to {args.model_path}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(xgb.XGBClassifier(**BEST_PARAMS), X_train, y_train, cv=cv)
    y_proba_cv = cross_val_predict(xgb.XGBClassifier(**BEST_PARAMS), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    cv_acc = accuracy_score(y_train, y_pred_cv)
    cv_auc = roc_auc_score(y_train, y_proba_cv)
    cv_mcc = matthews_corrcoef(y_train, y_pred_cv)
    cv_f1 = f1_score(y_train, y_pred_cv)
    cv_bal = balanced_accuracy_score(y_train, y_pred_cv)
    print(f"Internal 5-fold CV: AUC={cv_auc:.3f}  MCC={cv_mcc:.3f}  Acc={cv_acc:.3f}  F1={cv_f1:.3f}  BalAcc={cv_bal:.3f}")

    # ----------------------------------------------------------------
    # STEP 5: predict
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: PREDICTING ON YOUR LIBRARY")
    print("=" * 60)

    y_proba_screen = model.predict_proba(X_screen.values)[:, 1]
    y_pred_screen = (y_proba_screen >= 0.5).astype(int)
    n_active = (y_pred_screen == 1).sum()
    n_inactive = (y_pred_screen == 0).sum()
    print(f"Predicted active:   {n_active} ({n_active / len(y_pred_screen) * 100:.1f}%)")
    print(f"Predicted inactive: {n_inactive} ({n_inactive / len(y_pred_screen) * 100:.1f}%)")

    confidence = np.maximum(y_proba_screen, 1 - y_proba_screen)
    conf_categories = [confidence_category(c) for c in confidence]

    # ----------------------------------------------------------------
    # STEP 6: build results
    # ----------------------------------------------------------------
    results_df = pd.DataFrame({
        "name": meta_df["name"].values,
        "id": meta_df["id"].values,
        "smiles": meta_df["smiles"].values,
        "canonical_smiles": meta_df["canonical_smiles"].values,
        "predicted_class": y_pred_screen,
        "predicted_label": ["Active" if p == 1 else "Inactive" for p in y_pred_screen],
        "active_probability": np.round(y_proba_screen, 4),
        "confidence_score": np.round(confidence, 4),
        "confidence_category": conf_categories,
        "max_tanimoto_to_train": max_tanimoto,
        "inside_AD": inside_ad,
        "AD_note": ["Inside AD" if a else "Outside AD (novel scaffold)" for a in inside_ad],
    })
    results_df = results_df.sort_values("active_probability", ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    results_path = args.output_dir / "screening_predictions_all.xlsx"
    results_df.to_excel(results_path, index=False)
    print(f"\nFull results saved: {results_path}")

    actives_df = results_df[results_df["predicted_class"] == 1].copy()
    actives_path = args.output_dir / "screening_predictions_actives.xlsx"
    actives_df.to_excel(actives_path, index=False)
    print(f"Actives only saved: {actives_path} ({len(actives_df)} compounds)")

    # top-N predicted actives, for a quick manual look
    top_actives_df = actives_df.head(args.top_n_actives)
    top_actives_path = args.output_dir / "top_predicted_actives.xlsx"
    top_actives_df.to_excel(top_actives_path, index=False)
    print(f"Top {len(top_actives_df)} predicted actives saved: {top_actives_path}")

    # actives + full features, for the regression stage
    active_canonical = actives_df["canonical_smiles"].values
    active_mask = screen_df["canonical_smiles"].isin(active_canonical)
    active_features_df = pd.concat([
        screen_df.loc[active_mask, SCREEN_META].reset_index(drop=True),
        screen_df.loc[active_mask, X_cols].reset_index(drop=True),
    ], axis=1)
    prob_map = results_df.set_index("canonical_smiles")[
        ["active_probability", "confidence_score", "max_tanimoto_to_train", "inside_AD"]
    ]
    active_features_df = active_features_df.merge(prob_map, on="canonical_smiles", how="left")
    feat_path = args.output_dir / "actives_with_features_for_regression.xlsx"
    active_features_df.to_excel(feat_path, index=False)
    print(f"Actives + features saved (for stage 3): {feat_path}")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")
        _plot_overview(args.output_dir, y_pred_screen, confidence, n_active, n_inactive)
        _plot_applicability_domain(args.output_dir, max_tanimoto, args.tanimoto_threshold, ad)
        _plot_top50(args.output_dir, results_df)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total screening compounds: {len(screen_df)}")
    print(f"Predicted active:          {n_active} ({n_active / len(screen_df) * 100:.1f}%)")
    print(f"Active + high confidence:  {(actives_df['confidence_category'] == 'High').sum()}")
    print(f"Internal CV: AUC={cv_auc:.3f}  MCC={cv_mcc:.3f}  Acc={cv_acc:.3f}  F1={cv_f1:.3f}")
    if ad["warning"]:
        print(f"\n*** {ad['warning']} ***")
    print(f"\nOutput files saved to: {args.output_dir}")
    print("===== DONE =====")


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def _plot_overview(out_dir, y_pred_screen, confidence, n_active, n_inactive):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.pie([n_inactive, n_active], labels=[f"Inactive\n(n={n_inactive})", f"Active\n(n={n_active})"],
           colors=["#5B9BD5", "#E05C5C"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Predicted class distribution", fontweight="bold")

    ax = axes[1]
    conf_bins = {
        "High\n(>=0.8)": (confidence >= 0.8).sum(),
        "Medium\n(0.5-0.8)": ((confidence >= 0.5) & (confidence < 0.8)).sum(),
        "Low\n(<0.5)": (confidence < 0.5).sum(),
    }
    bars = ax.bar(conf_bins.keys(), conf_bins.values(), color=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.85)
    for bar, val in zip(bars, conf_bins.values()):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f"n={val}\n({val / len(confidence) * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Prediction confidence", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Screening Results Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "screening_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Overview plot saved.")


def _plot_applicability_domain(out_dir, max_tanimoto, threshold, ad):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(max_tanimoto, bins=30, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5, label=f"AD threshold ({threshold})")
    ax.set_xlabel("Max Tanimoto similarity to nearest training compound")
    ax.set_ylabel("Number of compounds")
    ax.set_title(
        f"Applicability Domain\n"
        f"{ad['n_inside']} inside AD ({(1 - ad['frac_outside']) * 100:.1f}%) | "
        f"{ad['n_outside']} outside AD ({ad['frac_outside'] * 100:.1f}%)",
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
    print("Applicability domain plot saved.")


def _plot_top50(out_dir, results_df):
    top50 = results_df[results_df["predicted_class"] == 1].head(50).copy()
    if len(top50) == 0:
        print("No predicted actives -- skipping top-50 plot.")
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(top50) * 0.18)))
    color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
    bar_colors = [color_map[c] for c in top50["confidence_category"]]

    ax.barh(range(len(top50)), top50["active_probability"].values[::-1], color=bar_colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top50)))
    labels_50 = [f"{str(row['name'])[:30]} ({'in-AD' if row['inside_AD'] else 'out-AD'})"
                 for _, row in top50.iloc[::-1].iterrows()]
    ax.set_yticklabels(labels_50, fontsize=7)
    ax.set_xlabel("Active probability")
    ax.set_title("Top predicted actives", fontweight="bold")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    patches = [mpatches.Patch(color=v, label=f"{k} confidence") for k, v in color_map.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "top_predicted_actives.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Top predicted actives plot saved.")


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
