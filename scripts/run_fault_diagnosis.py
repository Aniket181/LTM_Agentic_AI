"""
run_fault_diagnosis.py — Phase 5: Fault Diagnosis Baseline (Test 1)

Implements the Phase 5 Stratified Chronological Split experimental protocol,
trains a Random Forest classifier on Test 1, and evaluates it.

Usage:
    python scripts/run_fault_diagnosis.py --test_id 1

NOTE:
    This script creates its own experimental split from the raw Phase 3
    feature CSV. It does NOT use or overwrite Phase 4 processed outputs.
    A new StandardScaler is fitted exclusively on the Phase 5 X_train subset.

EXPERIMENTAL SPLIT DESCRIPTION:
    Phase 4 provides the leakage-safe preprocessing pipeline. Because the
    heuristic Faulty label begins at 80% of each run, its 70/30 split produces
    a single-class training partition. Therefore, Phase 5 uses a separate
    chronological development/evaluation protocol for supervised classification.
    This protocol is experimental and does not convert the heuristic labels
    into ground-truth fault annotations.

    Within each heuristic state period, the training portion chronologically
    precedes the corresponding testing portion. The Normal and Faulty periods
    are partitioned independently to ensure both classes are represented in
    training and evaluation.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from industrial_health.models.fault_classifier import (
    FaultClassifier,
    stratified_chronological_split,
    validate_split,
    WITHIN_CLASS_TRAIN_FRACTION,
    RF_PARAMS,
    LABEL_NAMES,
)


# ── Paths ─────────────────────────────────────────────────────────────────────

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def banner(title: str, width: int = 65) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def section(title: str, width: int = 65) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight_checks(split) -> bool:
    """Print pre-flight validation table. Return True if all pass."""
    section("PRE-FLIGHT CHECKS (must all pass before training)")

    checks = []

    def chk(cond: bool, label: str, detail: str = "") -> None:
        mark = "✓" if cond else "✗"
        print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
        checks.append(cond)

    train_classes = set(np.unique(split.y_train))
    test_classes  = set(np.unique(split.y_test))

    chk(0 in train_classes and 1 in train_classes,
        "Both classes in training set",
        f"classes present: {sorted(train_classes)}")
    chk(0 in test_classes and 1 in test_classes,
        "Both classes in test set",
        f"classes present: {sorted(test_classes)}")

    train_idx = set(split.meta_train["snapshot_index"])
    test_idx  = set(split.meta_test["snapshot_index"])
    idx_overlap = train_idx & test_idx
    chk(len(idx_overlap) == 0,
        "No snapshot_index overlap between train/test",
        f"overlapping: {len(idx_overlap)}")

    train_files = set(split.meta_train["source_file"])
    test_files  = set(split.meta_test["source_file"])
    file_overlap = train_files & test_files
    chk(len(file_overlap) == 0,
        "No source_file overlap between train/test",
        f"overlapping: {len(file_overlap)}")

    chk(not np.isnan(split.X_train).any(),  "No NaN in X_train")
    chk(not np.isinf(split.X_train).any(),  "No Inf in X_train")
    chk(not np.isnan(split.X_test).any(),   "No NaN in X_test")
    chk(not np.isinf(split.X_test).any(),   "No Inf in X_test")

    chk(split.X_train.shape[1] == split.X_test.shape[1],
        "Feature count consistent",
        f"{split.X_train.shape[1]} features")

    chk(len(split.feature_names) == split.X_train.shape[1],
        "feature_names length matches X columns",
        f"{len(split.feature_names)}")

    # Verify no metadata column leaked into features
    from industrial_health.models.fault_classifier import METADATA_COLS
    leaked = [f for f in split.feature_names if f in METADATA_COLS]
    chk(len(leaked) == 0,
        "No metadata columns in feature set",
        f"leaked: {leaked}" if leaked else "")

    # Scaler fitted on train only
    expected_scaler_n = split.X_train.shape[1]
    chk(hasattr(split.scaler, "mean_") and
        split.scaler.mean_.shape[0] == expected_scaler_n,
        "Scaler fitted with correct feature count",
        f"scaler.mean_.shape={split.scaler.mean_.shape}")

    all_ok = all(checks)
    print()
    if all_ok:
        print("  ✓ All pre-flight checks passed — proceeding to training.")
    else:
        print("  ✗ Pre-flight checks FAILED — training aborted.")
    return all_ok


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, test_id: int) -> Path:
    """Save a clean confusion matrix figure. Returns the saved path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase5_confusion_matrix_test{test_id}.png"

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = plt.cm.Blues
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Normal", "Faulty"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i, j]}",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_arr[i, j] > thresh else "black")

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — Test {test_id}\n"
        f"(Phase 5 Heuristic Labeling Protocol)",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_feature_importance(feat_df: pd.DataFrame, test_id: int) -> Path:
    """Save a horizontal bar chart of top-20 feature importances."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase5_rf_feature_importance_test{test_id}.png"

    # Shorten feature names for readability
    labels = [f.replace("bearing", "b").replace("_ch", "_c") for f in feat_df["feature"]]
    values = feat_df["importance"].values
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(values)))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(values)), values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
    ax.set_title(
        f"Top-{len(feat_df)} Feature Importances — Random Forest Test {test_id}\n"
        f"(Phase 5 Heuristic Labeling Protocol)",
        fontsize=11, pad=12,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 — Random Forest fault diagnosis (Test 1 only)"
    )
    parser.add_argument("--test_id",  type=int, default=1, choices=[1],
                        help="Test to process (currently only Test 1 supported)")
    parser.add_argument("--top_n_features", type=int, default=20,
                        help="Number of top features to report (default 20)")
    args = parser.parse_args()

    test_id  = args.test_id
    csv_path = FEATURES_DIR / f"test{test_id}_features.csv"

    banner(f"PHASE 5 — FAULT DIAGNOSIS BASELINE  |  TEST {test_id}")
    print(f"""
  EXPERIMENTAL PROTOCOL:
    Phase 4 provides the leakage-safe preprocessing pipeline. Because the
    heuristic Faulty label begins at 80% of each run, its 70/30 split produces
    a single-class training partition. Therefore, Phase 5 uses a separate
    chronological development/evaluation protocol for supervised classification.
    This protocol is experimental and does not convert the heuristic labels
    into ground-truth fault annotations.

  SPLIT METHOD:
    Within each heuristic state period, the training portion chronologically
    precedes the corresponding testing portion. The Normal and Faulty periods
    are partitioned independently to ensure both classes are represented in
    training and evaluation.

  LABEL DISCLAIMER:
    Results reflect performance under the heuristic temporal labeling protocol
    (final 20% of each run = Faulty). These are NOT verified fault-onset labels.
""")

    # ── Load raw Phase 3 feature CSV ─────────────────────────────────────────
    section(f"LOADING PHASE 3 FEATURES  —  test{test_id}_features.csv")
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Run Phase 3 first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    n_total = len(df)
    feat_cols_all = [c for c in df.columns
                     if c not in {"test_id","source_file","timestamp","n_channels",
                                  "snapshot_index","label","label_name",
                                  "label_method","fault_start_idx"}]
    print(f"  Total rows:     {n_total}")
    print(f"  Feature cols:   {len(feat_cols_all)}")
    print(f"  Normal rows:    {(df['label']==0).sum()}")
    print(f"  Faulty rows:    {(df['label']==1).sum()}")
    print(f"  fault_start_idx:{df['fault_start_idx'].iloc[0]}")
    print(f"  Date range:     {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Apply Stratified Chronological Split ──────────────────────────────────
    section("STRATIFIED CHRONOLOGICAL SPLIT")

    split = stratified_chronological_split(
        df, within_class_train_fraction=WITHIN_CLASS_TRAIN_FRACTION
    )

    print(f"  Within-class train fraction:   {WITHIN_CLASS_TRAIN_FRACTION:.0%}")
    print()
    print(f"  {'Partition':<22} {'Rows':>6}  {'Normal':>8}  {'Faulty':>8}")
    print(f"  {'─'*22} {'─'*6}  {'─'*8}  {'─'*8}")

    n_tr = len(split.y_train)
    n_te = len(split.y_test)
    print(f"  {'Train (combined)':<22} {n_tr:>6}  "
          f"{split.n_normal_train:>8}  {split.n_faulty_train:>8}")
    print(f"  {'Test  (combined)':<22} {n_te:>6}  "
          f"{split.n_normal_test:>8}  {split.n_faulty_test:>8}")
    print(f"  {'Total':<22} {n_tr+n_te:>6}  "
          f"{split.n_normal_train+split.n_normal_test:>8}  "
          f"{split.n_faulty_train+split.n_faulty_test:>8}")
    print()
    print(f"  Normal period:  {split.fault_start_idx} snapshots total → "
          f"train={split.n_normal_train}, test={split.n_normal_test}")
    print(f"  Faulty period:  {n_total - split.fault_start_idx} snapshots total → "
          f"train={split.n_faulty_train}, test={split.n_faulty_test}")
    print()
    print(f"  X_train shape:  {split.X_train.shape}")
    print(f"  X_test  shape:  {split.X_test.shape}")
    print(f"  Scaler:         StandardScaler fitted on X_train ONLY")
    print(f"  Scaler fit N:   {split.n_normal_train + split.n_faulty_train} rows")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not preflight_checks(split):
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    section("TRAINING — Random Forest")
    print(f"  Parameters:")
    for k, v in RF_PARAMS.items():
        print(f"    {k:<22} {v}")
    print()

    clf = FaultClassifier(params=RF_PARAMS)
    train_info = clf.train(split.X_train, split.y_train, split.feature_names)

    print(f"  Training accuracy:   {train_info['train_accuracy']:.4f}")
    print(f"  Training samples:    {train_info['n_train_samples']}")
    print(f"  Class distribution:  {train_info['class_distribution']}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    section("EVALUATION — Test set")
    result = clf.evaluate(split.X_test, split.y_test)

    print(f"\n  Test set size: {result.n_samples} rows")
    print(f"\n  Overall metrics:")
    print(f"    Accuracy:              {result.accuracy:.4f}")
    print(f"    Macro F1:              {result.f1_macro:.4f}")
    print(f"    Weighted F1:           {result.f1_weighted:.4f}")
    print()
    print(f"  Per-class metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Normal':<10} {result.precision_normal:>10.4f} "
          f"{result.recall_normal:>10.4f} {result.f1_normal:>10.4f}")
    print(f"  {'Faulty':<10} {result.precision_faulty:>10.4f} "
          f"{result.recall_faulty:>10.4f} {result.f1_faulty:>10.4f}")
    print()
    print("  Full classification report:")
    for line in result.classification_report_str.splitlines():
        print(f"    {line}")

    cm = np.array(result.confusion_matrix)
    print(f"\n  Confusion matrix (rows=True, cols=Predicted):")
    print(f"                 Pred Normal  Pred Faulty")
    print(f"  True Normal    {cm[0,0]:>11}  {cm[0,1]:>11}")
    print(f"  True Faulty    {cm[1,0]:>11}  {cm[1,1]:>11}")

    # ── Feature importance ────────────────────────────────────────────────────
    section(f"TOP {args.top_n_features} FEATURE IMPORTANCES")
    feat_df = clf.get_feature_importance(top_n=args.top_n_features)
    print(f"\n  {'Rank':<5} {'Feature':<42} {'Importance':>12}")
    print(f"  {'─'*5} {'─'*42} {'─'*12}")
    for _, row in feat_df.iterrows():
        print(f"  {int(row['rank']):<5} {row['feature']:<42} {row['importance']:>12.6f}")

    # ── Save figures ──────────────────────────────────────────────────────────
    section("SAVING FIGURES")
    cm_path   = plot_confusion_matrix(result.confusion_matrix, test_id)
    feat_path = plot_feature_importance(feat_df, test_id)
    print(f"  Confusion matrix:    {cm_path.relative_to(PROJECT_ROOT)}")
    print(f"  Feature importance:  {feat_path.relative_to(PROJECT_ROOT)}")

    # ── Save model ────────────────────────────────────────────────────────────
    section("SAVING MODEL ARTIFACTS")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODELS_DIR / f"random_forest_test{test_id}.joblib"
    scaler_path = MODELS_DIR / "scaler" / f"phase5_test{test_id}_scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    clf.save(model_path)
    joblib.dump(split.scaler, scaler_path)
    print(f"  Model:   {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Scaler:  {scaler_path.relative_to(PROJECT_ROOT)}")

    # ── Save experiment report ────────────────────────────────────────────────
    section("SAVING EXPERIMENT REPORT")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"phase5_experiment_test{test_id}.json"

    report = {
        "phase":          "5 — Fault Diagnosis Baseline",
        "test_id":        test_id,
        "timestamp":      datetime.now().isoformat(),
        "dataset":        str(csv_path.name),
        "n_total":        int(n_total),
        "split_method":   "stratified_chronological",
        "within_class_train_fraction": WITHIN_CLASS_TRAIN_FRACTION,
        "fault_start_idx": int(split.fault_start_idx),
        "split_sizes": {
            "n_train":          int(n_tr),
            "n_test":           int(n_te),
            "n_normal_train":   int(split.n_normal_train),
            "n_faulty_train":   int(split.n_faulty_train),
            "n_normal_test":    int(split.n_normal_test),
            "n_faulty_test":    int(split.n_faulty_test),
        },
        "feature_count":  int(len(split.feature_names)),
        "model_params":   RF_PARAMS,
        "scaler":         "StandardScaler fitted on X_train only (Phase 5 split)",
        "metrics": {
            "accuracy":          result.accuracy,
            "precision_normal":  result.precision_normal,
            "recall_normal":     result.recall_normal,
            "f1_normal":         result.f1_normal,
            "precision_faulty":  result.precision_faulty,
            "recall_faulty":     result.recall_faulty,
            "f1_faulty":         result.f1_faulty,
            "f1_macro":          result.f1_macro,
            "f1_weighted":       result.f1_weighted,
            "confusion_matrix":  result.confusion_matrix,
        },
        "top_features": feat_df[["rank","feature","importance"]].to_dict("records"),
        "leakage_checks": {
            "random_shuffle_used":       False,
            "metadata_used_as_features": False,
            "scaler_fit_on_test":        False,
            "phase4_scaler_reused":      False,
            "train_test_row_overlap":    False,
            "train_test_file_overlap":   False,
        },
        "label_disclaimer": (
            "The binary Normal/Faulty labels are heuristic temporal labels based "
            "on the final 20% of each run. They are not precise fault-onset "
            "annotations. Results reflect performance under the heuristic "
            "temporal labeling protocol."
        ),
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report JSON: {report_path.relative_to(PROJECT_ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("PHASE 5 COMPLETE — FINAL SUMMARY")
    print(f"""
  Dataset:            Test {test_id}  |  {n_total} total snapshots
  Feature count:      {len(split.feature_names)}
  Split method:       Stratified Chronological Split

  ┌─────────────────────────────────────────────┐
  │              SPLIT SIZES                    │
  ├─────────────────┬────────┬────────┬─────────┤
  │ Partition       │  Total │ Normal │  Faulty │
  ├─────────────────┼────────┼────────┼─────────┤
  │ Train           │  {n_tr:>5}  │  {split.n_normal_train:>5}  │   {split.n_faulty_train:>5}  │
  │ Test            │  {n_te:>5}  │  {split.n_normal_test:>5}  │   {split.n_faulty_test:>5}  │
  └─────────────────┴────────┴────────┴─────────┘

  ┌─────────────────────────────────────────────┐
  │              TEST SET METRICS               │
  ├─────────────────────────────────────────────┤
  │  Accuracy       {result.accuracy:>8.4f}                   │
  │  Macro F1       {result.f1_macro:>8.4f}                   │
  │  Weighted F1    {result.f1_weighted:>8.4f}                   │
  │                                             │
  │  Class       Precision  Recall      F1      │
  │  Normal      {result.precision_normal:>9.4f}  {result.recall_normal:>7.4f}  {result.f1_normal:>8.4f}  │
  │  Faulty      {result.precision_faulty:>9.4f}  {result.recall_faulty:>7.4f}  {result.f1_faulty:>8.4f}  │
  └─────────────────────────────────────────────┘

  IMPORTANT — ACADEMIC INTERPRETATION:
    These results reflect Random Forest performance under the heuristic
    temporal labeling protocol. Do NOT interpret as verified fault-diagnosis
    accuracy. The model distinguishes artificially defined Normal/Faulty
    temporal regions, not clinically validated fault states.

  STOP — Do NOT proceed to Tests 2/3 or Phase 6 without approval.
""")


if __name__ == "__main__":
    main()
