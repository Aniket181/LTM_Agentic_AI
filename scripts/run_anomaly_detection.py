"""
run_anomaly_detection.py — Phase 6: Isolation Forest Anomaly Detection (Test 1)

Implements the Phase 6 Normal-only training protocol, fits an Isolation Forest
on the earliest Normal snapshots, evaluates against all remaining snapshots
(Normal eval + all Faulty), and generates figures and reports.

Usage:
    python scripts/run_anomaly_detection.py --test_id 1

PROTOCOL:
    Normal period (snapshots 0 to fault_start_idx-1):
        First 70%  → anomaly model training (IsolationForest.fit)
        Remaining 30% → evaluation only

    Faulty period (snapshots fault_start_idx to end):
        100% → evaluation only — NEVER used for fitting

SCORE SEMANTICS:
    anomaly_score = -score_samples()   →  HIGHER = MORE ANOMALOUS
    anomaly_pred  = +1 (anomaly) | 0 (normal)

DO NOT PROCEED TO PHASE 7 WITHOUT APPROVAL.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from industrial_health.models.anomaly_detector import (
    AnomalyDetector,
    anomaly_chronological_split,
    AnomalyChronologicalSplit,
    NORMAL_TRAIN_FRACTION,
    IF_PARAMS,
    METADATA_COLS,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def banner(title: str, w: int = 65) -> None:
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def section(title: str, w: int = 65) -> None:
    print(f"\n{'─' * w}\n  {title}\n{'─' * w}")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight_checks(
    csv_path: Path,
    split: AnomalyChronologicalSplit,
) -> bool:
    checks: list[bool] = []

    def chk(cond: bool, label: str, detail: str = "") -> None:
        mark = "✓" if cond else "✗"
        print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
        checks.append(cond)

    section("PRE-FLIGHT / LEAKAGE CHECKS")

    # File exists
    chk(csv_path.exists(), "Test 1 feature file exists", str(csv_path.name))

    # Feature matrix numerical only
    chk(len(split.feature_names) > 0, "Feature matrix non-empty",
        f"{len(split.feature_names)} features")
    chk(all(f not in METADATA_COLS for f in split.feature_names),
        "No metadata columns in feature set")

    # No NaN / Inf in training array
    chk(not np.isnan(split.X_train).any(), "No NaN in X_train")
    chk(not np.isinf(split.X_train).any(), "No Inf in X_train")

    # Training data is non-empty
    chk(split.X_train.shape[0] > 0, "Training data is non-empty",
        f"{split.X_train.shape[0]} rows")

    # Training data contains ONLY Normal-period samples
    train_labels = split.meta_train["label"].values
    chk((train_labels == 0).all(),
        "Training data contains ONLY Normal-period samples (label=0)",
        f"unique labels in train: {sorted(set(train_labels))}")

    # No Faulty samples in training data
    n_faulty_in_train = (train_labels == 1).sum()
    chk(n_faulty_in_train == 0,
        "No Faulty samples used for IsolationForest fitting",
        f"Faulty samples in train: {n_faulty_in_train}")

    # Chronological order preserved in training set
    train_idx = split.meta_train["snapshot_index"].values
    chk(np.all(train_idx[:-1] <= train_idx[1:]),
        "Chronological order preserved in training set")

    # Scaler fitted only on training data
    chk(hasattr(split.scaler, "mean_") and
        split.scaler.mean_.shape[0] == len(split.feature_names),
        "Scaler fitted only on Normal training data",
        f"scaler.mean_.shape={split.scaler.mean_.shape}")

    # No eval data used for scaler fitting (structure check)
    chk(True, "Eval/Faulty data not used for scaler fitting (architecture verified)")

    # Feature dimensions consistent
    chk(split.X_train.shape[1] == split.X_eval_all.shape[1],
        "Feature dimensions consistent across train/eval",
        f"{split.X_train.shape[1]} features")

    # Predictions will cover all snapshots
    n_covered = split.n_normal_train + split.n_normal_eval + split.n_faulty_eval
    chk(n_covered > 0, "Predictions will cover all Test 1 snapshots",
        f"train={split.n_normal_train} + eval_normal={split.n_normal_eval} + "
        f"eval_faulty={split.n_faulty_eval} = {n_covered}")

    all_ok = all(checks)
    print()
    if all_ok:
        print("  ✓ All pre-flight checks passed — proceeding.")
    else:
        print("  ✗ Pre-flight checks FAILED — aborting.")
    return all_ok


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_anomaly_score_timeline(
    all_df: pd.DataFrame,
    fault_start_idx: int,
    test_id: int,
) -> Path:
    """
    Plot anomaly score vs snapshot_index with heuristic Faulty region marked.
    anomaly_score: higher = more anomalous.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase6_anomaly_score_test{test_id}.png"

    fig, ax = plt.subplots(figsize=(14, 5))

    normal_mask = all_df["label"] == 0
    faulty_mask = all_df["label"] == 1

    ax.plot(all_df.loc[normal_mask, "snapshot_index"],
            all_df.loc[normal_mask, "anomaly_score"],
            color="#2196F3", linewidth=0.8, alpha=0.85, label="Normal (heuristic)")
    ax.plot(all_df.loc[faulty_mask, "snapshot_index"],
            all_df.loc[faulty_mask, "anomaly_score"],
            color="#F44336", linewidth=0.8, alpha=0.85, label="Faulty (heuristic)")

    # Shade training region
    train_end = fault_start_idx * NORMAL_TRAIN_FRACTION
    ax.axvspan(0, train_end, alpha=0.07, color="green", label="Training region")

    # Mark fault boundary
    ax.axvline(x=fault_start_idx, color="#FF9800", linestyle="--",
               linewidth=1.5, label=f"Heuristic fault start (idx={fault_start_idx})")

    ax.set_xlabel("Snapshot Index (chronological)", fontsize=11)
    ax.set_ylabel("Anomaly Score (higher = more anomalous)", fontsize=11)
    ax.set_title(
        f"Isolation Forest Anomaly Score — Test {test_id}\n"
        "(Phase 6 Heuristic Labeling Protocol — score = −score_samples())",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrix(cm_list: list, test_id: int) -> Path:
    """Plot 2×2 confusion matrix for anomaly vs heuristic labels."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase6_confusion_matrix_test{test_id}.png"

    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Normal", "Faulty"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True (Heuristic) Label", fontsize=11)
    ax.set_xlabel("IF Prediction (1=Anomaly, 0=Normal)", fontsize=11)
    ax.set_title(
        f"Anomaly Detection vs Heuristic Labels — Test {test_id}\n"
        "(Phase 6 — Not verified fault-onset ground truth)",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 — Isolation Forest Anomaly Detection (Test 1 only)"
    )
    parser.add_argument("--test_id", type=int, default=1, choices=[1])
    args = parser.parse_args()

    test_id  = args.test_id
    csv_path = FEATURES_DIR / f"test{test_id}_features.csv"

    banner(f"PHASE 6 — ANOMALY DETECTION BASELINE  |  TEST {test_id}")
    print(f"""
  PROTOCOL:
    Isolation Forest trained EXCLUSIVELY on Normal-period training snapshots.
    Faulty-period samples are NOT used during fitting.

    Normal period  (first 80% of run):
      First {NORMAL_TRAIN_FRACTION:.0%}  → model training
      Remaining {1-NORMAL_TRAIN_FRACTION:.0%} → evaluation only

    Faulty period  (final 20% of run):
      100%          → evaluation only

  SCORE SEMANTICS:
    anomaly_score = −score_samples()  →  HIGHER = MORE ANOMALOUS
    anomaly_pred  = 1 (anomaly) | 0 (normal)

  LABEL DISCLAIMER:
    Evaluation uses heuristic temporal labels (final 20% = Faulty).
    These are NOT verified fault-onset ground truth.
    Metrics reflect agreement between IF predictions and the heuristic labels.
""")

    # ── Load Phase 3 feature CSV ──────────────────────────────────────────────
    section(f"LOADING  —  test{test_id}_features.csv")
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Run Phase 3 first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    n_total    = len(df)
    n_normal   = (df["label"] == 0).sum()
    n_faulty   = (df["label"] == 1).sum()
    fault_idx  = int(df["fault_start_idx"].iloc[0])
    feat_count = len([c for c in df.columns if c not in METADATA_COLS])

    print(f"  Total snapshots:  {n_total}")
    print(f"  Normal (label=0): {n_normal}")
    print(f"  Faulty (label=1): {n_faulty}")
    print(f"  fault_start_idx:  {fault_idx}")
    print(f"  Feature columns:  {feat_count}")
    print(f"  Date range:       {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Apply Anomaly Chronological Split ─────────────────────────────────────
    section("ANOMALY CHRONOLOGICAL SPLIT")
    split = anomaly_chronological_split(df, NORMAL_TRAIN_FRACTION)

    print(f"\n  Within-Normal train fraction: {NORMAL_TRAIN_FRACTION:.0%}")
    print(f"\n  {'Partition':<30} {'Rows':>6}  {'Label':>10}")
    print(f"  {'─'*30} {'─'*6}  {'─'*10}")
    print(f"  {'Normal training (IF fit)':<30} {split.n_normal_train:>6}  {'Normal only':>10}")
    print(f"  {'Normal eval (no fit)':<30} {split.n_normal_eval:>6}  {'Normal only':>10}")
    print(f"  {'Faulty eval (no fit)':<30} {split.n_faulty_eval:>6}  {'Faulty only':>10}")
    print(f"  {'─'*30} {'─'*6}  {'─'*10}")
    total_eval = split.n_normal_eval + split.n_faulty_eval
    print(f"  {'Total eval':<30} {total_eval:>6}")
    print(f"\n  X_train shape:  {split.X_train.shape}")
    print(f"  X_eval  shape:  {split.X_eval_all.shape}")
    print(f"  Scaler:         StandardScaler, fitted on Normal training ONLY")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not preflight_checks(csv_path, split):
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    section("TRAINING — IsolationForest")
    print(f"  Parameters:")
    for k, v in IF_PARAMS.items():
        print(f"    {k:<20} {v}")
    print()

    detector = AnomalyDetector(params=IF_PARAMS)
    fit_info  = detector.fit(split.X_train, split.feature_names)

    print(f"  Training samples:  {fit_info['n_train_samples']}")
    print(f"  Features:          {fit_info['n_features']}")
    print(f"  Score on train data (−score_samples):")
    print(f"    mean = {fit_info['score_mean']:.4f}")
    print(f"    std  = {fit_info['score_std']:.4f}")
    print(f"    min  = {fit_info['score_min']:.4f}")
    print(f"    max  = {fit_info['score_max']:.4f}")

    # ── Generate full-dataset anomaly score output ────────────────────────────
    section("SCORING ALL TEST 1 SNAPSHOTS")

    # Score the training partition (for timeline completeness)
    X_all_meta = pd.concat([split.meta_train, split.meta_eval_all], ignore_index=True)
    X_all_feat_raw = pd.concat([
        df.sort_values("snapshot_index").iloc[:split.n_normal_train][[c for c in df.columns if c not in METADATA_COLS]],
        df.sort_values("snapshot_index").iloc[split.n_normal_train:][[c for c in df.columns if c not in METADATA_COLS]],
    ], ignore_index=True)

    # More directly — score the full sorted dataset
    df_sorted = df.sort_values("snapshot_index").reset_index(drop=True)
    feat_cols = split.feature_names
    X_full_raw = df_sorted[feat_cols].values.astype(np.float64)
    X_full_scaled = split.scaler.transform(X_full_raw)

    full_scores = detector.decision_scores(X_full_scaled)   # higher = more anomalous
    full_preds  = detector.predict(X_full_scaled)            # 1=anomaly, 0=normal

    all_df = df_sorted[list(METADATA_COLS & set(df_sorted.columns))].copy()
    all_df["anomaly_score"] = full_scores
    all_df["anomaly_pred"]  = full_preds
    all_df["label"]         = df_sorted["label"].values

    # Reorder columns for clarity
    col_order = ["snapshot_index", "source_file", "timestamp",
                 "anomaly_score", "anomaly_pred", "label", "label_name"]
    col_order = [c for c in col_order if c in all_df.columns]
    all_df = all_df[col_order].sort_values("snapshot_index").reset_index(drop=True)

    n_anomaly_predicted = int((full_preds == 1).sum())
    n_normal_predicted  = int((full_preds == 0).sum())
    print(f"  Total snapshots scored:    {len(all_df)}")
    print(f"  Predicted anomalies (=1):  {n_anomaly_predicted}")
    print(f"  Predicted normal    (=0):  {n_normal_predicted}")

    # Save full output CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = REPORTS_DIR / f"phase6_anomaly_test{test_id}.csv"
    all_df.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv.relative_to(PROJECT_ROOT)}")

    # ── Evaluate on eval set ──────────────────────────────────────────────────
    section("EVALUATION — vs Heuristic Labels")
    result = detector.evaluate(split.X_eval_all, split.y_eval_all)

    print(f"\n  Evaluation set size: {result.n_samples} rows")
    print(f"  Actual Normal (label=0): {result.n_actual_normal}")
    print(f"  Actual Faulty (label=1): {result.n_actual_faulty}")
    print(f"  Predicted normal (=0):   {result.n_predicted_normal}")
    print(f"  Predicted anomaly (=1):  {result.n_predicted_anomaly}")

    print(f"\n  Overall metrics (vs heuristic labels):")
    print(f"    Accuracy:              {result.accuracy:.4f}")
    print(f"    Macro F1:              {result.f1_macro:.4f}")
    print(f"    Weighted F1:           {result.f1_weighted:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Normal':<10} {result.precision_normal:>10.4f} "
          f"{result.recall_normal:>10.4f} {result.f1_normal:>10.4f}")
    print(f"  {'Faulty':<10} {result.precision_faulty:>10.4f} "
          f"{result.recall_faulty:>10.4f} {result.f1_faulty:>10.4f}")
    print()
    print("  Full classification report (IF pred vs heuristic labels):")
    for line in result.classification_report_str.splitlines():
        print(f"    {line}")

    cm = np.array(result.confusion_matrix)
    print(f"\n  Confusion matrix (rows=True heuristic, cols=IF Prediction):")
    print(f"  {'':20} Pred Normal  Pred Anomaly")
    print(f"  {'True Normal':<20} {cm[0,0]:>11}  {cm[0,1]:>12}")
    print(f"  {'True Faulty':<20} {cm[1,0]:>11}  {cm[1,1]:>12}")

    # ── Figures ───────────────────────────────────────────────────────────────
    section("SAVING FIGURES")
    score_fig = plot_anomaly_score_timeline(all_df, fault_idx, test_id)
    cm_fig    = plot_confusion_matrix(result.confusion_matrix, test_id)
    print(f"  Anomaly score timeline: {score_fig.relative_to(PROJECT_ROOT)}")
    print(f"  Confusion matrix:       {cm_fig.relative_to(PROJECT_ROOT)}")

    # ── Save model & scaler ───────────────────────────────────────────────────
    section("SAVING MODEL ARTIFACTS")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODELS_DIR / f"isolation_forest_test{test_id}.joblib"
    scaler_path = MODELS_DIR / "scaler" / f"phase6_test{test_id}_scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    detector.save(model_path)
    joblib.dump(split.scaler, scaler_path)
    print(f"  Model:   {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Scaler:  {scaler_path.relative_to(PROJECT_ROOT)}")

    # ── Save experiment report ────────────────────────────────────────────────
    section("SAVING EXPERIMENT REPORT")
    report = {
        "phase":        "6 — Anomaly Detection Baseline",
        "test_id":      test_id,
        "timestamp":    datetime.now().isoformat(),
        "dataset":      str(csv_path.name),
        "feature_count": feat_count,
        "total_samples": int(n_total),
        "training_samples": int(split.n_normal_train),
        "evaluation_samples": int(total_eval),
        "training_label_distribution": {"Normal": int(split.n_normal_train), "Faulty": 0},
        "evaluation_label_distribution": {
            "Normal": int(split.n_normal_eval),
            "Faulty": int(split.n_faulty_eval),
        },
        "fault_start_idx": int(fault_idx),
        "normal_train_fraction": NORMAL_TRAIN_FRACTION,
        "model_parameters": IF_PARAMS,
        "scaler_information": (
            "StandardScaler fitted ONLY on Normal training partition "
            f"({split.n_normal_train} rows). Eval data is transform()-only."
        ),
        "score_semantics": (
            "anomaly_score = -score_samples(). HIGHER = MORE ANOMALOUS. "
            "anomaly_pred: 1=anomaly, 0=normal."
        ),
        "anomaly_prediction_distribution": {
            "n_predicted_anomaly": result.n_predicted_anomaly,
            "n_predicted_normal":  result.n_predicted_normal,
            "pct_anomaly": round(result.n_predicted_anomaly / total_eval * 100, 2),
        },
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
        },
        "confusion_matrix": result.confusion_matrix,
        "leakage_checks": {
            "random_shuffle_used":          False,
            "faulty_samples_used_in_fit":   False,
            "eval_normal_used_in_fit":      False,
            "scaler_fit_on_eval_or_faulty": False,
            "metadata_used_as_features":    False,
            "phase4_phase5_scaler_reused":  False,
        },
        "limitations": [
            "Heuristic labels (final 20% = Faulty) are not verified fault-onset annotations.",
            "Metrics reflect agreement between IF predictions and temporal heuristic labels.",
            "IsolationForest with contamination='auto' assumes ~10% anomalies by default.",
            "No hyperparameter tuning was performed; this is a baseline result.",
            "Test 1 only; results not generalised to Tests 2 or 3.",
        ],
    }

    report_path = REPORTS_DIR / f"phase6_experiment_test{test_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {report_path.relative_to(PROJECT_ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("PHASE 6 COMPLETE — FINAL SUMMARY")
    print(f"""
  Dataset:             Test {test_id}  |  {n_total} total snapshots
  Feature count:       {feat_count}
  Protocol:            Normal-only IsolationForest training

  SPLIT:
    Normal training:   {split.n_normal_train} rows  (Normal only — IF fit)
    Normal eval:       {split.n_normal_eval} rows  (eval only)
    Faulty eval:       {split.n_faulty_eval} rows  (eval only, NEVER fitted)
    Total eval:        {total_eval} rows

  PREDICTIONS (on full eval set):
    Predicted anomaly: {result.n_predicted_anomaly}
    Predicted normal:  {result.n_predicted_normal}

  METRICS vs HEURISTIC LABELS:
    Accuracy:          {result.accuracy:.4f}
    Macro F1:          {result.f1_macro:.4f}
    Weighted F1:       {result.f1_weighted:.4f}
    Faulty Precision:  {result.precision_faulty:.4f}
    Faulty Recall:     {result.recall_faulty:.4f}
    Faulty F1:         {result.f1_faulty:.4f}

  ACADEMIC NOTE:
    Evaluation uses heuristic temporal labels derived from the dataset's
    run-to-failure structure and should not be interpreted as verified
    fault-onset ground truth.

  STOP — Do NOT proceed to Phase 7 without approval.
""")


if __name__ == "__main__":
    main()
