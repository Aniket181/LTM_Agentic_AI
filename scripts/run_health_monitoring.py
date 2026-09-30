"""
run_health_monitoring.py — Phase 7: Health Monitoring Pipeline (Test 1)

Integrates Phase 5 (Random Forest) + Phase 6 (Isolation Forest) model
artifacts with Phase 3 RMS features to produce an explainable per-snapshot
Health Score for all Test 1 snapshots.

Usage:
    python scripts/run_health_monitoring.py --test_id 1

Requires (must exist — will NOT silently retrain):
    models/random_forest_test1.joblib
    models/scaler/phase5_test1_scaler.pkl
    models/isolation_forest_test1.joblib
    models/scaler/phase6_test1_scaler.pkl

IMPORTANT:
    This is a deterministic integration layer — no new ML training occurs.
    Results must be interpreted as:
    "Health Score under the Phase 5 heuristic label protocol + Phase 6
    Normal-only IF protocol."

DO NOT START PHASE 8 WITHOUT APPROVAL.
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

from industrial_health.health.health_monitor import (
    HealthMonitor,
    HealthConfig,
    DEFAULT_CONFIG,
)
from industrial_health.health.trend import (
    rolling_trend,
    compute_run_trend_summary,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = PROJECT_ROOT / "data"   / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def banner(title: str, w: int = 65) -> None:
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def section(title: str, w: int = 65) -> None:
    print(f"\n{'─' * w}\n  {title}\n{'─' * w}")


# ── Figures ───────────────────────────────────────────────────────────────────

COLOR_STABLE    = "#2196F3"
COLOR_DEGRADING = "#FF9800"
COLOR_CRITICAL  = "#F44336"


def plot_health_score_timeline(result_df: pd.DataFrame, cfg: HealthConfig,
                                fault_start_idx: int, test_id: int) -> Path:
    """Health score coloured by status, with threshold bands and fault marker."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase7_health_score_test{test_id}.png"

    fig, ax = plt.subplots(figsize=(15, 5))

    idx = result_df["snapshot_index"].values
    hs  = result_df["health_score"].values

    # Colour each segment by health_status
    status_map = {
        "Stable":    COLOR_STABLE,
        "Degrading": COLOR_DEGRADING,
        "Critical":  COLOR_CRITICAL,
    }
    for status, color in status_map.items():
        mask = result_df["health_status"] == status
        ax.scatter(idx[mask], hs[mask], s=3, color=color, alpha=0.7,
                   label=status, zorder=3)

    # Threshold bands (filled)
    ax.axhspan(cfg.threshold_stable, 100,
               alpha=0.06, color=COLOR_STABLE, label="_stable_band")
    ax.axhspan(cfg.threshold_degrading, cfg.threshold_stable,
               alpha=0.06, color=COLOR_DEGRADING, label="_degrading_band")
    ax.axhspan(0, cfg.threshold_degrading,
               alpha=0.06, color=COLOR_CRITICAL, label="_critical_band")

    # Threshold lines
    ax.axhline(cfg.threshold_stable,    color=COLOR_STABLE,    linestyle="--",
               linewidth=1.2, alpha=0.8)
    ax.axhline(cfg.threshold_degrading, color=COLOR_CRITICAL,  linestyle="--",
               linewidth=1.2, alpha=0.8)

    # Heuristic fault boundary
    ax.axvline(x=fault_start_idx, color="#9C27B0", linestyle=":",
               linewidth=1.8, label=f"Heuristic fault start (idx={fault_start_idx})")

    ax.set_xlabel("Snapshot Index (chronological)", fontsize=11)
    ax.set_ylabel("Health Score (0–100)", fontsize=11)
    ax.set_ylim(-2, 105)
    ax.set_title(
        f"Equipment Health Score — Test {test_id}\n"
        f"(Phase 7 Integration: RF + IF + RMS | Weights: "
        f"RMS={cfg.weight_rms}, IF={cfg.weight_anomaly}, RF={cfg.weight_fault})",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_component_comparison(result_df: pd.DataFrame, test_id: int) -> Path:
    """Three stacked subplots: RMS, anomaly, fault components + health."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase7_component_comparison_test{test_id}.png"

    idx = result_df["snapshot_index"].values

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    components = [
        ("health_rms_component",     "RMS Health Component (0–100)",     "#4CAF50"),
        ("health_anomaly_component", "Anomaly Health Component (0–100)", "#2196F3"),
        ("health_fault_component",   "Fault Health Component (0–100)",   "#FF9800"),
        ("health_score",             "Composite Health Score (0–100)",   "#9C27B0"),
    ]

    fault_start = int(result_df["snapshot_index"][result_df["label"] == 1].min()) \
        if (result_df["label"] == 1).any() else None

    for ax, (col, title, color) in zip(axes, components):
        ax.plot(idx, result_df[col].values, color=color, linewidth=0.9, alpha=0.85)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.2)
        if fault_start is not None:
            ax.axvline(x=fault_start, color="#F44336", linestyle=":",
                       linewidth=1.2, alpha=0.7)

    axes[-1].set_xlabel("Snapshot Index (chronological)", fontsize=11)
    fig.suptitle(
        f"Health Score Components — Test {test_id}\n"
        "(Red dashed = heuristic fault boundary | Phase 7 engineering integration layer)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trend_overlay(result_df: pd.DataFrame, trend_labels: np.ndarray,
                       test_id: int) -> Path:
    """Health score with rolling trend labels colour-coded."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"phase7_trend_overlay_test{test_id}.png"

    idx = result_df["snapshot_index"].values
    hs  = result_df["health_score"].values

    tcolor_map = {"Stable": COLOR_STABLE, "Degrading": COLOR_DEGRADING,
                  "Improving": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(15, 5))
    for tlabel, tcolor in tcolor_map.items():
        mask = trend_labels == tlabel
        if mask.any():
            ax.scatter(idx[mask], hs[mask], s=4, color=tcolor,
                       alpha=0.75, label=f"Trend: {tlabel}", zorder=3)

    ax.set_xlabel("Snapshot Index (chronological)", fontsize=11)
    ax.set_ylabel("Health Score (0–100)", fontsize=11)
    ax.set_ylim(-2, 105)
    ax.set_title(
        f"Health Score with Rolling Trend Labels — Test {test_id}\n"
        f"(Rolling window = {DEFAULT_CONFIG.trend_window} snapshots)",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7 — Health Monitoring Pipeline (Test 1 only)"
    )
    parser.add_argument("--test_id", type=int, default=1, choices=[1])
    parser.add_argument("--weight_rms",     type=float, default=0.20)
    parser.add_argument("--weight_anomaly", type=float, default=0.40)
    parser.add_argument("--weight_fault",   type=float, default=0.40)
    parser.add_argument("--threshold_stable",    type=float, default=70.0)
    parser.add_argument("--threshold_degrading", type=float, default=40.0)
    args = parser.parse_args()

    test_id = args.test_id

    # Build config (will validate weights sum to 1)
    try:
        cfg = HealthConfig(
            weight_rms=args.weight_rms,
            weight_anomaly=args.weight_anomaly,
            weight_fault=args.weight_fault,
            threshold_stable=args.threshold_stable,
            threshold_degrading=args.threshold_degrading,
        )
    except ValueError as e:
        print(f"  ERROR in configuration: {e}")
        sys.exit(1)

    banner(f"PHASE 7 — HEALTH MONITORING  |  TEST {test_id}")
    print(f"""
  TYPE:
    Deterministic integration layer — no new ML training.
    Uses existing Phase 5 (RF) and Phase 6 (IF) model artifacts.

  DISCLAIMER:
    The Health Score is an explainable engineering integration layer and
    is not presented as a scientifically validated clinical/industrial
    health index.

  CONFIGURATION:
    Weights:     RMS={cfg.weight_rms:.2f}, IF={cfg.weight_anomaly:.2f}, RF={cfg.weight_fault:.2f}
    Thresholds:  Stable>={cfg.threshold_stable}, Degrading>={cfg.threshold_degrading}, Critical<{cfg.threshold_degrading}
""")

    # ── Resolve paths ──────────────────────────────────────────────────────
    csv_path      = FEATURES_DIR / f"test{test_id}_features.csv"
    rf_model_path = MODELS_DIR / f"random_forest_test{test_id}.joblib"
    rf_scaler_path= MODELS_DIR / "scaler" / f"phase5_test{test_id}_scaler.pkl"
    if_model_path = MODELS_DIR / f"isolation_forest_test{test_id}.joblib"
    if_scaler_path= MODELS_DIR / "scaler" / f"phase6_test{test_id}_scaler.pkl"

    # ── Existence checks ───────────────────────────────────────────────────
    section("PRE-FLIGHT CHECKS")
    all_ok = True
    artifacts = {
        "Feature CSV":           csv_path,
        "RF model (Phase 5)":    rf_model_path,
        "RF scaler (Phase 5)":   rf_scaler_path,
        "IF model (Phase 6)":    if_model_path,
        "IF scaler (Phase 6)":   if_scaler_path,
    }
    for label, path in artifacts.items():
        exists = path.exists()
        mark = "✓" if exists else "✗"
        print(f"  [{mark}] {label:<25} {path.name}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  STOP — Missing artifacts. Run Phase 5 and Phase 6 first.")
        sys.exit(1)
    print("\n  ✓ All artifacts present.")

    # ── Load feature CSV ───────────────────────────────────────────────────
    section(f"LOADING  —  test{test_id}_features.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    n_total = len(df)
    fault_start_idx = int(df["fault_start_idx"].iloc[0])
    n_normal = (df["label"] == 0).sum()
    n_faulty = (df["label"] == 1).sum()
    print(f"  Total snapshots:  {n_total}")
    print(f"  Normal (label=0): {n_normal}")
    print(f"  Faulty (label=1): {n_faulty}")
    print(f"  fault_start_idx:  {fault_start_idx}")

    # ── Load models ────────────────────────────────────────────────────────
    section("LOADING MODEL ARTIFACTS")
    try:
        monitor = HealthMonitor(
            rf_model_path=rf_model_path,
            rf_scaler_path=rf_scaler_path,
            if_model_path=if_model_path,
            if_scaler_path=if_scaler_path,
            config=cfg,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    ref = monitor.reference_info
    print(f"  RF features:  {ref['rf_feature_count']}")
    print(f"  IF features:  {ref['if_feature_count']}")
    print(f"  RF & IF feature sets match: {ref['rf_feature_count'] == ref['if_feature_count']}")

    # ── Score all snapshots ────────────────────────────────────────────────
    section("SCORING ALL SNAPSHOTS (chronological)")
    result_df = monitor.score_dataset(df)

    n_scored = len(result_df)
    print(f"  Snapshots scored:  {n_scored}")
    assert n_scored == n_total, f"Expected {n_total} rows, got {n_scored}"

    # Verify no NaN/Inf
    bad_mask = ~result_df["health_score"].apply(np.isfinite)
    print(f"  NaN/Inf in health_score: {bad_mask.sum()}")

    # ── Health summary ─────────────────────────────────────────────────────
    section("HEALTH SCORE SUMMARY")
    hs = result_df["health_score"].values
    print(f"\n  Min:    {hs.min():.2f}")
    print(f"  Max:    {hs.max():.2f}")
    print(f"  Mean:   {hs.mean():.2f}")
    print(f"  Median: {np.median(hs):.2f}")
    print(f"  Std:    {hs.std():.2f}")

    status_counts = result_df["health_status"].value_counts()
    print(f"\n  Status distribution:")
    for s in ["Stable", "Degrading", "Critical"]:
        n = int(status_counts.get(s, 0))
        pct = 100 * n / n_scored
        print(f"    {s:<12} {n:>5}  ({pct:.1f}%)")

    # ── Trend analysis ─────────────────────────────────────────────────────
    section("TREND ANALYSIS")
    trend_labels = rolling_trend(hs, window=cfg.trend_window)
    result_df["trend_label"] = trend_labels

    trend_summary = compute_run_trend_summary(
        hs,
        result_df["health_status"].values,
        result_df["snapshot_index"].values,
        trend_labels,
    )

    print(f"  Overall run trend:     {trend_summary['overall_trend']}")
    print(f"  Q1 mean health:        {trend_summary['q1_mean_health']:.2f}")
    print(f"  Q4 mean health:        {trend_summary['q4_mean_health']:.2f}")
    print(f"  Health delta Q1→Q4:    {trend_summary['health_delta_q1_to_q4']:.2f}")
    print(f"  First degradation:     snapshot index {trend_summary['first_degradation_snapshot_idx']}")
    print(f"  % Stable:   {trend_summary['pct_stable']:.1f}%")
    print(f"  % Degrading:{trend_summary['pct_degrading']:.1f}%")
    print(f"  % Critical: {trend_summary['pct_critical']:.1f}%")

    # ── Component statistics ───────────────────────────────────────────────
    section("COMPONENT STATISTICS")
    for col, label in [
        ("health_rms_component",     "RMS component (0–100)"),
        ("health_anomaly_component", "Anomaly component (0–100)"),
        ("health_fault_component",   "Fault component (0–100)"),
    ]:
        vals = result_df[col].values
        print(f"  {label}")
        print(f"    mean={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")

    # ── Save output CSV ────────────────────────────────────────────────────
    section("SAVING OUTPUTS")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = REPORTS_DIR / f"phase7_health_test{test_id}.csv"
    result_df.to_csv(csv_out, index=False)
    print(f"  Health CSV:    {csv_out.relative_to(PROJECT_ROOT)}")

    # ── Save figures ───────────────────────────────────────────────────────
    fig1 = plot_health_score_timeline(result_df, cfg, fault_start_idx, test_id)
    fig2 = plot_component_comparison(result_df, test_id)
    fig3 = plot_trend_overlay(result_df, trend_labels, test_id)
    print(f"  Fig 1 (health score timeline):    {fig1.relative_to(PROJECT_ROOT)}")
    print(f"  Fig 2 (component comparison):     {fig2.relative_to(PROJECT_ROOT)}")
    print(f"  Fig 3 (trend overlay):            {fig3.relative_to(PROJECT_ROOT)}")

    # ── Build & save experiment JSON ───────────────────────────────────────
    # Feature column info
    feat_cols = [c for c in df.columns
                 if c not in {"test_id","source_file","timestamp","n_channels",
                              "snapshot_index","label","label_name",
                              "label_method","fault_start_idx"}]
    rms_cols = [c for c in feat_cols if c.endswith("__rms")]

    report = {
        "phase":    "7 — Health Monitoring & Explainable Health Score",
        "dataset":  f"test{test_id}_features.csv",
        "test_case": test_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": cfg.as_dict,
        "data_summary": {
            "total_snapshots": int(n_total),
            "feature_count":   len(feat_cols),
            "rms_feature_count": len(rms_cols),
            "normal_snapshots": int(n_normal),
            "faulty_snapshots": int(n_faulty),
            "fault_start_idx":  int(fault_start_idx),
        },
        "component_summary": {
            "rms": {
                "rms_columns_used": rms_cols,
                "composite_rms_mean": float(result_df["composite_rms"].mean()),
                "composite_rms_min":  float(result_df["composite_rms"].min()),
                "composite_rms_max":  float(result_df["composite_rms"].max()),
            },
            "anomaly": {
                "if_score_mean": float(result_df["anomaly_score"].mean()),
                "if_score_min":  float(result_df["anomaly_score"].min()),
                "if_score_max":  float(result_df["anomaly_score"].max()),
                "n_anomaly_pred": int((result_df["anomaly_pred"] == 1).sum()),
            },
            "fault": {
                "prob_faulty_mean": float(result_df["prob_faulty"].mean()),
                "prob_faulty_min":  float(result_df["prob_faulty"].min()),
                "prob_faulty_max":  float(result_df["prob_faulty"].max()),
                "n_faulty_pred":    int((result_df["rf_pred"] == 1).sum()),
            },
        },
        "health_summary": {
            "min_health_score":    float(hs.min()),
            "max_health_score":    float(hs.max()),
            "mean_health_score":   float(hs.mean()),
            "median_health_score": float(np.median(hs)),
            "std_health_score":    float(hs.std()),
        },
        "status_counts": {
            "Stable":    int(status_counts.get("Stable",    0)),
            "Degrading": int(status_counts.get("Degrading", 0)),
            "Critical":  int(status_counts.get("Critical",  0)),
        },
        "trend_summary": trend_summary,
        "methodology_notes": [
            "Health Score is a weighted integration of three components: "
            "RMS degradation (Phase 3), anomaly severity (Phase 6 IF), "
            "and fault confidence (Phase 5 RF).",
            "No new ML model was trained in Phase 7.",
            "RMS normalization reference derived from the Normal training period only "
            f"(first {cfg.normal_reference_fraction:.0%} of Normal snapshots).",
            "Phase 5 RF was trained using heuristic temporal labels and the "
            "stratified chronological exploratory protocol.",
            "Phase 6 IF was trained exclusively on Normal-period data.",
            "Anomaly score semantics: decision_scores() = -score_samples(); "
            "higher = more anomalous.",
        ],
        "limitations": [
            "Health Score is an engineering prototype — not a scientifically "
            "validated clinical/industrial health index.",
            "Component weights (RMS=0.20, IF=0.40, RF=0.40) are engineering "
            "defaults and have not been optimised against ground-truth fault data.",
            "Health thresholds (Stable>=70, Degrading>=40, Critical<40) are "
            "engineering rules, not industry-certified standards.",
            "Phase 5 RF probabilities are not calibrated — they should not be "
            "interpreted as precise Bayesian fault probabilities.",
            "Test 1 only. Results not generalisable to Tests 2 or 3 without "
            "retraining all constituent models.",
        ],
    }

    json_out = REPORTS_DIR / f"phase7_experiment_test{test_id}.json"
    json_out.write_text(json.dumps(report, indent=2))
    print(f"  Experiment JSON: {json_out.relative_to(PROJECT_ROOT)}")

    # ── Final summary ──────────────────────────────────────────────────────
    banner("PHASE 7 COMPLETE — FINAL SUMMARY")
    print(f"""
  Test:            Test {test_id}  |  {n_total} snapshots
  Health range:    {hs.min():.1f} – {hs.max():.1f}
  Mean health:     {hs.mean():.1f}
  Status:          Stable={status_counts.get("Stable",0)}  Degrading={status_counts.get("Degrading",0)}  Critical={status_counts.get("Critical",0)}
  Overall trend:   {trend_summary["overall_trend"]}
  First degradation at snapshot: {trend_summary["first_degradation_snapshot_idx"]}

  STOP — Do NOT start Phase 8 without approval.
""")


if __name__ == "__main__":
    main()
