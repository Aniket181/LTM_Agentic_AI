"""
health_monitor.py — Phase 7: Health Monitoring Integration Layer

PURPOSE:
    Integrate outputs from Phase 5 (Random Forest fault classifier) and
    Phase 6 (Isolation Forest anomaly detector) with Phase 3 RMS features
    to produce a per-snapshot explainable Health Score (0–100) for Test 1.

ARCHITECTURE:
    This is a DETERMINISTIC INTEGRATION LAYER — no new ML model is trained.

    Inputs:
        1. Phase 3 feature CSV  → RMS degradation component
        2. Phase 6 IF model     → anomaly score component
        3. Phase 6 IF scaler    → transform raw features for IF
        4. Phase 5 RF model     → fault probability component
        5. Phase 5 RF scaler    → transform raw features for RF

    Health Score formula:
        health_score = clip(100 × (
            w_rms     × rms_component(0–1) +
            w_anomaly × anomaly_component(0–1) +
            w_fault   × fault_component(0–1)
        ), 0, 100)

    where:
        rms_component     → 1 = normal RMS, 0 = worst observed RMS
        anomaly_component → 1 = no anomaly, 0 = maximum anomaly
        fault_component   → 1 = confident Normal pred, 0 = confident Faulty pred

NORMALIZATION REFERENCE (leakage-safe):
    RMS baseline and reference statistics are derived exclusively from
    the Normal training period (snapshots 0..n_normal_train-1, the same
    partition used by Phase 6 for its scaler fit).

    This ensures no future-data leakage into earlier health values.

IMPORTANT DISCLAIMER:
    The Health Score is an explainable engineering integration layer and
    is not presented as a scientifically validated clinical/industrial
    health index.

    - Phase 5 uses heuristic labels and the stratified chronological
      exploratory protocol.
    - Phase 6 is Normal-only Isolation Forest training.
    - Health monitoring integrates those existing experimental components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from industrial_health.models.fault_classifier import FaultClassifier, METADATA_COLS
from industrial_health.models.anomaly_detector import (
    AnomalyDetector,
    NORMAL_TRAIN_FRACTION,
)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class HealthConfig:
    """
    All configurable parameters for Phase 7 health monitoring.

    Weights must sum to 1.0. They are validated on construction.
    Thresholds define the Stable / Degrading / Critical boundaries.
    """

    # Component weights
    weight_rms:     float = 0.20
    weight_anomaly: float = 0.40
    weight_fault:   float = 0.40

    # Health status thresholds (inclusive lower bound)
    threshold_stable:    float = 70.0   # health_score >= 70  → Stable
    threshold_degrading: float = 40.0   # 40 <= score < 70    → Degrading
                                        # score < 40          → Critical

    # Normal-period fraction for RMS reference (matches Phase 6 training protocol)
    normal_reference_fraction: float = NORMAL_TRAIN_FRACTION  # 0.70

    # Anomaly score reference ceiling for normalization.
    # Derived from Phase 6 Normal training period distribution.
    # See _ANOMALY_REF_MAX_SCORE in anomaly_health_component().
    anomaly_ref_max_score: float = 0.5

    # Trend window (snapshots)
    trend_window: int = 50

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        weights = [self.weight_rms, self.weight_anomaly, self.weight_fault]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative.")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"(rms={self.weight_rms}, anomaly={self.weight_anomaly}, "
                f"fault={self.weight_fault})"
            )
        if not (0.0 < self.threshold_degrading < self.threshold_stable < 100.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < degrading ({self.threshold_degrading}) "
                f"< stable ({self.threshold_stable}) < 100."
            )

    @property
    def as_dict(self) -> dict:
        return {
            "weight_rms":     self.weight_rms,
            "weight_anomaly": self.weight_anomaly,
            "weight_fault":   self.weight_fault,
            "threshold_stable":    self.threshold_stable,
            "threshold_degrading": self.threshold_degrading,
            "normal_reference_fraction": self.normal_reference_fraction,
            "anomaly_ref_max_score": self.anomaly_ref_max_score,
            "trend_window": self.trend_window,
        }


# ── Default config (module-level singleton) ───────────────────────────────────
DEFAULT_CONFIG = HealthConfig()


# ── Health status mapping ─────────────────────────────────────────────────────

def get_health_status(score: float, cfg: HealthConfig = DEFAULT_CONFIG) -> str:
    """
    Map health score to a three-state status label.

    Stable:   score >= cfg.threshold_stable
    Degrading: cfg.threshold_degrading <= score < cfg.threshold_stable
    Critical:  score < cfg.threshold_degrading

    Returns: "Stable" | "Degrading" | "Critical"
    """
    if score >= cfg.threshold_stable:
        return "Stable"
    elif score >= cfg.threshold_degrading:
        return "Degrading"
    else:
        return "Critical"


# ── Component functions ───────────────────────────────────────────────────────

def rms_health_component(
    rms_values: np.ndarray,
    baseline_rms: float,
    ref_max_rms: float,
) -> np.ndarray:
    """
    Convert a vector of RMS values to health components in [0, 1].

    Mapping:
        rms == baseline_rms → 1.0 (healthy)
        rms >= ref_max_rms  → 0.0 (fully degraded)
        linear interpolation between.

    Args:
        rms_values:   Per-snapshot RMS array.
        baseline_rms: Median RMS of the Normal training period (reference).
        ref_max_rms:  Maximum RMS of the Normal training period + small margin,
                      or the maximum seen in the Normal-only reference period.
                      Must be > baseline_rms.

    Returns:
        Float array in [0, 1], same length as rms_values.
    """
    rms_range = ref_max_rms - baseline_rms
    if rms_range < 1e-12:
        return np.ones(len(rms_values), dtype=np.float64)

    degradation = (rms_values - baseline_rms) / rms_range
    return np.clip(1.0 - degradation, 0.0, 1.0).astype(np.float64)


# Fixed reference scale for anomaly score normalization.
# Derived from Phase 6 IF training: decision_scores = -score_samples().
# IsolationForest.score_samples() on normal inliers typically ranges
# [-0.5, -0.1]; therefore -score_samples() on inliers is in [0.1, 0.5].
# Scores well above 0.5 indicate strong anomalies.
# Using 0.5 as the reference ceiling maps:
#   anomaly_score = 0.0  → health_component = 1.0  (normal, no anomaly)
#   anomaly_score = 0.5  → health_component = 0.0  (strong anomaly)
#   anomaly_score > 0.5  → health_component = 0.0  (clipped; extreme anomaly)
# This constant is configurable via HealthConfig.anomaly_ref_max_score.
_ANOMALY_REF_MAX_SCORE: float = 0.5


def anomaly_health_component(
    anomaly_scores: np.ndarray,
    ref_max_score: float = _ANOMALY_REF_MAX_SCORE,
) -> np.ndarray:
    """
    Convert Phase 6 anomaly scores to health components in [0, 1].

    Phase 6 score semantics:
        decision_scores() = -score_samples()  →  HIGHER = MORE ANOMALOUS

    Normalization:
        Scores are divided by a FIXED reference ceiling (ref_max_score)
        rather than the per-call maximum.  Using a per-call max would
        collapse every single-element call to 0.0, destroying ordering
        across independently scored batches/snapshots.

        The reference ceiling (default 0.5) is derived from the Phase 6
        Normal-only training period: IsolationForest decision_scores on
        normal inliers typically fall in [0.1, 0.5]; values above 0.5
        indicate clear anomalies.

    Leakage:
        The reference ceiling is a fixed constant drawn from the
        Phase 6 Normal training distribution.  No statistics from the
        complete Test 1 timeline are used.

    Mapping:
        anomaly_score <= 0.0         → health_component = 1.0
            (negative = more normal than training baseline)
        anomaly_score == ref_max_score → health_component = 0.0
        anomaly_score >  ref_max_score → health_component = 0.0  (clipped)
        Monotonically non-increasing throughout.

    Args:
        anomaly_scores:  Raw anomaly scores from AnomalyDetector.decision_scores().
        ref_max_score:   Fixed reference ceiling for normalization.
                         Defaults to _ANOMALY_REF_MAX_SCORE (0.5).

    Returns:
        Float array in [0, 1], same length as anomaly_scores.
        Returns an empty array (length 0) when input is empty.
    """
    # ── Guard: empty input ────────────────────────────────────────────────
    if len(anomaly_scores) == 0:
        return np.empty(0, dtype=np.float64)

    # Clip negative scores to 0 (more normal than training average → full health)
    clipped = np.clip(anomaly_scores, 0.0, None)

    # ── Fixed reference normalization (preserves ordering across calls) ────
    if ref_max_score < 1e-12:
        # Degenerate: treat all scores as normal
        return np.ones(len(anomaly_scores), dtype=np.float64)

    normalized = clipped / ref_max_score
    return np.clip(1.0 - normalized, 0.0, 1.0).astype(np.float64)


def fault_health_component(prob_faulty: np.ndarray) -> np.ndarray:
    """
    Convert RF faulty-class probabilities to health components in [0, 1].

    Mapping:
        prob_faulty = 0.0 → 1.0 (confident Normal)
        prob_faulty = 1.0 → 0.0 (confident Faulty)

    Args:
        prob_faulty: Array of P(Faulty) from RandomForestClassifier.predict_proba()

    Returns:
        Float array in [0, 1].
    """
    return np.clip(1.0 - prob_faulty, 0.0, 1.0).astype(np.float64)


# ── Explanation generator ─────────────────────────────────────────────────────

def generate_explanation(
    health_score: float,
    rms_comp: float,
    anomaly_comp: float,
    fault_comp: float,
    cfg: HealthConfig,
) -> str:
    """
    Generate a deterministic rule-based explanation for a health score.

    Does NOT use any LLM. Pure threshold logic.

    Returns: Human-readable string describing the primary driver of health change.
    """
    status = get_health_status(health_score, cfg)

    # Identify the weakest component relative to weights
    weighted = {
        "RMS degradation":    rms_comp * cfg.weight_rms,
        "anomaly severity":   anomaly_comp * cfg.weight_anomaly,
        "fault confidence":   fault_comp * cfg.weight_fault,
    }
    weakest = min(weighted, key=weighted.get)
    weakest_val = weighted[weakest]

    # Build explanation
    if status == "Stable":
        return (
            f"Equipment appears stable (health={health_score:.1f}). "
            f"All components within normal range. "
            f"Lowest contribution: {weakest} ({weakest_val*100/max(cfg.weight_rms, cfg.weight_anomaly, cfg.weight_fault):.1f}% of its weight)."
        )
    elif status == "Degrading":
        return (
            f"Equipment degrading (health={health_score:.1f}). "
            f"Primary driver of health reduction: {weakest}. "
            f"Monitor closely."
        )
    else:  # Critical
        drivers = [k for k, v in weighted.items() if v < 0.3 * max(weighted.values()) + 1e-9]
        driver_str = " and ".join(drivers) if drivers else weakest
        return (
            f"CRITICAL health detected (health={health_score:.1f}). "
            f"Severe reduction driven by: {driver_str}. "
            f"Immediate inspection recommended."
        )


# ── Main monitor class ────────────────────────────────────────────────────────

class HealthMonitor:
    """
    Phase 7 Health Monitoring Integration Layer.

    Loads Phase 5 RF model+scaler and Phase 6 IF model+scaler from disk.
    Scores all Test 1 snapshots chronologically.
    Produces a per-snapshot health DataFrame with full explainability.

    No new ML training occurs. Existing model artifacts are used as-is.

    IMPORTANT: If any required model artifact is missing, the monitor
    raises FileNotFoundError immediately — it does NOT silently retrain.
    """

    def __init__(
        self,
        rf_model_path: Path,
        rf_scaler_path: Path,
        if_model_path: Path,
        if_scaler_path: Path,
        config: Optional[HealthConfig] = None,
    ):
        self.cfg = config or DEFAULT_CONFIG

        # ── Load RF (Phase 5) ──────────────────────────────────────────────
        rf_model_path  = Path(rf_model_path)
        rf_scaler_path = Path(rf_scaler_path)
        if_model_path  = Path(if_model_path)
        if_scaler_path = Path(if_scaler_path)

        for p in [rf_model_path, rf_scaler_path, if_model_path, if_scaler_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required model artifact not found: {p}\n"
                    "Phase 7 cannot silently retrain models. "
                    "Run Phase 5 and Phase 6 pipelines first."
                )

        self.rf_clf    = FaultClassifier.load(rf_model_path)
        self.rf_scaler = joblib.load(rf_scaler_path)
        self.if_det    = AnomalyDetector.load(if_model_path)
        self.if_scaler = joblib.load(if_scaler_path)

        # Verify model feature names agree
        rf_feats = set(self.rf_clf.feature_names)
        if_feats = set(self.if_det.feature_names)
        if rf_feats != if_feats:
            raise ValueError(
                f"RF and IF models were trained on different feature sets. "
                f"RF has {len(rf_feats)} features, IF has {len(if_feats)} features. "
                f"Ensure both models were trained on the same feature CSV."
            )
        self.feature_names: list[str] = self.rf_clf.feature_names

    def score_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score all snapshots in a Phase 3 feature DataFrame chronologically.

        Args:
            df: Phase 3 feature DataFrame (all columns including metadata).

        Returns:
            DataFrame with per-snapshot health scores and all component values.
            Sorted by snapshot_index (chronological order preserved).
        """
        # ── Sort chronologically ───────────────────────────────────────────
        df = df.sort_values("snapshot_index", ascending=True).reset_index(drop=True)
        n_total = len(df)

        feat_cols = self.feature_names
        X_raw = df[feat_cols].values.astype(np.float64)

        # Validate: no NaN/Inf in input features
        if not np.isfinite(X_raw).all():
            n_bad = (~np.isfinite(X_raw)).sum()
            raise ValueError(
                f"Input feature matrix contains {n_bad} non-finite value(s). "
                "Clean the data before scoring."
            )

        # ── Establish RMS reference from Normal training period ────────────
        fault_start_idx = int(df["fault_start_idx"].iloc[0])
        normal_df = df[df["label"] == 0]
        n_ref = int(len(normal_df) * self.cfg.normal_reference_fraction)
        n_ref = max(1, n_ref)
        ref_normal_df = normal_df.iloc[:n_ref]

        # Identify RMS columns — mean of all __rms features across all bearings
        rms_cols = [c for c in feat_cols if c.endswith("__rms")]
        if not rms_cols:
            raise ValueError(
                "No RMS feature columns found (expected columns ending with '__rms'). "
                "Verify Phase 3 feature extraction output."
            )

        # Composite RMS: mean of all bearing channel RMS values per snapshot
        composite_rms = df[rms_cols].mean(axis=1).values
        ref_rms_values = ref_normal_df[rms_cols].mean(axis=1).values

        baseline_rms = float(np.median(ref_rms_values))
        # ref_max_rms: use 95th percentile of Normal training period
        # (avoids outliers pulling the range unnecessarily wide)
        ref_max_rms  = float(np.percentile(ref_rms_values, 95))
        # Ensure ref_max_rms > baseline_rms; fallback to 110% of baseline
        if ref_max_rms <= baseline_rms:
            ref_max_rms = baseline_rms * 1.10 + 1e-9

        # ── Scale for models ───────────────────────────────────────────────
        # Each model has its own scaler — apply independently
        X_rf = self.rf_scaler.transform(X_raw)
        X_if = self.if_scaler.transform(X_raw)

        # ── RF inference ───────────────────────────────────────────────────
        rf_proba    = self.rf_clf.model.predict_proba(X_rf)  # shape (n, 2)
        rf_pred     = self.rf_clf.model.predict(X_rf)        # int array
        prob_normal = rf_proba[:, 0]
        prob_faulty = rf_proba[:, 1]

        # ── IF inference ───────────────────────────────────────────────────
        if_scores = self.if_det.decision_scores(X_if)  # higher = more anomalous
        if_preds  = self.if_det.predict(X_if)          # 1=anomaly, 0=normal

        # ── Compute health components ──────────────────────────────────────
        rms_comp     = rms_health_component(composite_rms, baseline_rms, ref_max_rms)
        anomaly_comp = anomaly_health_component(
            if_scores, ref_max_score=self.cfg.anomaly_ref_max_score
        )
        fault_comp   = fault_health_component(prob_faulty)

        # ── Weighted health score ──────────────────────────────────────────
        raw_health = (
            self.cfg.weight_rms     * rms_comp     * 100.0 +
            self.cfg.weight_anomaly * anomaly_comp * 100.0 +
            self.cfg.weight_fault   * fault_comp   * 100.0
        )
        health_scores = np.clip(raw_health, 0.0, 100.0)

        # ── Status + explanation (row-by-row) ──────────────────────────────
        statuses     = [get_health_status(s, self.cfg) for s in health_scores]
        explanations = [
            generate_explanation(h, rc, ac, fc, self.cfg)
            for h, rc, ac, fc in zip(health_scores, rms_comp, anomaly_comp, fault_comp)
        ]

        # ── Build output DataFrame ─────────────────────────────────────────
        meta_present = [c for c in
            ["snapshot_index", "source_file", "timestamp", "label", "label_name"]
            if c in df.columns]

        result = df[meta_present].copy()
        result["composite_rms"]          = np.round(composite_rms, 6)
        result["anomaly_score"]          = np.round(if_scores, 6)
        result["anomaly_pred"]           = if_preds
        result["rf_pred"]                = rf_pred
        result["prob_normal"]            = np.round(prob_normal, 4)
        result["prob_faulty"]            = np.round(prob_faulty, 4)
        result["health_rms_component"]   = np.round(rms_comp * 100.0, 2)
        result["health_anomaly_component"]= np.round(anomaly_comp * 100.0, 2)
        result["health_fault_component"] = np.round(fault_comp * 100.0, 2)
        result["health_score"]           = np.round(health_scores, 2)
        result["health_status"]          = statuses
        result["explanation"]            = explanations

        # Verify no NaN/Inf leaked into health outputs
        numeric_cols = ["composite_rms", "anomaly_score", "prob_normal",
                        "prob_faulty", "health_rms_component",
                        "health_anomaly_component", "health_fault_component",
                        "health_score"]
        for col in numeric_cols:
            bad = ~np.isfinite(result[col].values)
            if bad.any():
                raise RuntimeError(
                    f"Non-finite values found in output column '{col}': "
                    f"{bad.sum()} bad values at indices {np.where(bad)[0][:5]}."
                )

        return result.sort_values("snapshot_index").reset_index(drop=True)

    @property
    def reference_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "rf_feature_count": len(self.rf_clf.feature_names),
            "if_feature_count": len(self.if_det.feature_names),
            "config": self.cfg.as_dict,
        }
