"""
health_score.py — Equipment Health Score Calculator

Computes an explainable 0–100 health score from three components:

  1. Anomaly Component (40% weight):
     - Based on Isolation Forest normalized anomaly score.
     - 0 = highly anomalous → contributes 0 to health.
     - 1 = normal → contributes full weight.

  2. Fault Component (40% weight):
     - Based on Random Forest fault prediction confidence.
     - If Normal prediction: contributes proportionally to confidence.
     - If Faulty prediction: penalizes based on fault confidence.

  3. Degradation Component (20% weight):
     - Based on RMS trend relative to the test's baseline.
     - Increasing RMS → decreasing health.
     - Requires degradation context (RMS history).

Health thresholds (configurable via .env):
  90–100: Healthy
  70–89:  Normal / Slight Degradation
  40–69:  Degraded
   0–39:  Critical

These thresholds are initial engineering rules.
Not scientifically validated — treat as configurable indicators.

Risk levels:
  Health 90-100 → LOW
  Health 70-89  → MEDIUM (monitor closely)
  Health 40-69  → HIGH (schedule maintenance)
  Health 0-39   → CRITICAL (immediate action required)
"""

from typing import Optional

import numpy as np
from loguru import logger


# ── Configurable Weights ──────────────────────────────────────────────────────
WEIGHT_ANOMALY     = 0.40
WEIGHT_FAULT       = 0.40
WEIGHT_DEGRADATION = 0.20

# ── Health Thresholds ─────────────────────────────────────────────────────────
THRESHOLDS = {
    "healthy":   (90, 100),
    "normal":    (70,  89),
    "degraded":  (40,  69),
    "critical":  (0,   39),
}

STATUS_LABELS = {
    "healthy":  "HEALTHY",
    "normal":   "NORMAL",
    "degraded": "DEGRADED",
    "critical": "CRITICAL",
}

RISK_LEVELS = {
    "healthy":  "LOW",
    "normal":   "MEDIUM",
    "degraded": "HIGH",
    "critical": "CRITICAL",
}


def _compute_anomaly_component(anomaly_score: float) -> float:
    """
    Convert normalized anomaly score (0–1) to health contribution (0–100).
    0 = most anomalous → 0 health contribution.
    1 = most normal → 100 health contribution.
    """
    return float(np.clip(anomaly_score, 0.0, 1.0)) * 100.0


def _compute_fault_component(fault_prediction: dict) -> float:
    """
    Convert fault prediction result to health contribution (0–100).

    If predicted Normal: health contribution = confidence × 100
    If predicted Faulty: health contribution = (1 - confidence) × 100
                        (penalize proportionally to fault confidence)
    """
    if fault_prediction["fault_class"] == 0:  # Normal
        return fault_prediction["confidence"] * 100.0
    else:  # Faulty
        return (1.0 - fault_prediction["confidence"]) * 100.0


def _compute_degradation_component(
    current_rms: float,
    baseline_rms: float,
    max_rms: float,
) -> float:
    """
    Compute degradation health component from RMS trend.

    Maps current RMS relative to the test's RMS range to health (0–100).
    current_rms == baseline_rms → 100 (healthy baseline)
    current_rms == max_rms      →   0 (worst degradation seen)

    Args:
        current_rms:  RMS at the current snapshot.
        baseline_rms: Median RMS during early (Normal) period.
        max_rms:      Maximum RMS seen in the test (or estimated maximum).

    Returns:
        Degradation health component: 0–100.
    """
    rms_range = max_rms - baseline_rms
    if rms_range < 1e-10:
        return 100.0  # No degradation observable

    # Normalize: how far current RMS is from baseline relative to full range
    degradation_fraction = (current_rms - baseline_rms) / rms_range
    degradation_fraction = float(np.clip(degradation_fraction, 0.0, 1.0))

    return (1.0 - degradation_fraction) * 100.0


def get_status(score: float) -> str:
    """Map a health score (0–100) to a status label."""
    if score >= 90:
        return STATUS_LABELS["healthy"]
    elif score >= 70:
        return STATUS_LABELS["normal"]
    elif score >= 40:
        return STATUS_LABELS["degraded"]
    else:
        return STATUS_LABELS["critical"]


def get_risk_level(score: float) -> str:
    """Map a health score (0–100) to a risk level."""
    if score >= 90:
        return RISK_LEVELS["healthy"]
    elif score >= 70:
        return RISK_LEVELS["normal"]
    elif score >= 40:
        return RISK_LEVELS["degraded"]
    else:
        return RISK_LEVELS["critical"]


def get_trend(
    score_history: list[float],
    window: int = 5,
) -> str:
    """
    Determine health trend from recent score history.

    Args:
        score_history: List of recent health scores (oldest first).
        window: Number of recent scores to consider.

    Returns:
        'IMPROVING', 'STABLE', or 'DEGRADING'
    """
    if len(score_history) < 2:
        return "STABLE"

    recent = score_history[-window:]
    if len(recent) < 2:
        return "STABLE"

    # Linear regression slope
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])

    if slope > 1.0:
        return "IMPROVING"
    elif slope < -1.0:
        return "DEGRADING"
    else:
        return "STABLE"


def compute_health_score(
    anomaly_result: dict,
    fault_result: dict,
    current_rms: Optional[float] = None,
    baseline_rms: Optional[float] = None,
    max_rms: Optional[float] = None,
    score_history: Optional[list[float]] = None,
) -> dict:
    """
    Compute the complete equipment health assessment.

    Args:
        anomaly_result:  Output from AnomalyDetectionModel.score().
        fault_result:    Output from FaultDiagnosisModel.predict().
        current_rms:     Current snapshot RMS (for degradation component).
        baseline_rms:    Healthy baseline RMS.
        max_rms:         Maximum RMS seen in test (worst case).
        score_history:   Recent health score list for trend calculation.

    Returns:
        Comprehensive health assessment dictionary.
    """
    # ── Component 1: Anomaly ───────────────────────────────────────────────
    anomaly_component = _compute_anomaly_component(anomaly_result["anomaly_score"])

    # ── Component 2: Fault ────────────────────────────────────────────────
    fault_component = _compute_fault_component(fault_result)

    # ── Component 3: Degradation ──────────────────────────────────────────
    if current_rms is not None and baseline_rms is not None and max_rms is not None:
        degradation_component = _compute_degradation_component(
            current_rms, baseline_rms, max_rms
        )
        used_degradation = True
    else:
        # Redistribute weights if degradation data unavailable
        degradation_component = (anomaly_component + fault_component) / 2.0
        used_degradation = False

    # ── Weighted Health Score ─────────────────────────────────────────────
    health_score = (
        WEIGHT_ANOMALY     * anomaly_component +
        WEIGHT_FAULT       * fault_component +
        WEIGHT_DEGRADATION * degradation_component
    )
    health_score = float(np.clip(health_score, 0.0, 100.0))

    # ── Status and Risk ───────────────────────────────────────────────────
    status = get_status(health_score)
    risk_level = get_risk_level(health_score)
    trend = get_trend(score_history or [health_score])

    result = {
        "health_score":           round(health_score, 1),
        "status":                 status,
        "risk_level":             risk_level,
        "trend":                  trend,
        "components": {
            "anomaly_component":      round(anomaly_component, 1),
            "fault_component":        round(fault_component, 1),
            "degradation_component":  round(degradation_component, 1),
        },
        "weights": {
            "anomaly":     WEIGHT_ANOMALY,
            "fault":       WEIGHT_FAULT,
            "degradation": WEIGHT_DEGRADATION,
        },
        "degradation_data_available": used_degradation,
        "anomaly_status":   anomaly_result["anomaly_status"],
        "fault_label":      fault_result["fault_label"],
        "fault_confidence": round(fault_result["confidence"], 4),
    }

    logger.info(
        f"Health Score: {health_score:.1f} | Status: {status} | Risk: {risk_level} | Trend: {trend}"
    )
    return result
