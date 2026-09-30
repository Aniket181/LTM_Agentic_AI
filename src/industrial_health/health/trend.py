"""
trend.py — Phase 7: Chronological Health Trend Analysis

PURPOSE:
    Analyze the chronological Health Score trajectory for Test 1.
    Produces a trend label per snapshot and a run-level summary.

    All logic is deterministic — no ML involved.

TREND LOGIC:
    1. Rolling slope over a configurable window of snapshots.
    2. Slope threshold determines IMPROVING / STABLE / DEGRADING.
    3. First degradation detection: first snapshot where the rolling
       trend switches from Stable to Degrading/Critical.
    4. Sustained degradation: consecutive snapshots in Degrading/Critical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum slope (health points per snapshot) to declare DEGRADING or IMPROVING
SLOPE_DEGRADING_THRESHOLD: float = -0.5  # slope < this → DEGRADING
SLOPE_IMPROVING_THRESHOLD: float =  0.5  # slope > this → IMPROVING


def rolling_trend(
    scores: np.ndarray,
    window: int = 50,
    slope_degrade: float = SLOPE_DEGRADING_THRESHOLD,
    slope_improve: float = SLOPE_IMPROVING_THRESHOLD,
) -> np.ndarray:
    """
    Compute a per-snapshot trend label using a rolling linear regression slope.

    Args:
        scores:         Health score array (chronological order, length N).
        window:         Rolling window size in snapshots.
        slope_degrade:  Slope threshold below which trend is DEGRADING.
        slope_improve:  Slope threshold above which trend is IMPROVING.

    Returns:
        String array of length N with values "Stable" | "Degrading" | "Improving".
        First (window-1) entries use expanding windows.
    """
    n = len(scores)
    labels = np.empty(n, dtype=object)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = scores[start : i + 1]
        if len(segment) < 2:
            labels[i] = "Stable"
            continue
        x = np.arange(len(segment), dtype=np.float64)
        # np.polyfit([x], [y], 1) returns [slope, intercept]
        slope = float(np.polyfit(x, segment, 1)[0])
        if slope < slope_degrade:
            labels[i] = "Degrading"
        elif slope > slope_improve:
            labels[i] = "Improving"
        else:
            labels[i] = "Stable"

    return labels


def first_degradation_snapshot(trend_labels: np.ndarray) -> int:
    """
    Return the snapshot index (position in array) where sustained degradation
    first appears. Returns -1 if no degradation detected.

    'First degradation' is defined as the first position where the trend label
    is 'Degrading' or 'Critical' (from health_status) and remains so for at
    least 5 consecutive snapshots.
    """
    n = len(trend_labels)
    required_run = 5
    for i in range(n - required_run + 1):
        segment = trend_labels[i : i + required_run]
        if all(t in {"Degrading", "Critical"} for t in segment):
            return i
    return -1


def compute_run_trend_summary(
    health_scores: np.ndarray,
    health_statuses: np.ndarray,
    snapshot_indices: np.ndarray,
    trend_labels: np.ndarray,
) -> dict:
    """
    Compute a run-level trend summary for the full Test 1 timeline.

    Returns a dict suitable for the Phase 7 JSON report.
    """
    first_deg_pos = first_degradation_snapshot(trend_labels)
    if first_deg_pos >= 0:
        first_deg_snapshot = int(snapshot_indices[first_deg_pos])
    else:
        first_deg_snapshot = -1

    n_stable    = int((health_statuses == "Stable").sum())
    n_degrading = int((health_statuses == "Degrading").sum())
    n_critical  = int((health_statuses == "Critical").sum())

    # Overall run trend: compare first-quarter mean vs last-quarter mean
    q1_end = len(health_scores) // 4
    q4_start = 3 * len(health_scores) // 4
    q1_mean = float(health_scores[:q1_end].mean()) if q1_end > 0 else float(health_scores[0])
    q4_mean = float(health_scores[q4_start:].mean()) if q4_start < len(health_scores) else float(health_scores[-1])
    overall_delta = q4_mean - q1_mean

    if overall_delta < -5.0:
        overall_trend = "Degrading"
    elif overall_delta > 5.0:
        overall_trend = "Improving"
    else:
        overall_trend = "Stable"

    return {
        "overall_trend":           overall_trend,
        "q1_mean_health":          round(q1_mean, 2),
        "q4_mean_health":          round(q4_mean, 2),
        "health_delta_q1_to_q4":  round(overall_delta, 2),
        "first_degradation_snapshot_pos":  first_deg_pos,
        "first_degradation_snapshot_idx":  first_deg_snapshot,
        "n_snapshots_stable":     n_stable,
        "n_snapshots_degrading":  n_degrading,
        "n_snapshots_critical":   n_critical,
        "pct_stable":    round(100 * n_stable    / len(health_scores), 1),
        "pct_degrading": round(100 * n_degrading / len(health_scores), 1),
        "pct_critical":  round(100 * n_critical  / len(health_scores), 1),
    }
