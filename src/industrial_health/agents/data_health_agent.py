"""
data_health_agent.py — Phase 8: Data & Health Agent

RESPONSIBILITIES:
  1. Load the Phase 3 feature CSV for the requested test_id.
  2. Validate required columns.
  3. Invoke the Phase 7 HealthMonitor to score all snapshots chronologically.
  4. Run trend analysis (Phase 7 trend module).
  5. Write structured health results back to AgentState.

DOES NOT:
  - Retrain any model.
  - Fit any scaler.
  - Recompute the health formula.
  - Duplicate Phase 7 logic.

REUSES:
  - industrial_health.health.health_monitor.HealthMonitor  (Phase 7)
  - industrial_health.health.trend.rolling_trend          (Phase 7)
  - industrial_health.health.trend.compute_run_trend_summary (Phase 7)

PROTOCOL:
  If a required artifact is missing, agent records the error in state
  and sets status to "failed". It does NOT silently retrain.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from industrial_health.agents.state import AgentState
from industrial_health.health.health_monitor import HealthMonitor, HealthConfig
from industrial_health.health.trend import rolling_trend, compute_run_trend_summary


AGENT_NAME = "DataHealthAgent"


class DataHealthAgent:
    """
    Loads feature data and runs the Phase 7 health monitoring pipeline.

    Args:
        features_dir: Directory containing test{N}_features.csv files.
        rf_model_path:  Path to Phase 5 RF model (.joblib).
        rf_scaler_path: Path to Phase 5 RF scaler (.pkl).
        if_model_path:  Path to Phase 6 IF model (.joblib).
        if_scaler_path: Path to Phase 6 IF scaler (.pkl).
        config:         HealthConfig (weights, thresholds). Uses defaults if None.
    """

    def __init__(
        self,
        features_dir: Path,
        rf_model_path: Path,
        rf_scaler_path: Path,
        if_model_path: Path,
        if_scaler_path: Path,
        config: Optional[HealthConfig] = None,
    ):
        self.features_dir   = Path(features_dir)
        self.rf_model_path  = Path(rf_model_path)
        self.rf_scaler_path = Path(rf_scaler_path)
        self.if_model_path  = Path(if_model_path)
        self.if_scaler_path = Path(if_scaler_path)
        self.config         = config
        self._monitor: Optional[HealthMonitor] = None

    def _get_monitor(self) -> HealthMonitor:
        """Lazy-load HealthMonitor (loads model artifacts on first call)."""
        if self._monitor is None:
            self._monitor = HealthMonitor(
                rf_model_path=self.rf_model_path,
                rf_scaler_path=self.rf_scaler_path,
                if_model_path=self.if_model_path,
                if_scaler_path=self.if_scaler_path,
                config=self.config,
            )
        return self._monitor

    def run(self, state: AgentState) -> AgentState:
        """
        Execute health monitoring and update shared state.

        Reads:  state.test_id
        Writes: state.feature_csv_path, state.n_snapshots, state.n_features,
                state.fault_start_idx, state.health_results,
                state.mean_health_score, state.final_health_score,
                state.final_health_status, state.final_health_trend,
                state.status_counts

        Returns:
            Updated AgentState.
        """
        t0 = time.monotonic()
        test_id = state.test_id
        csv_path = self.features_dir / f"test{test_id}_features.csv"

        # ── Validate CSV ────────────────────────────────────────────────────
        if not csv_path.exists():
            msg = (
                f"Feature CSV not found: {csv_path}. "
                "Run Phase 3 feature extraction first."
            )
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        # ── Load features ───────────────────────────────────────────────────
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        required = ["snapshot_index", "label", "fault_start_idx"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            msg = f"Feature CSV missing required columns: {missing}"
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        state.feature_csv_path = str(csv_path)
        state.n_snapshots = len(df)
        feat_cols = [c for c in df.columns
                     if c not in {"test_id","source_file","timestamp","n_channels",
                                  "snapshot_index","label","label_name",
                                  "label_method","fault_start_idx"}]
        state.n_features      = len(feat_cols)
        state.fault_start_idx = int(df["fault_start_idx"].iloc[0])

        # ── Load monitor (Phase 7) ──────────────────────────────────────────
        try:
            monitor = self._get_monitor()
        except FileNotFoundError as e:
            msg = str(e)
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        # ── Score all snapshots ─────────────────────────────────────────────
        try:
            result_df = monitor.score_dataset(df)
        except Exception as e:
            msg = f"HealthMonitor.score_dataset() failed: {e}"
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        # ── Trend analysis (Phase 7 trend module) ───────────────────────────
        hs = result_df["health_score"].values
        cfg = self._monitor.cfg if self._monitor else HealthConfig()
        trend_labels = rolling_trend(hs, window=cfg.trend_window)
        result_df["trend_label"] = trend_labels

        trend_summary = compute_run_trend_summary(
            hs,
            result_df["health_status"].values,
            result_df["snapshot_index"].values,
            trend_labels,
        )

        # ── Populate state ──────────────────────────────────────────────────
        status_vc = result_df["health_status"].value_counts()
        state.status_counts = {
            "Stable":    int(status_vc.get("Stable",    0)),
            "Degrading": int(status_vc.get("Degrading", 0)),
            "Critical":  int(status_vc.get("Critical",  0)),
        }
        state.mean_health_score   = float(np.mean(hs))
        state.final_health_score  = float(hs[-1])
        state.final_health_status = str(result_df["health_status"].iloc[-1])
        state.final_health_trend  = trend_summary["overall_trend"]

        # Store full per-snapshot results for downstream agents/reporting
        state.health_results = {
            "n_snapshots":      int(len(result_df)),
            "health_min":       float(hs.min()),
            "health_max":       float(hs.max()),
            "health_mean":      float(hs.mean()),
            "health_median":    float(np.median(hs)),
            "health_std":       float(hs.std()),
            "status_counts":    state.status_counts,
            "trend_summary":    trend_summary,
            "n_rms_cols":       len([c for c in feat_cols if c.endswith("__rms")]),
        }

        state.log_agent(AGENT_NAME, "completed", duration_s=time.monotonic() - t0)
        return state
