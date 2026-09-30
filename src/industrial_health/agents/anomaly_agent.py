"""
anomaly_agent.py — Phase 8: Anomaly Detection Agent

RESPONSIBILITIES:
  1. Load the Phase 6 Isolation Forest model (AnomalyDetector) and its scaler.
  2. Load the Phase 3 feature CSV.
  3. Run anomaly scoring on ALL snapshots chronologically.
  4. Write structured anomaly results to AgentState.

DOES NOT:
  - Retrain the Isolation Forest.
  - Modify Phase 6 model behaviour or score semantics.
  - Create a new anomaly model.

REUSES:
  - industrial_health.models.anomaly_detector.AnomalyDetector (Phase 6)

SCORE SEMANTICS (Phase 6, preserved exactly):
  decision_scores() = -score_samples()  →  HIGHER = MORE ANOMALOUS
  predict() returns 1 = anomaly, 0 = normal.

  These semantics MUST NOT be reversed or redefined here.

LABEL DISCLAIMER:
  Evaluation uses heuristic temporal labels. IF was trained on Normal-only data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from industrial_health.agents.state import AgentState
from industrial_health.models.anomaly_detector import AnomalyDetector, METADATA_COLS


AGENT_NAME = "AnomalyAgent"


class AnomalyAgent:
    """
    Runs Phase 6 Isolation Forest anomaly detection on all Test 1 snapshots.

    Args:
        features_dir:   Directory containing test{N}_features.csv.
        if_model_path:  Path to isolation_forest_test{N}.joblib.
        if_scaler_path: Path to phase6_test{N}_scaler.pkl.
    """

    def __init__(
        self,
        features_dir: Path,
        if_model_path: Path,
        if_scaler_path: Path,
    ):
        self.features_dir   = Path(features_dir)
        self.if_model_path  = Path(if_model_path)
        self.if_scaler_path = Path(if_scaler_path)
        self._detector: Optional[AnomalyDetector] = None
        self._scaler = None

    def _load_model(self) -> None:
        """Lazy-load IF detector and scaler (once per agent instance)."""
        if self._detector is None:
            for p in [self.if_model_path, self.if_scaler_path]:
                if not p.exists():
                    raise FileNotFoundError(
                        f"Phase 6 artifact not found: {p}\n"
                        "Run Phase 6 first. AnomalyAgent will NOT retrain."
                    )
            self._detector = AnomalyDetector.load(self.if_model_path)
            self._scaler   = joblib.load(self.if_scaler_path)

    def run(self, state: AgentState) -> AgentState:
        """
        Execute anomaly detection and update shared state.

        SCORE SEMANTICS (Phase 6):
          decision_scores() = -score_samples()  →  HIGHER = MORE ANOMALOUS

        Reads:  state.test_id
        Writes: state.anomaly_results, state.final_anomaly_score,
                state.final_anomaly_pred, state.n_anomaly_predicted

        Returns:
            Updated AgentState.
        """
        t0 = time.monotonic()
        test_id = state.test_id

        # ── Load model artifacts ────────────────────────────────────────────
        try:
            self._load_model()
        except FileNotFoundError as e:
            msg = str(e)
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        # ── Load feature CSV ────────────────────────────────────────────────
        csv_path = self.features_dir / f"test{test_id}_features.csv"
        if not csv_path.exists():
            msg = f"Feature CSV not found: {csv_path}"
            state.add_error(AGENT_NAME, msg)
            state.log_agent(AGENT_NAME, "failed",
                            duration_s=time.monotonic() - t0, error=msg)
            return state

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("snapshot_index", ascending=True).reset_index(drop=True)

        feat_cols = self._detector.feature_names
        X_raw    = df[feat_cols].values.astype(np.float64)
        X_scaled = self._scaler.transform(X_raw)

        # ── Run IF inference (all snapshots) ────────────────────────────────
        # SCORE SEMANTICS: decision_scores() = -score_samples()
        # HIGHER = MORE ANOMALOUS (Phase 6 semantics preserved exactly)
        if_scores = self._detector.decision_scores(X_scaled)   # higher = more anomalous
        if_preds  = self._detector.predict(X_scaled)           # 1=anomaly, 0=normal

        n_anomaly = int((if_preds == 1).sum())
        n_normal  = int((if_preds == 0).sum())

        last_idx = -1
        state.final_anomaly_score = float(round(float(if_scores[last_idx]), 4))
        state.final_anomaly_pred  = int(if_preds[last_idx])
        state.n_anomaly_predicted = n_anomaly

        state.anomaly_results = {
            "n_snapshots":        int(len(df)),
            "n_anomaly_predicted": n_anomaly,
            "n_normal_predicted":  n_normal,
            "pct_anomaly":         round(100 * n_anomaly / len(df), 2),
            "score_mean":          float(round(float(np.mean(if_scores)), 4)),
            "score_min":           float(round(float(np.min(if_scores)), 4)),
            "score_max":           float(round(float(np.max(if_scores)), 4)),
            "final_snapshot": {
                "snapshot_index":  int(df["snapshot_index"].iloc[last_idx]),
                "anomaly_score":   state.final_anomaly_score,
                "anomaly_pred":    state.final_anomaly_pred,
                "anomaly_label":   "Anomaly" if state.final_anomaly_pred == 1 else "Normal",
            },
            "score_semantics": (
                "decision_scores() = -score_samples(). HIGHER = MORE ANOMALOUS. "
                "anomaly_pred: 1=anomaly, 0=normal."
            ),
            "label_disclaimer": (
                "IF trained on Normal-only data (Phase 6). "
                "Evaluation uses heuristic temporal labels — not verified fault-onset annotations."
            ),
        }

        state.log_agent(AGENT_NAME, "completed", duration_s=time.monotonic() - t0)
        return state
