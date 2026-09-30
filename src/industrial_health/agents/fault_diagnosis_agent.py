"""
fault_diagnosis_agent.py — Phase 8: Fault Diagnosis Agent

RESPONSIBILITIES:
  1. Load the Phase 5 Random Forest model (FaultClassifier) and its scaler.
  2. Load the Phase 3 feature CSV.
  3. Run inference on ALL snapshots chronologically.
  4. Report per-snapshot RF predictions and the final-snapshot summary.
  5. Write structured fault results to AgentState.

DOES NOT:
  - Retrain the Random Forest.
  - Modify Phase 5 model behaviour.
  - Create a new classifier or scaler.

REUSES:
  - industrial_health.models.fault_classifier.FaultClassifier (Phase 5)

LABEL DISCLAIMER:
  The RF was trained on heuristic temporal labels (final 20% = Faulty).
  RF probabilities are not calibrated. They reflect model confidence under
  the Phase 5 experimental split protocol, not verified fault probabilities.

SCORE SEMANTICS:
  prob_faulty  = predict_proba()[:, 1]  →  P(Faulty class)
  fault_pred   = 0 (Normal) | 1 (Faulty)
  fault_confidence = max(prob_normal, prob_faulty)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from industrial_health.agents.state import AgentState
from industrial_health.models.fault_classifier import FaultClassifier, METADATA_COLS


AGENT_NAME = "FaultDiagnosisAgent"


class FaultDiagnosisAgent:
    """
    Runs Phase 5 Random Forest fault diagnosis on all Test 1 snapshots.

    Args:
        features_dir:   Directory containing test{N}_features.csv.
        rf_model_path:  Path to random_forest_test{N}.joblib.
        rf_scaler_path: Path to phase5_test{N}_scaler.pkl.
    """

    def __init__(
        self,
        features_dir: Path,
        rf_model_path: Path,
        rf_scaler_path: Path,
    ):
        self.features_dir   = Path(features_dir)
        self.rf_model_path  = Path(rf_model_path)
        self.rf_scaler_path = Path(rf_scaler_path)
        self._clf: Optional[FaultClassifier] = None
        self._scaler = None

    def _load_model(self) -> None:
        """Lazy-load RF classifier and scaler (once per agent instance)."""
        if self._clf is None:
            for p in [self.rf_model_path, self.rf_scaler_path]:
                if not p.exists():
                    raise FileNotFoundError(
                        f"Phase 5 artifact not found: {p}\n"
                        "Run Phase 5 first. FaultDiagnosisAgent will NOT retrain."
                    )
            self._clf    = FaultClassifier.load(self.rf_model_path)
            self._scaler = joblib.load(self.rf_scaler_path)

    def run(self, state: AgentState) -> AgentState:
        """
        Execute fault diagnosis and update shared state.

        Reads:  state.test_id, state.feature_csv_path (optional — reloads CSV
                if needed, using test_id)
        Writes: state.fault_results, state.final_rf_pred,
                state.final_prob_normal, state.final_prob_faulty,
                state.final_fault_label

        Returns:
            Updated AgentState.
        """
        t0 = time.monotonic()
        test_id  = state.test_id

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

        feat_cols = self._clf.feature_names
        X_raw = df[feat_cols].values.astype(np.float64)
        X_scaled = self._scaler.transform(X_raw)

        # ── Run RF inference (all snapshots) ────────────────────────────────
        rf_proba = self._clf.model.predict_proba(X_scaled)   # (n, 2)
        rf_pred  = self._clf.model.predict(X_scaled)          # int array

        prob_normal = rf_proba[:, 0]
        prob_faulty = rf_proba[:, 1]
        # fault_confidence: confidence in whichever class was predicted
        fault_confidence = np.where(rf_pred == 1, prob_faulty, prob_normal)

        # Summary for the final snapshot (most recent chronological point)
        last_idx = -1
        state.final_rf_pred     = int(rf_pred[last_idx])
        state.final_prob_normal = float(round(prob_normal[last_idx], 4))
        state.final_prob_faulty = float(round(prob_faulty[last_idx], 4))
        state.final_fault_label = "Faulty" if rf_pred[last_idx] == 1 else "Normal"

        n_faulty_pred = int((rf_pred == 1).sum())
        n_normal_pred = int((rf_pred == 0).sum())

        state.fault_results = {
            "n_snapshots":       int(len(df)),
            "n_faulty_predicted": n_faulty_pred,
            "n_normal_predicted": n_normal_pred,
            "pct_faulty_pred":   round(100 * n_faulty_pred / len(df), 2),
            "final_snapshot": {
                "snapshot_index":  int(df["snapshot_index"].iloc[last_idx]),
                "rf_pred":         int(rf_pred[last_idx]),
                "fault_label":     state.final_fault_label,
                "prob_normal":     state.final_prob_normal,
                "prob_faulty":     state.final_prob_faulty,
                "fault_confidence":float(round(float(fault_confidence[last_idx]), 4)),
            },
            "label_disclaimer": (
                "RF trained on heuristic temporal labels (final 20% = Faulty). "
                "Probabilities are not calibrated Bayesian fault probabilities."
            ),
        }

        state.log_agent(AGENT_NAME, "completed", duration_s=time.monotonic() - t0)
        return state
