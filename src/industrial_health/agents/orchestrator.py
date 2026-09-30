"""
orchestrator.py — Phase 8: Orchestrator Agent

PURPOSE:
    Coordinate the three Phase 8 agents in a deterministic sequential pipeline:

        1. DataHealthAgent     — load features + Phase 7 health scoring
        2. FaultDiagnosisAgent — Phase 5 RF fault prediction
        3. AnomalyAgent        — Phase 6 IF anomaly detection

    The orchestrator contains NO ML logic. It:
        - initialises the shared AgentState
        - calls agents in sequence
        - stops if a required upstream agent fails (configurable)
        - assembles the final structured result

ARCHITECTURE NOTE:
    Phase 8 introduces orchestration around existing deterministic ML and
    health-monitoring components. It does not introduce autonomous LLM
    reasoning or new machine-learning training.

    The architecture is deliberately simple and state-based so that it
    can be upgraded to LangGraph (or similar) in a later phase without
    changing the individual agent APIs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from industrial_health.agents.state import AgentState
from industrial_health.agents.data_health_agent import DataHealthAgent
from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
from industrial_health.agents.anomaly_agent import AnomalyAgent
from industrial_health.health.health_monitor import HealthConfig


class OrchestratorAgent:
    """
    Deterministic orchestrator for the Phase 8 agentic pipeline.

    Execution sequence (always in this order):
        1. DataHealthAgent        — Phase 7 health monitoring
        2. FaultDiagnosisAgent    — Phase 5 RF fault diagnosis
        3. AnomalyAgent           — Phase 6 IF anomaly detection

    Args:
        models_dir:     Directory containing saved model .joblib files.
        features_dir:   Directory containing test{N}_features.csv files.
        scalers_dir:    Directory containing .pkl scaler files.
        test_id:        IMS test ID (1 only in Phase 8).
        health_config:  HealthConfig for Phase 7 weights/thresholds.
        stop_on_failure: If True, halt pipeline when DataHealthAgent fails
                         (downstream agents require its output). Default True.
    """

    _EXECUTION_ORDER = [
        "DataHealthAgent",
        "FaultDiagnosisAgent",
        "AnomalyAgent",
    ]

    def __init__(
        self,
        models_dir: Path,
        features_dir: Path,
        scalers_dir: Path,
        test_id: int = 1,
        health_config: Optional[HealthConfig] = None,
        stop_on_failure: bool = True,
    ):
        self.models_dir     = Path(models_dir)
        self.features_dir   = Path(features_dir)
        self.scalers_dir    = Path(scalers_dir)
        self.test_id        = test_id
        self.health_config  = health_config
        self.stop_on_failure = stop_on_failure

        # ── Resolve artifact paths ──────────────────────────────────────────
        self._rf_model_path  = self.models_dir / f"random_forest_test{test_id}.joblib"
        self._rf_scaler_path = self.scalers_dir / f"phase5_test{test_id}_scaler.pkl"
        self._if_model_path  = self.models_dir / f"isolation_forest_test{test_id}.joblib"
        self._if_scaler_path = self.scalers_dir / f"phase6_test{test_id}_scaler.pkl"

        # ── Instantiate agents ──────────────────────────────────────────────
        self.data_health_agent = DataHealthAgent(
            features_dir=self.features_dir,
            rf_model_path=self._rf_model_path,
            rf_scaler_path=self._rf_scaler_path,
            if_model_path=self._if_model_path,
            if_scaler_path=self._if_scaler_path,
            config=self.health_config,
        )
        self.fault_diagnosis_agent = FaultDiagnosisAgent(
            features_dir=self.features_dir,
            rf_model_path=self._rf_model_path,
            rf_scaler_path=self._rf_scaler_path,
        )
        self.anomaly_agent = AnomalyAgent(
            features_dir=self.features_dir,
            if_model_path=self._if_model_path,
            if_scaler_path=self._if_scaler_path,
        )

    def run(self) -> AgentState:
        """
        Execute the full Phase 8 agentic pipeline.

        Returns:
            Populated AgentState with health, fault, and anomaly results,
            plus a complete execution log.
        """
        state = AgentState(test_id=self.test_id)
        state.status = "RUNNING"

        # ── Step 1: DataHealthAgent ─────────────────────────────────────────
        state = self.data_health_agent.run(state)
        if self.stop_on_failure and _agent_failed(state, "DataHealthAgent"):
            state.status = "FAILED"
            return state

        # ── Step 2: FaultDiagnosisAgent ────────────────────────────────────
        state = self.fault_diagnosis_agent.run(state)
        if self.stop_on_failure and _agent_failed(state, "FaultDiagnosisAgent"):
            state.status = "FAILED"
            return state

        # ── Step 3: AnomalyAgent ───────────────────────────────────────────
        state = self.anomaly_agent.run(state)

        # ── Finalize ───────────────────────────────────────────────────────
        state.status = "COMPLETED" if not state.errors else "COMPLETED_WITH_ERRORS"
        return state

    def build_final_result(self, state: AgentState) -> dict[str, Any]:
        """
        Assemble a structured final result dict from the completed AgentState.

        This is the primary output of the Phase 8 pipeline — a deterministic,
        serializable summary.

        Returns:
            Nested dict suitable for JSON serialization and display.
        """
        return {
            "test_id": state.test_id,
            "health": {
                "mean_health_score":   state.mean_health_score,
                "final_health_score":  state.final_health_score,
                "final_health_status": state.final_health_status,
                "health_trend":        state.final_health_trend,
                "status_counts":       state.status_counts,
            },
            "fault": {
                "final_rf_pred":       state.final_rf_pred,
                "predicted_fault":     state.final_fault_label,
                "normal_probability":  state.final_prob_normal,
                "fault_probability":   state.final_prob_faulty,
                "fault_confidence": (
                    state.final_prob_faulty if state.final_rf_pred == 1
                    else state.final_prob_normal
                ) if state.final_rf_pred is not None else None,
            },
            "anomaly": {
                "anomaly_score":      state.final_anomaly_score,
                "anomaly_prediction": state.final_anomaly_pred,
                "anomaly_label": (
                    "Anomaly" if state.final_anomaly_pred == 1 else "Normal"
                ) if state.final_anomaly_pred is not None else None,
                "n_anomaly_predicted": state.n_anomaly_predicted,
            },
            "execution": {
                "agents_executed":  [e["agent"] for e in state.execution_log],
                "execution_order":  self._EXECUTION_ORDER,
                "execution_status": state.status,
                "errors":           state.errors,
                "execution_log":    state.execution_log,
            },
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_failed(state: AgentState, agent_name: str) -> bool:
    """Return True if the most recent log entry for agent_name is 'failed'."""
    for entry in reversed(state.execution_log):
        if entry.get("agent") == agent_name:
            return entry.get("status") == "failed"
    return False
