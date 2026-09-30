"""
diagnosis_agent.py — Fault Diagnosis Agent

Responsibilities:
  1. Load the trained Random Forest fault model
  2. Run inference on the current snapshot features
  3. Output: fault_class, confidence, probabilities
  4. Update shared state

Reads from state: current_snapshot
Writes to state:  fault_result, diagnosis_agent_status
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from industrial_health.models.fault_model import FaultDiagnosisModel


class DiagnosisAgent:
    """
    Runs Random Forest fault diagnosis on the current snapshot.
    """

    def __init__(self, models_path: Path):
        self.models_path = Path(models_path)
        self._fault_model: Optional[FaultDiagnosisModel] = None

    def _load_fault_model(self) -> FaultDiagnosisModel:
        if self._fault_model is None:
            model_path = self.models_path / "fault_model.pkl"
            self._fault_model = FaultDiagnosisModel.load(model_path)
        return self._fault_model

    def run(self, state: dict) -> dict:
        """
        Run fault diagnosis and update state.

        Reads:  state['current_snapshot']
        Writes: state['fault_result'], state['diagnosis_agent_status']
        """
        snapshot = state.get("current_snapshot")
        if not snapshot:
            logger.warning("No snapshot in state — skipping fault diagnosis")
            state["fault_result"] = {
                "fault_class": 0,
                "fault_label": "Unknown",
                "confidence": 0.0,
                "prob_normal": 0.5,
                "prob_faulty": 0.5,
            }
            state["diagnosis_agent_status"] = "SKIPPED_NO_SNAPSHOT"
            return state

        try:
            model = self._load_fault_model()
            fault_result = model.predict(snapshot)
            state["fault_result"] = fault_result
            state["feature_importance"] = model.get_feature_importance(top_n=10)
            state["diagnosis_agent_status"] = "COMPLETED"
            logger.info(
                f"Fault diagnosis: {fault_result['fault_label']} "
                f"(confidence={fault_result['confidence']:.3f})"
            )
        except FileNotFoundError:
            logger.warning("Fault model not found — using placeholder result")
            state["fault_result"] = {
                "fault_class": 0,
                "fault_label": "Unknown (model not trained)",
                "confidence": 0.0,
                "prob_normal": 0.5,
                "prob_faulty": 0.5,
            }
            state["feature_importance"] = []
            state["diagnosis_agent_status"] = "SKIPPED_NO_MODEL"

        # Update health result with correct fault result if health already computed
        if "health_result" in state and "anomaly_result" in state:
            from industrial_health.health.health_score import compute_health_score
            from industrial_health.health.health_score import get_trend
            health_result = compute_health_score(
                anomaly_result=state["anomaly_result"],
                fault_result=state["fault_result"],
                current_rms=state.get("rms_current"),
                baseline_rms=state.get("rms_baseline"),
                max_rms=state.get("rms_max"),
                score_history=state.get("score_history"),
            )
            state["health_result"] = health_result

        return state
