"""
health_agent.py — Data & Health Agent

Responsibilities:
  1. Load pre-extracted feature dataset for the requested test
  2. Select the requested snapshot (by index)
  3. Run anomaly detection
  4. Compute equipment health score
  5. Compute degradation trend from RMS history
  6. Write results to shared state

Models used:
  - AnomalyDetectionModel (Isolation Forest)
  - health_score.compute_health_score()
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from industrial_health.models.anomaly_model import AnomalyDetectionModel
from industrial_health.health.health_score import compute_health_score, get_trend
from industrial_health.data.preprocess import load_dataset


class HealthAgent:
    """
    Loads feature data, runs anomaly detection, and computes health score.
    """

    def __init__(self, models_path: Path, features_path: Path):
        self.models_path = Path(models_path)
        self.features_path = Path(features_path)
        self._anomaly_model: Optional[AnomalyDetectionModel] = None

    def _load_anomaly_model(self) -> AnomalyDetectionModel:
        if self._anomaly_model is None:
            model_path = self.models_path / "anomaly_model.pkl"
            self._anomaly_model = AnomalyDetectionModel.load(model_path)
        return self._anomaly_model

    def _load_features(self, test_id: int) -> pd.DataFrame:
        """Load the pre-built feature dataset for the given test."""
        path = self.features_path / f"test{test_id}_features.csv"
        return load_dataset(path)

    def _get_snapshot(self, df: pd.DataFrame, snapshot_index: int) -> pd.Series:
        """Get a single snapshot row from the feature DataFrame."""
        if snapshot_index == -1:
            return df.iloc[-1]
        return df.iloc[snapshot_index]

    def run(self, state: dict) -> dict:
        """
        Execute health analysis and update the shared state.

        Writes to state:
          - features_df: full feature DataFrame for the test
          - current_snapshot: the selected snapshot row
          - snapshot_timestamp: timestamp of the snapshot
          - anomaly_result: anomaly detection output
          - rms_baseline: median RMS in normal period
          - rms_current: current snapshot RMS
          - rms_max: max RMS in the test
          - health_result: full health assessment
          - score_history: list of recent health scores
        """
        test_id = state["test_id"]
        snapshot_index = state.get("snapshot_index", -1)

        # Load features
        try:
            df = self._load_features(test_id)
        except FileNotFoundError:
            msg = (
                f"Feature dataset for test {test_id} not found. "
                "Run feature extraction first (Phase 3)."
            )
            logger.warning(msg)
            state["health_result"] = {"error": msg}
            state["health_agent_status"] = "SKIPPED_NO_DATA"
            return state

        state["features_df"] = df

        # Select snapshot
        snapshot = self._get_snapshot(df, snapshot_index)
        state["current_snapshot"] = snapshot.to_dict()
        state["snapshot_timestamp"] = str(snapshot.get("timestamp", "unknown"))

        # Compute RMS context from primary bearing channel
        rms_cols = [col for col in df.columns if "__rms" in col]
        if rms_cols:
            primary_rms_col = rms_cols[0]
            rms_series = df[primary_rms_col].values
            normal_mask = df.get("label", pd.Series([0] * len(df))).values == 0
            baseline_rms = float(np.median(rms_series[normal_mask])) if any(normal_mask) else float(np.median(rms_series[:int(len(rms_series) * 0.3)]))
            current_rms = float(snapshot.get(primary_rms_col, baseline_rms))
            max_rms = float(np.max(rms_series))
        else:
            baseline_rms = current_rms = max_rms = None

        state["rms_baseline"] = baseline_rms
        state["rms_current"] = current_rms
        state["rms_max"] = max_rms

        # Anomaly detection
        try:
            anomaly_model = self._load_anomaly_model()
            anomaly_result = anomaly_model.score(snapshot.to_dict())
        except FileNotFoundError:
            logger.warning("Anomaly model not found — using placeholder")
            anomaly_result = {
                "anomaly_score": 0.5,
                "anomaly_status": "UNKNOWN",
                "is_anomaly": False,
                "raw_score": 0.0,
            }

        state["anomaly_result"] = anomaly_result

        # Health score (fault_result may not be available yet — use placeholder)
        fault_result = state.get("fault_result", {
            "fault_class": 0,
            "fault_label": "Normal",
            "confidence": 0.5,
        })

        # Compute score history for trend
        recent_indices = list(range(max(0, snapshot_index - 10), snapshot_index if snapshot_index > 0 else len(df) - 1))
        score_history = []
        for idx in recent_indices[-5:]:
            if idx < len(df):
                pass  # Will be populated once model is fully trained
        score_history.append(0.0)  # placeholder until model is integrated

        health_result = compute_health_score(
            anomaly_result=anomaly_result,
            fault_result=fault_result,
            current_rms=current_rms,
            baseline_rms=baseline_rms,
            max_rms=max_rms,
            score_history=score_history,
        )
        state["health_result"] = health_result
        state["health_agent_status"] = "COMPLETED"
        return state
