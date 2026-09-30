"""
test_agents.py — Phase 8: Comprehensive agent tests.

All unit tests use synthetic data and mocking — they do NOT require
real model artifacts or the full Test 1 CSV. Integration tests are
run by the CI pipeline script.

Test groups:
  1.  AgentState initialization
  2.  AgentState execution log
  3.  AgentState to_dict serialization
  4.  DataHealthAgent with mocked HealthMonitor
  5.  FaultDiagnosisAgent with mocked RF model
  6.  AnomalyAgent with mocked IF model
  7.  OrchestratorAgent executes agents in correct order
  8.  Successful pipeline produces structured output
  9.  Execution log records each agent
  10. Missing model artifact produces explicit error
  11. Agent failure propagates safely (stop_on_failure)
  12. No agent silently retrains models
  13. Phase 5 semantics remain unchanged (prob encoding)
  14. Phase 6 anomaly semantics remain unchanged (higher=more anomalous)
"""

import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 200, n_feat: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2003, 10, 22)
    fault_start = int(n * 0.8)
    rows = []
    for idx in range(n):
        label = 0 if idx < fault_start else 1
        row = {
            "test_id":         1,
            "source_file":     f"f_{idx:05d}",
            "timestamp":       base + timedelta(minutes=10 * idx),
            "n_channels":      4,
            "snapshot_index":  idx,
            "label":           label,
            "label_name":      "Normal" if label == 0 else "Faulty",
            "label_method":    "heuristic",
            "fault_start_idx": fault_start,
        }
        for i in range(n_feat):
            if i % 2 == 0:
                row[f"b1_ch1__rms{i}"] = float(rng.uniform(0.1, 0.5))
            else:
                row[f"b1_ch1__feat{i}"] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


# ── 1. AgentState initialization ─────────────────────────────────────────────

class TestAgentStateInit:

    def test_default_test_id(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.test_id == 1

    def test_custom_test_id(self):
        from industrial_health.agents.state import AgentState
        state = AgentState(test_id=1)
        assert state.test_id == 1

    def test_initial_status_initialized(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.status == "INITIALIZED"

    def test_initial_errors_empty(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.errors == []

    def test_initial_log_empty(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.execution_log == []

    def test_health_fields_none_by_default(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.health_results is None
        assert state.final_health_score is None

    def test_fault_fields_none_by_default(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.fault_results is None
        assert state.final_rf_pred is None

    def test_anomaly_fields_none_by_default(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        assert state.anomaly_results is None
        assert state.final_anomaly_score is None


# ── 2. Execution log ──────────────────────────────────────────────────────────

class TestExecutionLog:

    def test_log_agent_adds_entry(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        state.log_agent("MyAgent", "completed", duration_s=1.23)
        assert len(state.execution_log) == 1
        entry = state.execution_log[0]
        assert entry["agent"]  == "MyAgent"
        assert entry["status"] == "completed"
        assert entry["duration_s"] == pytest.approx(1.23, abs=0.01)

    def test_log_agent_error_included(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        state.log_agent("BadAgent", "failed", error="file not found")
        assert "error" in state.execution_log[0]
        assert "file not found" in state.execution_log[0]["error"]

    def test_multiple_agents_logged_in_order(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        state.log_agent("AgentA", "completed")
        state.log_agent("AgentB", "completed")
        state.log_agent("AgentC", "completed")
        names = [e["agent"] for e in state.execution_log]
        assert names == ["AgentA", "AgentB", "AgentC"]


# ── 3. AgentState.to_dict ─────────────────────────────────────────────────────

class TestAgentStateToDict:

    def test_to_dict_returns_dict(self):
        from industrial_health.agents.state import AgentState
        state = AgentState(test_id=1)
        d = state.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        d = state.to_dict()
        for key in ["test_id", "health_summary", "fault_summary",
                    "anomaly_summary", "execution_log", "errors", "status"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_no_model_objects(self):
        """State dict must never contain non-serializable ML objects."""
        import json
        from industrial_health.agents.state import AgentState
        state = AgentState(test_id=1)
        state.mean_health_score = 75.3
        state.final_health_status = "Stable"
        # Should not raise
        json.dumps(state.to_dict())


# ── 4. DataHealthAgent (mocked HealthMonitor) ─────────────────────────────────

class TestDataHealthAgent:

    def _make_mock_result_df(self, n: int = 100) -> pd.DataFrame:
        idx = np.arange(n)
        hs  = np.linspace(90, 50, n)
        # Split into Stable / Degrading proportionally to n (avoid hardcoded 60+40)
        n_stable    = int(n * 0.6)
        n_degrading = n - n_stable          # exact remainder — sum is always n
        return pd.DataFrame({
            "snapshot_index":          idx,
            "health_score":            hs,
            "health_status":           ["Stable"] * n_stable + ["Degrading"] * n_degrading,
            "health_rms_component":    np.full(n, 80.0),
            "health_anomaly_component":np.full(n, 70.0),
            "health_fault_component":  np.full(n, 75.0),
            "anomaly_score":           np.full(n, 0.15),
            "anomaly_pred":            np.zeros(n, dtype=int),
            "rf_pred":                 np.zeros(n, dtype=int),
            "prob_normal":             np.full(n, 0.8),
            "prob_faulty":             np.full(n, 0.2),
            "composite_rms":           np.full(n, 0.25),
            "label":                   np.zeros(n, dtype=int),
            "explanation":             ["OK"] * n,
        })

    def test_run_updates_state_health_results(self, tmp_path):
        from industrial_health.agents.data_health_agent import DataHealthAgent
        from industrial_health.agents.state import AgentState

        # Write synthetic CSV
        df = _make_feature_df(n=100)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        result_df = self._make_mock_result_df(n=100)

        agent = DataHealthAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "rf_sc.pkl",
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )

        # Mock HealthMonitor so we don't need real model files
        mock_monitor = MagicMock()
        mock_monitor.score_dataset.return_value = result_df
        mock_monitor.cfg.trend_window = 20
        agent._monitor = mock_monitor

        state = AgentState(test_id=1)
        state = agent.run(state)

        assert state.health_results is not None
        assert state.mean_health_score is not None
        assert state.final_health_score is not None
        assert state.final_health_status is not None
        assert state.status_counts is not None

    def test_missing_csv_sets_error(self, tmp_path):
        from industrial_health.agents.data_health_agent import DataHealthAgent
        from industrial_health.agents.state import AgentState

        agent = DataHealthAgent(
            features_dir=tmp_path,  # no CSV in tmp_path
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "rf_sc.pkl",
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )
        state = AgentState(test_id=1)
        state = agent.run(state)

        assert len(state.errors) > 0
        assert state.health_results is None

    def test_execution_log_recorded(self, tmp_path):
        from industrial_health.agents.data_health_agent import DataHealthAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=80)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        result_df = self._make_mock_result_df(n=80)

        agent = DataHealthAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "rf_sc.pkl",
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )
        mock_monitor = MagicMock()
        mock_monitor.score_dataset.return_value = result_df
        mock_monitor.cfg.trend_window = 10
        agent._monitor = mock_monitor

        state = AgentState(test_id=1)
        state = agent.run(state)

        names = [e["agent"] for e in state.execution_log]
        assert "DataHealthAgent" in names


# ── 5. FaultDiagnosisAgent (mocked RF) ───────────────────────────────────────

class TestFaultDiagnosisAgent:

    def test_run_updates_fault_results(self, tmp_path):
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=100)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "rf_sc.pkl",
        )

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = len(df)

        mock_clf = MagicMock()
        mock_clf.feature_names = feat_cols
        mock_clf.model.predict_proba.return_value = np.column_stack([
            np.full(n, 0.8), np.full(n, 0.2)
        ])
        mock_clf.model.predict.return_value = np.zeros(n, dtype=int)

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._clf    = mock_clf
        agent._scaler = mock_scaler

        state = AgentState(test_id=1)
        state = agent.run(state)

        assert state.fault_results is not None
        assert state.final_fault_label is not None
        assert state.final_prob_normal is not None
        assert state.final_prob_faulty is not None

    def test_missing_model_sets_error(self, tmp_path):
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        (tmp_path / "test1_features.csv").write_text(df.to_csv(index=False))

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "nonexistent_rf.joblib",
            rf_scaler_path=tmp_path / "nonexistent_sc.pkl",
        )
        state = AgentState(test_id=1)
        state = agent.run(state)

        assert len(state.errors) > 0
        assert state.fault_results is None

    def test_phase5_prob_encoding(self, tmp_path):
        """
        Phase 5 RF predict_proba() returns [P(Normal), P(Faulty)].
        FaultDiagnosisAgent must use column 0 for prob_normal, column 1 for prob_faulty.
        """
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = 50

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "sc.pkl",
        )
        # Proba: last row = [0.3, 0.7] → prob_normal=0.3, prob_faulty=0.7
        proba = np.tile([0.8, 0.2], (n, 1))
        proba[-1] = [0.3, 0.7]

        mock_clf = MagicMock()
        mock_clf.feature_names = feat_cols
        mock_clf.model.predict_proba.return_value = proba
        mock_clf.model.predict.return_value = np.zeros(n, dtype=int)

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._clf    = mock_clf
        agent._scaler = mock_scaler

        state = AgentState(test_id=1)
        state = agent.run(state)

        # Last-row values — column 0 = prob_normal, column 1 = prob_faulty
        assert state.final_prob_normal == pytest.approx(0.3, abs=0.01)
        assert state.final_prob_faulty == pytest.approx(0.7, abs=0.01)


# ── 6. AnomalyAgent (mocked IF) ──────────────────────────────────────────────

class TestAnomalyAgent:

    def test_run_updates_anomaly_results(self, tmp_path):
        from industrial_health.agents.anomaly_agent import AnomalyAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=100)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = len(df)

        agent = AnomalyAgent(
            features_dir=tmp_path,
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )

        mock_det = MagicMock()
        mock_det.feature_names = feat_cols
        mock_det.decision_scores.return_value = np.random.default_rng(9).uniform(0, 0.6, n)
        mock_det.predict.return_value = np.zeros(n, dtype=int)

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._detector = mock_det
        agent._scaler   = mock_scaler

        state = AgentState(test_id=1)
        state = agent.run(state)

        assert state.anomaly_results is not None
        assert state.final_anomaly_score is not None
        assert state.final_anomaly_pred is not None

    def test_missing_if_model_sets_error(self, tmp_path):
        from industrial_health.agents.anomaly_agent import AnomalyAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        (tmp_path / "test1_features.csv").write_text(df.to_csv(index=False))

        agent = AnomalyAgent(
            features_dir=tmp_path,
            if_model_path=tmp_path / "no_model.joblib",
            if_scaler_path=tmp_path / "no_sc.pkl",
        )
        state = AgentState(test_id=1)
        state = agent.run(state)

        assert len(state.errors) > 0
        assert state.anomaly_results is None

    def test_phase6_anomaly_semantics_higher_is_more_anomalous(self, tmp_path):
        """
        Phase 6 semantics: decision_scores() = -score_samples().
        AnomalyAgent must pass raw decision_scores to state without negating again.
        A higher stored anomaly_score must mean more anomalous.
        """
        from industrial_health.agents.anomaly_agent import AnomalyAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = 50

        agent = AnomalyAgent(
            features_dir=tmp_path,
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )
        # Simulate: last snapshot has score 0.9 (high = anomalous)
        scores = np.full(n, 0.1)
        scores[-1] = 0.9   # last snapshot is very anomalous

        mock_det = MagicMock()
        mock_det.feature_names = feat_cols
        mock_det.decision_scores.return_value = scores
        mock_det.predict.return_value = np.zeros(n, dtype=int)
        mock_det.predict.return_value[-1] = 1   # last = anomaly

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._detector = mock_det
        agent._scaler   = mock_scaler

        state = AgentState(test_id=1)
        state = agent.run(state)

        # The stored score must be 0.9, not further modified
        assert state.final_anomaly_score == pytest.approx(0.9, abs=0.01), (
            "AnomalyAgent must store decision_scores() directly. "
            "Higher = more anomalous per Phase 6 semantics."
        )
        assert state.final_anomaly_pred == 1


# ── 7. Orchestrator executes in correct order ─────────────────────────────────

class TestOrchestratorOrder:

    def test_execution_order_is_documented(self, tmp_path):
        from industrial_health.agents.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent(
            models_dir=tmp_path / "models",
            features_dir=tmp_path / "features",
            scalers_dir=tmp_path / "scalers",
            test_id=1,
        )
        assert orch._EXECUTION_ORDER == [
            "DataHealthAgent",
            "FaultDiagnosisAgent",
            "AnomalyAgent",
        ]

    def test_agents_logged_in_order_on_success(self, tmp_path):
        """
        When all agents succeed, execution_log entries must appear in
        DataHealthAgent → FaultDiagnosisAgent → AnomalyAgent order.
        """
        from industrial_health.agents.orchestrator import OrchestratorAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=100)
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        (features_dir / "test1_features.csv").write_text(df.to_csv(index=False))

        # Build a real orchestrator but inject mocked agents
        orch = OrchestratorAgent(
            models_dir=tmp_path / "models",
            features_dir=features_dir,
            scalers_dir=tmp_path / "scalers",
            test_id=1,
        )

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = len(df)

        # Mock the three inner agents
        def make_success_agent(agent_name: str):
            def run(state: AgentState) -> AgentState:
                state.log_agent(agent_name, "completed", duration_s=0.01)
                return state
            m = MagicMock()
            m.run.side_effect = run
            return m

        orch.data_health_agent    = make_success_agent("DataHealthAgent")
        orch.fault_diagnosis_agent = make_success_agent("FaultDiagnosisAgent")
        orch.anomaly_agent         = make_success_agent("AnomalyAgent")

        state = orch.run()
        logged_names = [e["agent"] for e in state.execution_log]
        assert logged_names == ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"]


# ── 8. Successful pipeline produces structured output ─────────────────────────

class TestSuccessfulPipeline:

    def test_build_final_result_has_required_keys(self, tmp_path):
        from industrial_health.agents.orchestrator import OrchestratorAgent
        from industrial_health.agents.state import AgentState

        orch = OrchestratorAgent(
            models_dir=tmp_path,
            features_dir=tmp_path,
            scalers_dir=tmp_path,
            test_id=1,
        )
        state = AgentState(test_id=1)
        state.status = "COMPLETED"
        state.mean_health_score   = 72.5
        state.final_health_score  = 65.3
        state.final_health_status = "Degrading"
        state.final_health_trend  = "Degrading"
        state.status_counts       = {"Stable": 100, "Degrading": 50, "Critical": 10}
        state.final_rf_pred       = 0
        state.final_prob_normal   = 0.82
        state.final_prob_faulty   = 0.18
        state.final_fault_label   = "Normal"
        state.final_anomaly_score = 0.23
        state.final_anomaly_pred  = 0
        state.n_anomaly_predicted = 42
        state.log_agent("DataHealthAgent",    "completed")
        state.log_agent("FaultDiagnosisAgent","completed")
        state.log_agent("AnomalyAgent",        "completed")

        result = orch.build_final_result(state)
        for key in ["test_id", "health", "fault", "anomaly", "execution"]:
            assert key in result, f"Missing top-level key: {key}"
        for key in ["final_health_score", "final_health_status", "health_trend"]:
            assert key in result["health"], f"Missing health key: {key}"
        for key in ["predicted_fault", "normal_probability", "fault_probability"]:
            assert key in result["fault"], f"Missing fault key: {key}"
        for key in ["anomaly_score", "anomaly_prediction"]:
            assert key in result["anomaly"], f"Missing anomaly key: {key}"


# ── 9. Execution log records each agent ───────────────────────────────────────

class TestExecutionLogRecording:

    def test_each_agent_appears_in_log(self, tmp_path):
        from industrial_health.agents.orchestrator import OrchestratorAgent
        from industrial_health.agents.state import AgentState

        orch = OrchestratorAgent(
            models_dir=tmp_path,
            features_dir=tmp_path,
            scalers_dir=tmp_path,
            test_id=1,
        )

        state = AgentState(test_id=1)
        state.log_agent("DataHealthAgent",    "completed", duration_s=1.1)
        state.log_agent("FaultDiagnosisAgent","completed", duration_s=0.5)
        state.log_agent("AnomalyAgent",        "completed", duration_s=0.3)

        logged = {e["agent"] for e in state.execution_log}
        assert "DataHealthAgent"     in logged
        assert "FaultDiagnosisAgent" in logged
        assert "AnomalyAgent"         in logged

    def test_failed_agent_has_failed_status(self):
        from industrial_health.agents.state import AgentState
        state = AgentState()
        state.log_agent("DataHealthAgent", "failed", error="file missing")
        entry = state.execution_log[0]
        assert entry["status"] == "failed"
        assert "error" in entry


# ── 10. Missing model artifact produces explicit error ────────────────────────

class TestMissingArtifactError:

    def test_fault_agent_reports_error_for_missing_rf(self, tmp_path):
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "MISSING.joblib",
            rf_scaler_path=tmp_path / "MISSING.pkl",
        )
        state = agent.run(AgentState(test_id=1))
        assert state.errors, "Expected an error for missing artifact"
        assert "not found" in state.errors[0].lower() or "missing" in state.errors[0].lower() or "FaultDiagnosisAgent" in state.errors[0]

    def test_anomaly_agent_reports_error_for_missing_if(self, tmp_path):
        from industrial_health.agents.anomaly_agent import AnomalyAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        agent = AnomalyAgent(
            features_dir=tmp_path,
            if_model_path=tmp_path / "MISSING.joblib",
            if_scaler_path=tmp_path / "MISSING.pkl",
        )
        state = agent.run(AgentState(test_id=1))
        assert state.errors, "Expected an error for missing artifact"


# ── 11. Agent failure propagates safely ──────────────────────────────────────

class TestFailurePropagation:

    def test_stop_on_failure_prevents_downstream_execution(self, tmp_path):
        """
        If DataHealthAgent fails and stop_on_failure=True, downstream agents
        must NOT be called.
        """
        from industrial_health.agents.orchestrator import OrchestratorAgent
        from industrial_health.agents.state import AgentState

        orch = OrchestratorAgent(
            models_dir=tmp_path,
            features_dir=tmp_path,  # no CSV here → DataHealthAgent fails
            scalers_dir=tmp_path,
            test_id=1,
            stop_on_failure=True,
        )

        # Inject mock downstream agents with side_effect tracking
        downstream_called = []

        def track_run(name):
            def run(state: AgentState) -> AgentState:
                downstream_called.append(name)
                return state
            m = MagicMock()
            m.run.side_effect = run
            return m

        orch.fault_diagnosis_agent = track_run("FaultDiagnosisAgent")
        orch.anomaly_agent         = track_run("AnomalyAgent")

        state = orch.run()

        assert state.status == "FAILED"
        assert "FaultDiagnosisAgent" not in downstream_called
        assert "AnomalyAgent"         not in downstream_called

    def test_errors_list_populated_on_failure(self, tmp_path):
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "no.joblib",
            rf_scaler_path=tmp_path / "no.pkl",
        )
        state = agent.run(AgentState(test_id=1))
        assert len(state.errors) >= 1
        assert "FaultDiagnosisAgent" in state.errors[0]


# ── 12. No agent silently retrains models ─────────────────────────────────────

class TestNoSilentRetraining:

    def test_fault_agent_does_not_call_fit(self, tmp_path):
        from industrial_health.agents.fault_diagnosis_agent import FaultDiagnosisAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = 50

        agent = FaultDiagnosisAgent(
            features_dir=tmp_path,
            rf_model_path=tmp_path / "rf.joblib",
            rf_scaler_path=tmp_path / "sc.pkl",
        )
        mock_clf = MagicMock()
        mock_clf.feature_names = feat_cols
        mock_clf.model.predict_proba.return_value = np.column_stack(
            [np.full(n, 0.8), np.full(n, 0.2)]
        )
        mock_clf.model.predict.return_value = np.zeros(n, dtype=int)
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._clf    = mock_clf
        agent._scaler = mock_scaler

        state = agent.run(AgentState(test_id=1))

        # verify fit() was NEVER called on the model
        mock_clf.model.fit.assert_not_called()
        mock_clf.train.assert_not_called()

    def test_anomaly_agent_does_not_call_fit(self, tmp_path):
        from industrial_health.agents.anomaly_agent import AnomalyAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=50)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        feat_cols = [c for c in df.columns if c not in {
            "test_id","source_file","timestamp","n_channels",
            "snapshot_index","label","label_name","label_method","fault_start_idx"
        }]
        n = 50

        agent = AnomalyAgent(
            features_dir=tmp_path,
            if_model_path=tmp_path / "if.joblib",
            if_scaler_path=tmp_path / "if_sc.pkl",
        )
        mock_det = MagicMock()
        mock_det.feature_names = feat_cols
        mock_det.decision_scores.return_value = np.full(n, 0.1)
        mock_det.predict.return_value = np.zeros(n, dtype=int)
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((n, len(feat_cols)))

        agent._detector = mock_det
        agent._scaler   = mock_scaler

        state = agent.run(AgentState(test_id=1))

        # fit() must never be called
        mock_det.fit.assert_not_called()
        mock_det.model.fit.assert_not_called()


# ── 13. Phase 5 semantics: prob encoding ──────────────────────────────────────
# (Covered in TestFaultDiagnosisAgent.test_phase5_prob_encoding above)

# ── 14. Repeated execution is deterministic ───────────────────────────────────

class TestDeterminism:

    def test_two_runs_same_inputs_same_log_statuses(self, tmp_path):
        from industrial_health.agents.data_health_agent import DataHealthAgent
        from industrial_health.agents.state import AgentState

        df = _make_feature_df(n=80)
        csv = tmp_path / "test1_features.csv"
        df.to_csv(csv, index=False)

        result_df = pd.DataFrame({
            "snapshot_index":          np.arange(80),
            "health_score":            np.linspace(85, 50, 80),
            "health_status":           ["Stable"] * 60 + ["Degrading"] * 20,
            "health_rms_component":    np.full(80, 80.0),
            "health_anomaly_component":np.full(80, 70.0),
            "health_fault_component":  np.full(80, 75.0),
            "anomaly_score":           np.full(80, 0.1),
            "anomaly_pred":            np.zeros(80, dtype=int),
            "rf_pred":                 np.zeros(80, dtype=int),
            "prob_normal":             np.full(80, 0.8),
            "prob_faulty":             np.full(80, 0.2),
            "composite_rms":           np.full(80, 0.25),
            "label":                   np.zeros(80, dtype=int),
            "explanation":             ["OK"] * 80,
        })

        for run_i in range(2):
            agent = DataHealthAgent(
                features_dir=tmp_path,
                rf_model_path=tmp_path / "rf.joblib",
                rf_scaler_path=tmp_path / "rf_sc.pkl",
                if_model_path=tmp_path / "if.joblib",
                if_scaler_path=tmp_path / "if_sc.pkl",
            )
            mock_monitor = MagicMock()
            mock_monitor.score_dataset.return_value = result_df.copy()
            mock_monitor.cfg.trend_window = 20
            agent._monitor = mock_monitor

            state = agent.run(AgentState(test_id=1))
            if run_i == 0:
                first_status = state.final_health_status
                first_score  = state.final_health_score
            else:
                assert state.final_health_status == first_status
                assert state.final_health_score  == pytest.approx(first_score, abs=0.01)
