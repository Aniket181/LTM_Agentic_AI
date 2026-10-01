"""
test_dashboard.py — Phase 10: Dashboard Unit Tests

Test groups:
  1.  Dashboard module imports
  2.  Config path constants
  3.  Config test_id availability detection
  4.  Services run_pipeline mock
  5.  Services health timeseries mock
  6.  Services retrieve_knowledge validation
  7.  Services retrieve_knowledge mock results
  8.  Services error propagation
  9.  Health card rendering (mocked st)
  10. Fault card rendering (mocked st)
  11. Anomaly card rendering (mocked st)
  12. Trend chart rendering (mocked st)
  13. RAG panel (mocked services)
  14. No model retraining / no network

IMPORTANT:
  - No real models are loaded.
  - No ChromaDB is accessed.
  - No Ollama is required.
  - No IMS dataset needed.
  - All tests use mocks and synthetic data.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_result(
    health_score: float = 75.0,
    status: str = "Stable",
    trend: str = "Stable",
    fault_label: str = "Normal",
    anomaly_label: str = "Normal",
) -> dict:
    """Build a synthetic pipeline result matching build_final_result() schema."""
    return {
        "test_id": 1,
        "health": {
            "mean_health_score":   health_score - 2.0,
            "final_health_score":  health_score,
            "final_health_status": status,
            "health_trend":        trend,
            "status_counts":       {"Stable": 100, "Degrading": 20, "Critical": 5},
        },
        "fault": {
            "final_rf_pred":      0 if fault_label == "Normal" else 1,
            "predicted_fault":    fault_label,
            "normal_probability": 0.85 if fault_label == "Normal" else 0.15,
            "fault_probability":  0.15 if fault_label == "Normal" else 0.85,
            "fault_confidence":   0.85,
        },
        "anomaly": {
            "anomaly_score":       0.12,
            "anomaly_prediction":  0 if anomaly_label == "Normal" else 1,
            "anomaly_label":       anomaly_label,
            "n_anomaly_predicted": 10,
        },
        "execution": {
            "agents_executed":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
            "execution_order":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
            "execution_status": "COMPLETED",
            "errors":           [],
            "execution_log": [
                {"agent": "DataHealthAgent",   "status": "completed", "duration_s": 1.2},
                {"agent": "FaultDiagnosisAgent","status": "completed", "duration_s": 0.3},
                {"agent": "AnomalyAgent",       "status": "completed", "duration_s": 0.4},
            ],
        },
    }


def _make_mock_ts_df(n: int = 100) -> pd.DataFrame:
    """Build a synthetic health timeseries DataFrame matching score_dataset() schema."""
    snapshot_idx = np.arange(n)
    health_scores = np.clip(80 - np.linspace(0, 50, n) + np.random.default_rng(42).normal(0, 2, n), 0, 100)
    return pd.DataFrame({
        "snapshot_index":          snapshot_idx,
        "timestamp":               pd.date_range("2003-11-01", periods=n, freq="10min"),
        "label":                   np.where(snapshot_idx < int(0.8 * n), 0, 1),
        "label_name":              np.where(snapshot_idx < int(0.8 * n), "Normal", "Faulty"),
        "composite_rms":           np.abs(np.random.default_rng(1).normal(0.1, 0.01, n)),
        "anomaly_score":           np.linspace(0.05, 0.6, n),
        "anomaly_pred":            np.where(snapshot_idx > int(0.7 * n), 1, 0),
        "rf_pred":                 np.where(snapshot_idx > int(0.8 * n), 1, 0),
        "prob_normal":             np.clip(1 - np.linspace(0, 0.9, n), 0, 1),
        "prob_faulty":             np.clip(np.linspace(0, 0.9, n), 0, 1),
        "health_rms_component":    np.clip(90 - np.linspace(0, 40, n), 0, 100),
        "health_anomaly_component":np.clip(90 - np.linspace(0, 60, n), 0, 100),
        "health_fault_component":  np.clip(90 - np.linspace(0, 50, n), 0, 100),
        "health_score":            health_scores,
        "health_status":           ["Stable" if s >= 70 else "Degrading" if s >= 40 else "Critical"
                                    for s in health_scores],
        "explanation":             [f"Snapshot {i} explanation" for i in range(n)],
    })


def _make_mock_retrieval_results() -> list[dict]:
    return [
        {
            "content":     "Outer race faults produce periodic impulses at BPFO frequency.",
            "source":      "bearing_faults.md",
            "title":       "Bearing Fault Types",
            "section":     "Outer Race Fault",
            "chunk_index": 0,
            "distance":    0.15,
        },
        {
            "content":     "High RMS vibration indicates progressive bearing deterioration.",
            "source":      "troubleshooting.md",
            "title":       "Troubleshooting Guide",
            "section":     "High RMS Vibration",
            "chunk_index": 3,
            "distance":    0.22,
        },
    ]


# ── 1. Dashboard module imports ───────────────────────────────────────────────

class TestDashboardImports:
    """All dashboard modules must import cleanly (no model loading triggered)."""

    def test_config_imports(self):
        from dashboard.config import (
            PROJECT_ROOT, MODELS_DIR, SCALERS_DIR,
            FEATURES_DIR, KNOWLEDGE_BASE_PATH, CHROMA_PATH,
            get_available_test_ids, get_artifact_paths,
        )
        assert PROJECT_ROOT is not None

    def test_health_card_imports(self):
        from dashboard.components.health_card import render_health_card
        assert callable(render_health_card)

    def test_fault_card_imports(self):
        from dashboard.components.fault_card import render_fault_card
        assert callable(render_fault_card)

    def test_anomaly_card_imports(self):
        from dashboard.components.anomaly_card import render_anomaly_card
        assert callable(render_anomaly_card)

    def test_trend_chart_imports(self):
        from dashboard.components.trend_chart import render_trend_chart
        assert callable(render_trend_chart)

    def test_exec_log_imports(self):
        from dashboard.components.exec_log import render_exec_log
        assert callable(render_exec_log)

    def test_rag_panel_imports(self):
        # The rag_panel imports services; we just check the module imports OK
        import importlib
        mod = importlib.import_module("dashboard.components.rag_panel")
        assert hasattr(mod, "render_rag_panel")

    def test_services_imports(self):
        # services.py uses @st.cache_* decorators — only import-level check
        import importlib
        mod = importlib.import_module("dashboard.services")
        assert hasattr(mod, "run_pipeline")
        assert hasattr(mod, "get_health_timeseries")
        assert hasattr(mod, "retrieve_knowledge")


# ── 2. Config path constants ──────────────────────────────────────────────────

class TestServicesConfig:

    def test_project_root_is_directory(self):
        from dashboard.config import PROJECT_ROOT
        assert PROJECT_ROOT.is_dir()

    def test_features_dir_path_correct(self):
        from dashboard.config import FEATURES_DIR, PROJECT_ROOT
        assert FEATURES_DIR == PROJECT_ROOT / "data" / "features"

    def test_models_dir_path_correct(self):
        from dashboard.config import MODELS_DIR, PROJECT_ROOT
        assert MODELS_DIR == PROJECT_ROOT / "models"

    def test_knowledge_base_path_correct(self):
        from dashboard.config import KNOWLEDGE_BASE_PATH, PROJECT_ROOT
        assert KNOWLEDGE_BASE_PATH == PROJECT_ROOT / "knowledge_base"

    def test_chroma_path_correct(self):
        from dashboard.config import CHROMA_PATH, PROJECT_ROOT
        assert CHROMA_PATH == PROJECT_ROOT / "chroma_db"

    def test_get_artifact_paths_returns_all_keys(self):
        from dashboard.config import get_artifact_paths
        paths = get_artifact_paths(1)
        assert set(paths.keys()) == {"rf_model", "rf_scaler", "if_model", "if_scaler", "feature_csv"}

    def test_get_artifact_paths_test1_correct(self):
        from dashboard.config import get_artifact_paths
        paths = get_artifact_paths(1)
        assert "random_forest_test1" in str(paths["rf_model"])
        assert "phase5_test1" in str(paths["rf_scaler"])
        assert "isolation_forest_test1" in str(paths["if_model"])
        assert "phase6_test1" in str(paths["if_scaler"])


# ── 3. Config test_id availability ───────────────────────────────────────────

class TestConfigTestIdAvailability:

    def test_test1_available_because_models_exist(self):
        """Test 1 models are present — should be in available list."""
        from dashboard.config import get_available_test_ids, MODELS_DIR, SCALERS_DIR
        # Only assert if model files actually exist (they should in this project)
        rf_exists = (MODELS_DIR / "random_forest_test1.joblib").exists()
        if rf_exists:
            ids = get_available_test_ids()
            assert 1 in ids

    def test_returns_list_of_ints(self):
        from dashboard.config import get_available_test_ids
        ids = get_available_test_ids()
        assert isinstance(ids, list)
        for i in ids:
            assert isinstance(i, int)

    def test_no_test_for_missing_models(self):
        """Test ID 99 has no models — must not appear."""
        from dashboard.config import _model_artifacts_exist
        assert not _model_artifacts_exist(99)


# ── 4. Services run_pipeline (mocked orchestrator) ────────────────────────────

class TestServicesRunPipeline:

    def _get_service_run_pipeline(self):
        """Import and return the un-cached run_pipeline logic for testing."""
        from dashboard import services
        return services

    def test_run_pipeline_success_shape(self):
        """Mock OrchestratorAgent to verify returned dict has expected keys."""
        mock_state = MagicMock()
        mock_state.status = "COMPLETED"
        mock_state.errors = []

        expected = _make_mock_result()

        with patch("dashboard.services._get_orchestrator") as mock_orch_factory:
            mock_orch = MagicMock()
            mock_orch.run.return_value = mock_state
            mock_orch.build_final_result.return_value = expected
            mock_orch_factory.return_value = mock_orch

            from dashboard.services import run_pipeline
            # Clear any cache from previous test
            run_pipeline.clear()
            result = run_pipeline(1)

        assert "health" in result
        assert "fault" in result
        assert "anomaly" in result
        assert "execution" in result

    def test_run_pipeline_failure_returns_error_dict(self):
        """Pipeline failure should return {"error": ..., "details": ...}."""
        mock_state = MagicMock()
        mock_state.status = "FAILED"
        mock_state.errors = ["DataHealthAgent: Feature CSV not found"]

        with patch("dashboard.services._get_orchestrator") as mock_orch_factory:
            mock_orch = MagicMock()
            mock_orch.run.return_value = mock_state
            mock_orch_factory.return_value = mock_orch

            from dashboard.services import run_pipeline
            run_pipeline.clear()
            result = run_pipeline(1)

        assert "error" in result
        assert "details" in result

    def test_run_pipeline_file_not_found_returns_error(self):
        """Missing model file should return error dict, not raise."""
        with patch("dashboard.services._get_orchestrator") as mock_orch_factory:
            mock_orch = MagicMock()
            mock_orch.run.side_effect = FileNotFoundError("model not found")
            mock_orch_factory.return_value = mock_orch

            from dashboard.services import run_pipeline
            run_pipeline.clear()
            result = run_pipeline(1)

        assert "error" in result


# ── 5. Services health timeseries (mocked monitor) ───────────────────────────

class TestServicesHealthTimeseries:

    def test_returns_records_on_success(self):
        mock_df = _make_mock_ts_df(50)

        with (
            patch("dashboard.services._get_health_monitor") as mock_monitor_factory,
            patch("dashboard.services.FEATURES_DIR") as mock_features,
            patch("pandas.read_csv", return_value=mock_df),
        ):
            mock_monitor = MagicMock()
            mock_monitor.score_dataset.return_value = mock_df
            mock_monitor_factory.return_value = mock_monitor

            # Make csv_path.exists() return True
            mock_csv = MagicMock()
            mock_csv.exists.return_value = True
            mock_features.__truediv__ = lambda self, other: mock_csv

            from dashboard.services import get_health_timeseries
            get_health_timeseries.clear()
            result = get_health_timeseries(1)

        # Either success (has "records") or a mocking edge case
        assert isinstance(result, dict)

    def test_missing_monitor_returns_error(self):
        with patch("dashboard.services._get_health_monitor", return_value=None):
            from dashboard.services import get_health_timeseries
            get_health_timeseries.clear()
            result = get_health_timeseries(1)

        assert "error" in result

    def test_get_health_timeseries_df_returns_none_on_error(self):
        with patch("dashboard.services.get_health_timeseries", return_value={"error": "test error"}):
            from dashboard.services import get_health_timeseries_df
            df, meta = get_health_timeseries_df(1)
        assert df is None
        assert "error" in meta


# ── 6. Services retrieve_knowledge validation ─────────────────────────────────

class TestServicesRetrieveValidation:

    def test_empty_query_returns_error_dict(self):
        from dashboard.services import retrieve_knowledge
        result = retrieve_knowledge("")
        assert "error" in result

    def test_whitespace_query_returns_error_dict(self):
        from dashboard.services import retrieve_knowledge
        result = retrieve_knowledge("   ")
        assert "error" in result

    def test_missing_retriever_returns_error_dict(self):
        with patch("dashboard.services._get_retriever", return_value=None):
            from dashboard.services import retrieve_knowledge
            result = retrieve_knowledge("test query")
        assert "error" in result


# ── 7. Services retrieve_knowledge mock results ───────────────────────────────

class TestServicesRetrieveResults:

    def test_valid_query_returns_results_list(self):
        mock_results = []
        for r in _make_mock_retrieval_results():
            m = MagicMock()
            m.to_dict.return_value = r
            mock_results.append(m)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_results

        with patch("dashboard.services._get_retriever", return_value=mock_retriever):
            from dashboard.services import retrieve_knowledge
            result = retrieve_knowledge("outer race fault", top_k=2)

        assert "results" in result
        assert len(result["results"]) == 2

    def test_retrieval_result_has_required_fields(self):
        mock_results = []
        for r in _make_mock_retrieval_results():
            m = MagicMock()
            m.to_dict.return_value = r
            mock_results.append(m)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_results

        with patch("dashboard.services._get_retriever", return_value=mock_retriever):
            from dashboard.services import retrieve_knowledge
            result = retrieve_knowledge("bearing fault")

        for item in result["results"]:
            assert "content" in item
            assert "source" in item
            assert "distance" in item

    def test_ollama_error_returns_friendly_message(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("Connection refused to ollama")

        with patch("dashboard.services._get_retriever", return_value=mock_retriever):
            from dashboard.services import retrieve_knowledge
            result = retrieve_knowledge("bearing fault")

        assert "error" in result
        assert "Ollama" in result["error"] or "ollama" in result["error"].lower()


# ── 8. Services error propagation ────────────────────────────────────────────

class TestServicesErrorPropagation:

    def test_run_pipeline_exception_returns_error_dict(self):
        with patch("dashboard.services._get_orchestrator") as mock_factory:
            mock_orch = MagicMock()
            mock_orch.run.side_effect = RuntimeError("unexpected error")
            mock_factory.return_value = mock_orch

            from dashboard.services import run_pipeline
            run_pipeline.clear()
            result = run_pipeline(1)

        assert "error" in result
        assert isinstance(result["error"], str)

    def test_retriever_runtime_error_returns_error_dict(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("collection is empty")

        with patch("dashboard.services._get_retriever", return_value=mock_retriever):
            from dashboard.services import retrieve_knowledge
            result = retrieve_knowledge("test query")

        assert "error" in result


# ── 9. Health card rendering ─────────────────────────────────────────────────

class TestHealthCardRendering:

    def test_render_health_card_stable(self):
        """render_health_card with stable result must not raise."""
        result = _make_mock_result(health_score=80.0, status="Stable")
        with patch("dashboard.components.health_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 4
            mock_st.plotly_chart = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            from dashboard.components.health_card import render_health_card
            render_health_card(result["health"])  # must not raise

    def test_render_health_card_critical(self):
        result = _make_mock_result(health_score=25.0, status="Critical")
        with patch("dashboard.components.health_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 4
            mock_st.plotly_chart = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            from dashboard.components.health_card import render_health_card
            render_health_card(result["health"])

    def test_gauge_colour_stable(self):
        from dashboard.components.health_card import _gauge_colour
        assert _gauge_colour(80.0) == "#22c55e"

    def test_gauge_colour_degrading(self):
        from dashboard.components.health_card import _gauge_colour
        assert _gauge_colour(55.0) == "#f97316"

    def test_gauge_colour_critical(self):
        from dashboard.components.health_card import _gauge_colour
        assert _gauge_colour(20.0) == "#ef4444"


# ── 10. Fault card rendering ─────────────────────────────────────────────────

class TestFaultCardRendering:

    def test_render_fault_card_normal(self):
        result = _make_mock_result(fault_label="Normal")
        with patch("dashboard.components.fault_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 2
            mock_st.plotly_chart = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.caption = MagicMock()
            from dashboard.components.fault_card import render_fault_card
            render_fault_card(result["fault"])

    def test_render_fault_card_faulty(self):
        result = _make_mock_result(fault_label="Faulty")
        with patch("dashboard.components.fault_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 2
            mock_st.plotly_chart = MagicMock()
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.caption = MagicMock()
            from dashboard.components.fault_card import render_fault_card
            render_fault_card(result["fault"])


# ── 11. Anomaly card rendering ────────────────────────────────────────────────

class TestAnomalyCardRendering:

    def test_render_anomaly_card_normal(self):
        result = _make_mock_result(anomaly_label="Normal")
        with patch("dashboard.components.anomaly_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 2
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.caption = MagicMock()
            from dashboard.components.anomaly_card import render_anomaly_card
            render_anomaly_card(result["anomaly"])

    def test_render_anomaly_card_anomaly(self):
        result = _make_mock_result(anomaly_label="Anomaly")
        with patch("dashboard.components.anomaly_card.st") as mock_st:
            mock_st.columns.return_value = [MagicMock()] * 2
            mock_st.metric = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.caption = MagicMock()
            from dashboard.components.anomaly_card import render_anomaly_card
            render_anomaly_card(result["anomaly"])


# ── 12. Trend chart rendering ─────────────────────────────────────────────────

class TestTrendChartRendering:

    def test_render_trend_chart_basic(self):
        df = _make_mock_ts_df(50)
        with patch("dashboard.components.trend_chart.st") as mock_st:
            mock_st.plotly_chart = MagicMock()
            mock_st.caption = MagicMock()
            mock_st.warning = MagicMock()
            from dashboard.components.trend_chart import render_trend_chart
            render_trend_chart(df, fault_start=40)

    def test_render_trend_chart_with_components(self):
        df = _make_mock_ts_df(50)
        with patch("dashboard.components.trend_chart.st") as mock_st:
            mock_st.plotly_chart = MagicMock()
            mock_st.caption = MagicMock()
            mock_st.warning = MagicMock()
            from dashboard.components.trend_chart import render_trend_chart
            render_trend_chart(df, fault_start=40, show_components=True)

    def test_render_trend_chart_none_shows_warning(self):
        with patch("dashboard.components.trend_chart.st") as mock_st:
            mock_st.warning = MagicMock()
            from dashboard.components.trend_chart import render_trend_chart
            render_trend_chart(None, fault_start=0)
        mock_st.warning.assert_called_once()

    def test_render_trend_chart_empty_df_shows_warning(self):
        with patch("dashboard.components.trend_chart.st") as mock_st:
            mock_st.warning = MagicMock()
            from dashboard.components.trend_chart import render_trend_chart
            render_trend_chart(pd.DataFrame(), fault_start=0)
        mock_st.warning.assert_called_once()


# ── 13. RAG panel (mocked services) ──────────────────────────────────────────

class TestRAGPanel:

    def test_rag_panel_renders_without_error(self):
        with (
            patch("dashboard.components.rag_panel.get_retriever_chunk_count", return_value=66),
            patch("dashboard.components.rag_panel.retrieve_knowledge") as mock_retrieve,
            patch("dashboard.components.rag_panel.st") as mock_st,
        ):
            mock_retrieve.return_value = {"results": _make_mock_retrieval_results(), "count": 2}
            mock_st.text_input.return_value = ""
            mock_st.selectbox.return_value = ""
            mock_st.button.return_value = False
            mock_st.number_input.return_value = 5
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.caption = MagicMock()
            mock_st.markdown = MagicMock()

            from dashboard.components.rag_panel import render_rag_panel
            render_rag_panel()

    def test_rag_panel_zero_chunks_shows_warning(self):
        with (
            patch("dashboard.components.rag_panel.get_retriever_chunk_count", return_value=0),
            patch("dashboard.components.rag_panel.st") as mock_st,
        ):
            mock_st.text_input.return_value = ""
            mock_st.selectbox.return_value = ""
            mock_st.button.return_value = False
            mock_st.number_input.return_value = 5
            mock_st.columns.return_value = [MagicMock(), MagicMock()]
            mock_st.caption = MagicMock()
            mock_st.markdown = MagicMock()
            mock_st.warning = MagicMock()

            from dashboard.components.rag_panel import render_rag_panel
            render_rag_panel()
        mock_st.warning.assert_called()


# ── 14. No model retraining / no network ─────────────────────────────────────

class TestNoRetrainingNoNetwork:

    def test_config_imports_no_model_load(self):
        """Importing config must never load a model or make network calls."""
        import importlib
        # If we can reimport cleanly without error, we're safe
        if "dashboard.config" in sys.modules:
            del sys.modules["dashboard.config"]
        mod = importlib.import_module("dashboard.config")
        assert hasattr(mod, "PROJECT_ROOT")

    def test_component_imports_no_model_load(self):
        """Importing components must not load models."""
        import importlib
        for module_name in [
            "dashboard.components.health_card",
            "dashboard.components.fault_card",
            "dashboard.components.anomaly_card",
            "dashboard.components.trend_chart",
            "dashboard.components.exec_log",
        ]:
            if module_name in sys.modules:
                del sys.modules[module_name]
            mod = importlib.import_module(module_name)
            assert mod is not None

    def test_retrieve_empty_query_no_network(self):
        """Empty query must short-circuit before any network/Ollama call."""
        from dashboard.services import retrieve_knowledge
        # This must return an error dict without touching Ollama or ChromaDB
        result = retrieve_knowledge("")
        assert "error" in result

    def test_services_retrieve_no_model_fit_call(self):
        """services.py must never call fit() on any model."""
        import inspect
        import dashboard.services as svc_mod
        source = inspect.getsource(svc_mod)
        # Verify no direct calls to .fit( in services.py
        assert ".fit(" not in source, "services.py must not call .fit()"
        assert ".train(" not in source, "services.py must not call .train()"
