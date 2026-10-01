"""
tests/test_api.py — Phase 11: FastAPI REST API Tests

Coverage:
  1.  TestHealthEndpoint           — GET /health
  2.  TestAvailableTests           — GET /api/v1/pipeline/available-tests
  3.  TestPipelineRunSuccess        — POST /api/v1/pipeline/run (mock, success)
  4.  TestPipelineRunInvalidId      — test_id=99 → 422 (Pydantic ge/le)
  5.  TestPipelineRunMissingArtifact— no models → 404
  6.  TestPipelineRunServiceFailure — service returns error → 500
  7.  TestPipelineSummaryHit        — GET /summary uses cache → 200
  8.  TestPipelineSummaryMiss       — GET /summary no cache → 409
  9.  TestTimeseriesEndpoint        — GET /timeseries → correct schema
  10. TestFaultEndpoint             — GET /fault → FaultSummary
  11. TestAnomalyEndpoint           — GET /anomaly → AnomalySummary
  12. TestRAGQuerySuccess            — POST /rag/query → RAGQueryResponse
  13. TestRAGQueryValidation         — empty string, top_k=0 → 422
  14. TestRAGOllamaDown             — Ollama error → 503

RULES:
  - No real models loaded.
  - No ChromaDB accessed.
  - No Ollama required.
  - No network access required.
  - No running uvicorn server required.
  - All service calls are mocked via app.dependency_overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import get_pipeline_service


# ── Synthetic data ────────────────────────────────────────────────────────────

def _mock_pipeline_result(test_id: int = 1) -> dict:
    """Full build_final_result() structure."""
    return {
        "test_id": test_id,
        "health": {
            "mean_health_score":   72.5,
            "final_health_score":  68.1,
            "final_health_status": "Degrading",
            "health_trend":        "Degrading",
            "status_counts":       {"Stable": 1500, "Degrading": 500, "Critical": 156},
        },
        "fault": {
            "final_rf_pred":      1,
            "predicted_fault":    "Faulty",
            "normal_probability": 0.18,
            "fault_probability":  0.82,
            "fault_confidence":   0.82,
        },
        "anomaly": {
            "anomaly_score":       0.45,
            "anomaly_prediction":  1,
            "anomaly_label":       "Anomaly",
            "n_anomaly_predicted": 312,
        },
        "execution": {
            "agents_executed":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
            "execution_order":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
            "execution_status": "COMPLETED",
            "errors":           [],
            "execution_log": [
                {"agent": "DataHealthAgent",    "status": "completed", "duration_s": 5.2},
                {"agent": "FaultDiagnosisAgent","status": "completed", "duration_s": 0.4},
                {"agent": "AnomalyAgent",       "status": "completed", "duration_s": 0.5},
            ],
        },
    }


def _mock_timeseries(n: int = 5) -> dict:
    """Minimal timeseries payload from get_health_timeseries()."""
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    records = [
        {
            "snapshot_index":            i,
            "timestamp":                 f"2003-11-01 00:{i:02d}:00",
            "label":                     0,
            "label_name":                "Normal",
            "composite_rms":             round(float(rng.uniform(0.05, 0.15)), 6),
            "anomaly_score":             round(float(rng.uniform(0.05, 0.5)), 6),
            "anomaly_pred":              0,
            "rf_pred":                   0,
            "prob_normal":               0.85,
            "prob_faulty":               0.15,
            "health_rms_component":      85.0,
            "health_anomaly_component":  80.0,
            "health_fault_component":    90.0,
            "health_score":              84.0,
            "health_status":             "Stable",
            "explanation":               f"Snapshot {i}: Normal operation.",
        }
        for i in range(n)
    ]
    return {"records": records, "fault_start": 4, "n_snapshots": n}


def _mock_rag_results() -> dict:
    return {
        "results": [
            {
                "content":     "Outer race faults produce periodic impulses at BPFO.",
                "source":      "bearing_faults.md",
                "title":       "Bearing Fault Types",
                "section":     "Outer Race Fault",
                "chunk_index": 0,
                "distance":    0.15,
            },
        ],
        "count": 1,
    }


# ── Mock service factory ──────────────────────────────────────────────────────

def _make_mock_service(
    artifact_ok:       bool  = True,
    available_tests:   list  = None,
    pipeline_result:   dict  = None,
    cached_result:     dict  = None,
    timeseries_result: dict  = None,
    rag_result:        dict  = None,
) -> MagicMock:
    """Build a mock that matches the pipeline_service module interface."""
    svc = MagicMock()

    svc.artifact_status.return_value = {
        "test1_models":   artifact_ok,
        "knowledge_base": artifact_ok,
        "chroma_db":      artifact_ok,
    }
    svc.get_available_tests.return_value = (
        available_tests if available_tests is not None else ([1] if artifact_ok else [])
    )
    svc._model_artifacts_exist.return_value = artifact_ok
    svc.run_pipeline.return_value  = pipeline_result or _mock_pipeline_result()
    svc.get_cached_pipeline.return_value = cached_result
    svc.get_health_timeseries.return_value = timeseries_result or _mock_timeseries()
    svc.retrieve_knowledge.return_value = rag_result or _mock_rag_results()

    return svc


def _client(svc: MagicMock) -> TestClient:
    """Create a TestClient with the mock service injected."""
    app = create_app()
    app.dependency_overrides[get_pipeline_service] = lambda: svc
    return TestClient(app, raise_server_exceptions=False)


# ════════════════════════════════════════════════════════════════════════════
# 1. TestHealthEndpoint
# ════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self):
        svc = _make_mock_service(artifact_ok=True)
        response = _client(svc).get("/health")
        assert response.status_code == 200

    def test_health_schema_fields(self):
        svc = _make_mock_service(artifact_ok=True)
        data = _client(svc).get("/health").json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "artifacts" in data

    def test_health_status_ok_when_all_artifacts_present(self):
        svc = _make_mock_service(artifact_ok=True)
        data = _client(svc).get("/health").json()
        assert data["status"] == "ok"

    def test_health_status_degraded_when_artifact_missing(self):
        svc = _make_mock_service(artifact_ok=False)
        data = _client(svc).get("/health").json()
        assert data["status"] == "degraded"

    def test_health_artifacts_structure(self):
        svc = _make_mock_service(artifact_ok=True)
        data = _client(svc).get("/health").json()
        arts = data["artifacts"]
        assert "test1_models" in arts
        assert "knowledge_base" in arts
        assert "chroma_db" in arts


# ════════════════════════════════════════════════════════════════════════════
# 2. TestAvailableTests
# ════════════════════════════════════════════════════════════════════════════

class TestAvailableTests:

    def test_returns_200(self):
        svc = _make_mock_service()
        assert _client(svc).get("/api/v1/pipeline/available-tests").status_code == 200

    def test_returns_list_of_ints(self):
        svc = _make_mock_service()
        data = _client(svc).get("/api/v1/pipeline/available-tests").json()
        assert "available_test_ids" in data
        assert isinstance(data["available_test_ids"], list)

    def test_returns_test1_when_models_present(self):
        svc = _make_mock_service(artifact_ok=True)
        data = _client(svc).get("/api/v1/pipeline/available-tests").json()
        assert 1 in data["available_test_ids"]

    def test_returns_empty_when_no_models(self):
        svc = _make_mock_service(artifact_ok=False, available_tests=[])
        data = _client(svc).get("/api/v1/pipeline/available-tests").json()
        assert data["available_test_ids"] == []


# ════════════════════════════════════════════════════════════════════════════
# 3. TestPipelineRunSuccess
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineRunSuccess:

    def test_returns_200(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1})
        assert r.status_code == 200

    def test_response_has_health_fault_anomaly_execution(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()
        assert "health" in data
        assert "fault" in data
        assert "anomaly" in data
        assert "execution" in data

    def test_health_sub_schema_fields(self):
        svc = _make_mock_service()
        h = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()["health"]
        assert "final_health_score" in h
        assert "final_health_status" in h
        assert "health_trend" in h
        assert "status_counts" in h

    def test_fault_disclaimer_present(self):
        svc = _make_mock_service()
        f = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()["fault"]
        assert "label_disclaimer" in f
        assert "heuristic" in f["label_disclaimer"].lower()

    def test_anomaly_semantics_present(self):
        svc = _make_mock_service()
        a = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()["anomaly"]
        assert "score_semantics" in a
        assert "HIGHER" in a["score_semantics"]

    def test_test_id_echoed(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()
        assert data["test_id"] == 1


# ════════════════════════════════════════════════════════════════════════════
# 4. TestPipelineRunInvalidId
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineRunInvalidId:

    def test_test_id_0_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 0})
        assert r.status_code == 422

    def test_test_id_4_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 4})
        assert r.status_code == 422

    def test_test_id_string_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": "abc"})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# 5. TestPipelineRunMissingArtifact
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineRunMissingArtifact:

    def test_missing_models_returns_404(self):
        svc = _make_mock_service(artifact_ok=False)
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1})
        assert r.status_code == 404

    def test_error_response_no_internal_paths(self):
        svc = _make_mock_service(artifact_ok=False)
        data = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()
        body = str(data)
        assert "C:\\" not in body
        assert "venvagentic" not in body
        assert "Traceback" not in body


# ════════════════════════════════════════════════════════════════════════════
# 6. TestPipelineRunServiceFailure
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineRunServiceFailure:

    def test_service_error_returns_500(self):
        svc = _make_mock_service(
            artifact_ok=True,
            pipeline_result={"error": "Unexpected RuntimeError", "details": []},
        )
        r = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1})
        assert r.status_code == 500

    def test_error_response_does_not_expose_traceback(self):
        svc = _make_mock_service(
            artifact_ok=True,
            pipeline_result={"error": "Some error", "details": ["internal detail"]},
        )
        data = _client(svc).post("/api/v1/pipeline/run", json={"test_id": 1}).json()
        assert "Traceback" not in str(data)


# ════════════════════════════════════════════════════════════════════════════
# 7. TestPipelineSummaryHit
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineSummaryHit:

    def test_returns_200_when_cached(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        r = _client(svc).get("/api/v1/pipeline/1/summary")
        assert r.status_code == 200

    def test_summary_has_correct_schema(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        data = _client(svc).get("/api/v1/pipeline/1/summary").json()
        assert "health" in data
        assert "fault" in data
        assert "anomaly" in data


# ════════════════════════════════════════════════════════════════════════════
# 8. TestPipelineSummaryMiss
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineSummaryMiss:

    def test_returns_409_when_not_run(self):
        svc = _make_mock_service(artifact_ok=True, cached_result=None)
        r = _client(svc).get("/api/v1/pipeline/1/summary")
        assert r.status_code == 409

    def test_409_body_mentions_run(self):
        svc = _make_mock_service(artifact_ok=True, cached_result=None)
        body = str(_client(svc).get("/api/v1/pipeline/1/summary").json())
        assert "run" in body.lower() or "POST" in body


# ════════════════════════════════════════════════════════════════════════════
# 9. TestTimeseriesEndpoint
# ════════════════════════════════════════════════════════════════════════════

class TestTimeseriesEndpoint:

    def test_returns_200(self):
        svc = _make_mock_service()
        r = _client(svc).get("/api/v1/pipeline/1/timeseries")
        assert r.status_code == 200

    def test_response_schema_fields(self):
        svc = _make_mock_service()
        data = _client(svc).get("/api/v1/pipeline/1/timeseries").json()
        assert "test_id" in data
        assert "n_snapshots" in data
        assert "fault_start" in data
        assert "snapshots" in data
        assert "disclaimer" in data

    def test_snapshots_are_list(self):
        svc = _make_mock_service()
        data = _client(svc).get("/api/v1/pipeline/1/timeseries").json()
        assert isinstance(data["snapshots"], list)

    def test_snapshot_has_required_fields(self):
        svc = _make_mock_service()
        data = _client(svc).get("/api/v1/pipeline/1/timeseries").json()
        snap = data["snapshots"][0]
        for field in ["snapshot_index", "health_score", "health_status", "anomaly_score"]:
            assert field in snap, f"Missing field: {field}"

    def test_no_models_returns_404(self):
        svc = _make_mock_service(artifact_ok=False)
        r = _client(svc).get("/api/v1/pipeline/1/timeseries")
        assert r.status_code == 404

    def test_timeseries_error_returns_error_code(self):
        svc = _make_mock_service(timeseries_result={"error": "Feature CSV not found.", "details": []})
        r = _client(svc).get("/api/v1/pipeline/1/timeseries")
        assert r.status_code in (404, 500)


# ════════════════════════════════════════════════════════════════════════════
# 10. TestFaultEndpoint
# ════════════════════════════════════════════════════════════════════════════

class TestFaultEndpoint:

    def test_returns_200_with_cache(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        r = _client(svc).get("/api/v1/pipeline/1/fault")
        assert r.status_code == 200

    def test_fault_schema_fields(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        data = _client(svc).get("/api/v1/pipeline/1/fault").json()
        assert "predicted_fault" in data
        assert "fault_probability" in data
        assert "label_disclaimer" in data

    def test_returns_409_no_cache(self):
        svc = _make_mock_service(artifact_ok=True, cached_result=None)
        r = _client(svc).get("/api/v1/pipeline/1/fault")
        assert r.status_code == 409


# ════════════════════════════════════════════════════════════════════════════
# 11. TestAnomalyEndpoint
# ════════════════════════════════════════════════════════════════════════════

class TestAnomalyEndpoint:

    def test_returns_200_with_cache(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        r = _client(svc).get("/api/v1/pipeline/1/anomaly")
        assert r.status_code == 200

    def test_anomaly_schema_fields(self):
        svc = _make_mock_service(cached_result=_mock_pipeline_result())
        data = _client(svc).get("/api/v1/pipeline/1/anomaly").json()
        assert "anomaly_score" in data
        assert "anomaly_label" in data
        assert "score_semantics" in data
        assert "HIGHER" in data["score_semantics"]

    def test_returns_409_no_cache(self):
        svc = _make_mock_service(artifact_ok=True, cached_result=None)
        r = _client(svc).get("/api/v1/pipeline/1/anomaly")
        assert r.status_code == 409


# ════════════════════════════════════════════════════════════════════════════
# 12. TestRAGQuerySuccess
# ════════════════════════════════════════════════════════════════════════════

class TestRAGQuerySuccess:

    def test_returns_200(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query",
                               json={"query": "outer race fault", "top_k": 3})
        assert r.status_code == 200

    def test_response_schema_fields(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/rag/query",
                                  json={"query": "outer race fault", "top_k": 3}).json()
        assert "query" in data
        assert "count" in data
        assert "results" in data
        assert "disclaimer" in data
        assert "embedding_model" in data

    def test_result_item_fields(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/rag/query",
                                  json={"query": "bearing maintenance"}).json()
        result = data["results"][0]
        assert "content" in result
        assert "source" in result
        assert "distance" in result
        assert "relevance_pct" in result

    def test_disclaimer_says_retrieval_only(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/rag/query",
                                  json={"query": "kurtosis"}).json()
        assert "retrieval" in data["disclaimer"].lower()

    def test_relevance_pct_in_range(self):
        svc = _make_mock_service()
        data = _client(svc).post("/api/v1/rag/query",
                                  json={"query": "bearing fault"}).json()
        for r in data["results"]:
            assert 0.0 <= r["relevance_pct"] <= 100.0

    def test_default_top_k_accepted(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query", json={"query": "vibration"})
        assert r.status_code == 200


# ════════════════════════════════════════════════════════════════════════════
# 13. TestRAGQueryValidation
# ════════════════════════════════════════════════════════════════════════════

class TestRAGQueryValidation:

    def test_empty_string_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query", json={"query": "", "top_k": 5})
        assert r.status_code == 422

    def test_top_k_zero_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query",
                               json={"query": "fault", "top_k": 0})
        assert r.status_code == 422

    def test_top_k_11_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query",
                               json={"query": "fault", "top_k": 11})
        assert r.status_code == 422

    def test_missing_query_returns_422(self):
        svc = _make_mock_service()
        r = _client(svc).post("/api/v1/rag/query", json={"top_k": 5})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# 14. TestRAGOllamaDown
# ════════════════════════════════════════════════════════════════════════════

class TestRAGOllamaDown:

    def test_ollama_error_returns_503(self):
        svc = _make_mock_service(
            rag_result={
                "error":           "Embedding service unavailable. Ollama is not running.",
                "details":         ["ollama serve"],
                "is_ollama_error": True,
            }
        )
        r = _client(svc).post("/api/v1/rag/query",
                               json={"query": "outer race fault", "top_k": 3})
        assert r.status_code == 503

    def test_503_body_is_sanitized(self):
        svc = _make_mock_service(
            rag_result={
                "error":           "Embedding service unavailable.",
                "details":         [],
                "is_ollama_error": True,
            }
        )
        body = str(_client(svc).post("/api/v1/rag/query",
                                     json={"query": "kurtosis"}).json())
        assert "Traceback" not in body
        assert "C:\\" not in body

    def test_non_ollama_error_returns_500(self):
        svc = _make_mock_service(
            rag_result={
                "error":           "Unexpected ChromaDB error.",
                "details":         [],
                "is_ollama_error": False,
            }
        )
        r = _client(svc).post("/api/v1/rag/query",
                               json={"query": "vibration analysis", "top_k": 3})
        assert r.status_code == 500
