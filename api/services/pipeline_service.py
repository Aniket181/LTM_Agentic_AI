"""
api/services/pipeline_service.py — Phase 11: API Pipeline Service

PURPOSE:
  Singleton service layer that wraps the existing Phase 5–9 classes for use
  by FastAPI routes. This is a SEPARATE service from dashboard/services.py.

  dashboard/services.py uses @st.cache_resource / @st.cache_data — Streamlit
  runtime decorators that crash if imported outside Streamlit. This module
  provides an equivalent caching strategy using plain Python module-level
  dicts and lazy initialization.

DESIGN:
  - OrchestratorAgent, HealthMonitor, KnowledgeRetriever are created ONCE
    per test_id and held in module-level dicts for the lifetime of the API
    server process.
  - Pipeline results and health timeseries records are cached in memory keyed
    by test_id. Clients use POST /run to (re)compute; GET /summary reads cache.
  - No Streamlit imports anywhere in this file.
  - No ML logic is implemented here. All computation is in Phase 5–9 modules.

DO NOT:
  - Retrain any model.
  - Implement a health score, anomaly score, or RF/IF algorithm.
  - Import anything from dashboard.services (it requires Streamlit runtime).
  - Expose Python tracebacks or filesystem paths to callers.

THREAD SAFETY NOTE:
  For an M.Tech prototype with single-user API load, module-level dict access
  is sufficient. Model inference (sklearn predict) is GIL-bound and safe for
  sequential requests. Do not call this service from multiple async tasks
  concurrently without adding a lock around inference calls.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# dashboard/config.py is pure Python (no Streamlit) — safe to import here
from dashboard.config import (
    MODELS_DIR, SCALERS_DIR, FEATURES_DIR,
    KNOWLEDGE_BASE_PATH, CHROMA_PATH,
    get_artifact_paths, _model_artifacts_exist, get_available_test_ids,
)


# ── Module-level singletons ───────────────────────────────────────────────────

_orchestrators:   dict[int, Any] = {}   # test_id → OrchestratorAgent
_health_monitors: dict[int, Any] = {}   # test_id → HealthMonitor | None
_retriever:       Any            = None  # KnowledgeRetriever | None

# In-memory result caches — populated by run_pipeline()
_pipeline_cache:   dict[int, dict] = {}  # test_id → full pipeline result dict
_timeseries_cache: dict[int, dict] = {}  # test_id → {records, fault_start, n_snapshots}


# ── Artifact availability ─────────────────────────────────────────────────────

def artifact_status() -> dict[str, bool]:
    """
    Return a dict describing whether key artifacts exist on disk.

    Does NOT load models. Used by GET /health.
    """
    return {
        "test1_models":   _model_artifacts_exist(1),
        "knowledge_base": KNOWLEDGE_BASE_PATH.exists(),
        "chroma_db":      CHROMA_PATH.exists(),
    }


def get_available_tests() -> list[int]:
    """Return test IDs for which all model artifacts exist."""
    return get_available_test_ids()


# Public alias so routes can call service._model_artifacts_exist(test_id)
# (imported from dashboard.config at module level above)

# ── OrchestratorAgent ─────────────────────────────────────────────────────────

def _get_orchestrator(test_id: int) -> Any:
    """
    Lazy-initialize OrchestratorAgent for test_id and cache it.

    Raises FileNotFoundError if model artifacts are missing.
    """
    if test_id not in _orchestrators:
        from industrial_health.agents import OrchestratorAgent
        _orchestrators[test_id] = OrchestratorAgent(
            models_dir   = MODELS_DIR,
            features_dir = FEATURES_DIR,
            scalers_dir  = SCALERS_DIR,
            test_id      = test_id,
        )
    return _orchestrators[test_id]


def run_pipeline(test_id: int) -> dict[str, Any]:
    """
    Execute the Phase 8 agentic pipeline for test_id and cache the result.

    Returns a dict with keys: test_id, health, fault, anomaly, execution.
    This is the output of OrchestratorAgent.build_final_result().

    On success, the result is stored in _pipeline_cache[test_id].

    Returns {"error": str, "details": list[str]} on failure —
    callers convert this to an appropriate HTTP error code.
    Never raises.
    """
    try:
        orch  = _get_orchestrator(test_id)
        state = orch.run()

        if state.status == "FAILED":
            return {
                "error":   "Pipeline failed — one or more agents could not run.",
                "details": state.errors,
            }

        result = orch.build_final_result(state)
        _pipeline_cache[test_id] = result
        return result

    except FileNotFoundError:
        return {
            "error":   f"Required model artifact for test {test_id} not found.",
            "details": [],
        }
    except Exception as exc:
        # Log full traceback server-side; return sanitized message to client
        traceback.print_exc()
        return {
            "error":   f"Unexpected pipeline error: {type(exc).__name__}",
            "details": [],
        }


def get_cached_pipeline(test_id: int) -> dict[str, Any] | None:
    """
    Return the cached pipeline result for test_id, or None if not yet run.

    Callers should check for None and return 404/409 rather than
    silently triggering an expensive pipeline run.
    """
    return _pipeline_cache.get(test_id)


def clear_pipeline_cache(test_id: int | None = None) -> None:
    """Clear cached results. If test_id is None, clear all caches."""
    if test_id is None:
        _pipeline_cache.clear()
        _timeseries_cache.clear()
    else:
        _pipeline_cache.pop(test_id, None)
        _timeseries_cache.pop(test_id, None)


# ── HealthMonitor timeseries ──────────────────────────────────────────────────

def _get_health_monitor(test_id: int) -> Any | None:
    """
    Lazy-initialize HealthMonitor for test_id and cache it.
    Returns None if any model artifact is missing.
    """
    if test_id not in _health_monitors:
        from industrial_health.health.health_monitor import HealthMonitor
        paths = get_artifact_paths(test_id)
        if not all(paths[k].exists() for k in ["rf_model", "rf_scaler", "if_model", "if_scaler"]):
            _health_monitors[test_id] = None
        else:
            _health_monitors[test_id] = HealthMonitor(
                rf_model_path  = paths["rf_model"],
                rf_scaler_path = paths["rf_scaler"],
                if_model_path  = paths["if_model"],
                if_scaler_path = paths["if_scaler"],
            )
    return _health_monitors[test_id]


def get_health_timeseries(test_id: int) -> dict[str, Any]:
    """
    Compute and cache health timeseries from HealthMonitor.score_dataset().

    Returns a dict with:
        "records":     list of row dicts (JSON-serializable)
        "fault_start": int — heuristic label boundary snapshot index
        "n_snapshots": int

    Returns {"error": str, "details": list[str]} on failure.
    """
    if test_id in _timeseries_cache:
        return _timeseries_cache[test_id]

    try:
        monitor = _get_health_monitor(test_id)
        if monitor is None:
            return {
                "error":   f"Health Monitor for test {test_id} could not be loaded — model artifacts missing.",
                "details": [],
            }

        csv_path = FEATURES_DIR / f"test{test_id}_features.csv"
        if not csv_path.exists():
            return {
                "error":   f"Feature CSV for test {test_id} not found.",
                "details": [],
            }

        df_raw    = pd.read_csv(csv_path, parse_dates=["timestamp"])
        result_df = monitor.score_dataset(df_raw)

        fault_start = int(df_raw["fault_start_idx"].iloc[0])

        # Stringify timestamps for JSON serialization
        if "timestamp" in result_df.columns:
            result_df = result_df.copy()
            result_df["timestamp"] = result_df["timestamp"].astype(str)

        records = result_df.to_dict(orient="records")

        payload = {
            "records":     records,
            "fault_start": fault_start,
            "n_snapshots": len(records),
        }
        _timeseries_cache[test_id] = payload
        return payload

    except FileNotFoundError:
        return {
            "error":   f"Required model artifact for test {test_id} not found.",
            "details": [],
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "error":   f"Health timeseries computation failed: {type(exc).__name__}",
            "details": [],
        }


# ── KnowledgeRetriever ────────────────────────────────────────────────────────

def _get_retriever() -> Any | None:
    """
    Lazy-initialize KnowledgeRetriever and cache it at module level.
    Returns None if knowledge_base/ or chroma_db/ directories are missing.
    """
    global _retriever
    if _retriever is None:
        if not KNOWLEDGE_BASE_PATH.exists() or not CHROMA_PATH.exists():
            return None
        from industrial_health.rag import KnowledgeRetriever
        _retriever = KnowledgeRetriever(
            knowledge_base_path=KNOWLEDGE_BASE_PATH,
            chroma_path=CHROMA_PATH,
        )
    return _retriever


def retrieve_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve relevant knowledge chunks for a query via KnowledgeRetriever.

    Returns:
        {"results": list[dict], "count": int} on success
        {"error": str, "details": list[str], "is_ollama_error": bool} on failure

    The "is_ollama_error" flag lets the route return HTTP 503 specifically
    for Ollama unavailability vs HTTP 500 for other errors.
    """
    retriever = _get_retriever()
    if retriever is None:
        return {
            "error":            "Knowledge base not available. "
                                "Ensure knowledge_base/ and chroma_db/ exist.",
            "details":          [],
            "is_ollama_error":  False,
        }

    try:
        results = retriever.retrieve(query.strip(), top_k=top_k)
        return {
            "results": [r.to_dict() for r in results],
            "count":   len(results),
        }
    except RuntimeError as exc:
        return {
            "error":           str(exc),
            "details":         ["Run: python scripts/build_knowledge_base.py"],
            "is_ollama_error": False,
        }
    except Exception as exc:
        msg = str(exc)
        is_ollama = any(
            kw in msg.lower()
            for kw in ["connection", "ollama", "refused", "404", "connect"]
        )
        if is_ollama:
            return {
                "error":           "Embedding service unavailable. Ollama is not running.",
                "details":         ["ollama serve", "ollama pull nomic-embed-text"],
                "is_ollama_error": True,
            }
        traceback.print_exc()
        return {
            "error":           f"Retrieval failed: {type(exc).__name__}",
            "details":         [],
            "is_ollama_error": False,
        }


def get_chunk_count() -> int:
    """Return number of indexed chunks, or 0 on failure."""
    retriever = _get_retriever()
    if retriever is None:
        return 0
    try:
        return retriever.count()
    except Exception:
        return 0
