"""
dashboard/services.py — Phase 10: Pipeline Service Layer

RESPONSIBILITIES:
  Bridge between dashboard/app.py (presentation) and existing
  Phase 5–9 pipeline components. This module:

  1. Constructs and caches OrchestratorAgent (Phase 8)
  2. Runs the agent pipeline and returns a structured result dict
  3. Constructs and caches HealthMonitor (Phase 7) for timeseries data
  4. Constructs and caches KnowledgeRetriever (Phase 9) for RAG queries

DESIGN RULES:
  - No ML logic lives here. All computation is in Phase 5–9 modules.
  - All expensive objects (model loading, ChromaDB connection) are cached
    with @st.cache_resource so they survive Streamlit reruns.
  - All heavy computations (pipeline run, scoring) are cached with
    @st.cache_data(ttl=3600) to avoid re-running on every widget interaction.
  - On any error, functions return {"error": str, "details": list}
    instead of raising. app.py is responsible for showing st.error().

DO NOT:
  - Retrain any model.
  - Modify Phase 5–9 logic.
  - Store ML model objects anywhere except @st.cache_resource functions.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so `dashboard` is importable as a package,
# and src/ is on sys.path so `industrial_health` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.config import (
    MODELS_DIR, SCALERS_DIR, FEATURES_DIR,
    KNOWLEDGE_BASE_PATH, CHROMA_PATH,
    get_artifact_paths,
)


# ── OrchestratorAgent (Phase 8) ───────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading pipeline agents…")
def _get_orchestrator(test_id: int):
    """
    Cached factory for OrchestratorAgent.

    Model artifacts are loaded once and reused across Streamlit reruns.
    Uses @st.cache_resource so the agent object (which holds loaded models)
    persists in memory.
    """
    from industrial_health.agents import OrchestratorAgent
    return OrchestratorAgent(
        models_dir   = MODELS_DIR,
        features_dir = FEATURES_DIR,
        scalers_dir  = SCALERS_DIR,
        test_id      = test_id,
    )


@st.cache_data(ttl=3600, show_spinner="Running agentic pipeline…")
def run_pipeline(test_id: int) -> dict[str, Any]:
    """
    Execute the Phase 8 agentic pipeline and return a structured result.

    Returns a dict with keys: test_id, health, fault, anomaly, execution.
    On failure, returns {"error": str, "details": list[str]}.

    The result is cached for 1 hour. Use clear_pipeline_cache() to reset.
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
        return result

    except FileNotFoundError as exc:
        return {
            "error":   f"Required model artifact not found: {exc}",
            "details": [str(exc)],
        }
    except Exception as exc:
        return {
            "error":   f"Unexpected pipeline error: {type(exc).__name__}: {exc}",
            "details": [traceback.format_exc()],
        }


def clear_pipeline_cache() -> None:
    """Clear cached pipeline results to force a re-run on next call."""
    run_pipeline.clear()
    get_health_timeseries.clear()


# ── HealthMonitor timeseries (Phase 7) ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading health monitor…")
def _get_health_monitor(test_id: int):
    """
    Cached factory for HealthMonitor (loads RF + IF models once).

    Returns the HealthMonitor instance, or None if artifacts are missing.
    """
    from industrial_health.health.health_monitor import HealthMonitor

    paths = get_artifact_paths(test_id)
    for key in ["rf_model", "rf_scaler", "if_model", "if_scaler"]:
        if not paths[key].exists():
            return None

    return HealthMonitor(
        rf_model_path  = paths["rf_model"],
        rf_scaler_path = paths["rf_scaler"],
        if_model_path  = paths["if_model"],
        if_scaler_path = paths["if_scaler"],
    )


@st.cache_data(ttl=3600, show_spinner="Computing health timeseries…")
def get_health_timeseries(test_id: int) -> dict[str, Any]:
    """
    Run HealthMonitor.score_dataset() on all snapshots and return the result.

    Returns a dict with:
      "df":            serializable DataFrame as a list-of-records (JSON-safe)
      "fault_start":   int — heuristic fault start snapshot index
      "n_snapshots":   int
    On failure: {"error": str, "details": list[str]}

    The DataFrame is returned as a list of records (not a DataFrame object)
    to satisfy st.cache_data serialization requirements.
    The caller must reconstruct it with pd.DataFrame(result["records"]).
    """
    try:
        monitor = _get_health_monitor(test_id)
        if monitor is None:
            return {
                "error":   "Health Monitor could not be loaded — model artifacts missing.",
                "details": [],
            }

        csv_path = FEATURES_DIR / f"test{test_id}_features.csv"
        if not csv_path.exists():
            return {
                "error":   f"Feature CSV not found: {csv_path}",
                "details": [],
            }

        df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
        result_df = monitor.score_dataset(df_raw)

        fault_start = int(df_raw["fault_start_idx"].iloc[0])

        # Convert to JSON-serializable records so st.cache_data can hash it
        return {
            "records":     result_df.to_dict(orient="records"),
            "fault_start": fault_start,
            "n_snapshots": len(result_df),
            "columns":     list(result_df.columns),
        }

    except Exception as exc:
        return {
            "error":   f"Health timeseries computation failed: {exc}",
            "details": [traceback.format_exc()],
        }


def get_health_timeseries_df(test_id: int) -> tuple[pd.DataFrame | None, dict]:
    """
    Convenience wrapper that returns a DataFrame + metadata dict.

    Returns (df, meta) where df is the full scored DataFrame, or
    (None, {"error": ...}) if computation failed.
    """
    result = get_health_timeseries(test_id)
    if "error" in result:
        return None, result
    df = pd.DataFrame(result["records"])
    meta = {
        "fault_start": result["fault_start"],
        "n_snapshots": result["n_snapshots"],
    }
    return df, meta


# ── KnowledgeRetriever (Phase 9) ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to knowledge base…")
def _get_retriever():
    """
    Cached factory for KnowledgeRetriever.

    ChromaDB connection is persistent and reused across reruns.
    Returns None if the knowledge base directory does not exist.
    """
    from industrial_health.rag import KnowledgeRetriever

    if not KNOWLEDGE_BASE_PATH.exists():
        return None
    if not CHROMA_PATH.exists():
        return None

    retriever = KnowledgeRetriever(
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        chroma_path=CHROMA_PATH,
    )
    return retriever


def retrieve_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve relevant knowledge chunks for a query.

    Returns:
        {"results": list[dict]} on success, each dict having:
            content, source, title, section, chunk_index, distance
        {"error": str, "details": list[str]} on failure

    Common failure causes:
        - Ollama not running (required to embed the query)
        - ChromaDB collection empty (run build_knowledge_base.py first)
        - Empty/whitespace query
    """
    if not query or not query.strip():
        return {"error": "Query must be a non-empty string.", "details": []}

    retriever = _get_retriever()
    if retriever is None:
        return {
            "error":   "Knowledge base not available. "
                       "Check that knowledge_base/ and chroma_db/ directories exist.",
            "details": [],
        }

    try:
        results = retriever.retrieve(query.strip(), top_k=top_k)
        return {
            "results": [r.to_dict() for r in results],
            "count":   len(results),
        }
    except RuntimeError as exc:
        # Empty collection
        return {
            "error":   str(exc),
            "details": ["Run: python scripts/build_knowledge_base.py"],
        }
    except Exception as exc:
        msg = str(exc)
        # Detect Ollama-specific connection failures
        if any(kw in msg.lower() for kw in ["connection", "ollama", "refused", "404"]):
            return {
                "error":   "Ollama is not available. "
                           "Start Ollama and ensure nomic-embed-text is pulled.",
                "details": ["ollama serve", "ollama pull nomic-embed-text"],
            }
        return {
            "error":   f"Retrieval failed: {exc}",
            "details": [traceback.format_exc()],
        }


def get_retriever_chunk_count() -> int:
    """Return the number of chunks in the ChromaDB collection, or 0 on failure."""
    retriever = _get_retriever()
    if retriever is None:
        return 0
    try:
        return retriever.count()
    except Exception:
        return 0
