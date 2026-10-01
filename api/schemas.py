"""
api/schemas.py — Phase 11: Pydantic v2 Request and Response Models

All schemas are mapped directly to the actual output of:
  - OrchestratorAgent.build_final_result()  (Phase 8)
  - HealthMonitor.score_dataset()            (Phase 7)
  - KnowledgeRetriever.retrieve()            (Phase 9)

Field names, types, and descriptions match the inspected source code exactly.
No ML logic lives here.

HEURISTIC LABEL DISCLAIMER (present in all fault/anomaly schemas):
  RF model was trained on temporal heuristic labels (final 20% = Faulty).
  Probabilities are NOT calibrated Bayesian fault probabilities.

ANOMALY SCORE SEMANTICS (present in anomaly schema):
  decision_scores() = -score_samples(). HIGHER = MORE ANOMALOUS.
  anomaly_pred: 1 = anomaly, 0 = normal.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    """Request body for POST /api/v1/pipeline/run."""

    test_id: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "IMS Test dataset ID (1, 2, or 3). "
            "Only Test 1 has trained models in Phase 11."
        ),
    )


class RAGQueryRequest(BaseModel):
    """Request body for POST /api/v1/rag/query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language query against the bearing knowledge base.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1–10).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error response
# ─────────────────────────────────────────────────────────────────────────────

class APIError(BaseModel):
    """Standardised error response. Never exposes internal paths or tracebacks."""

    error:  str
    detail: str | None              = None
    hint:   list[str] | None        = None


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactStatus(BaseModel):
    test1_models:   bool
    knowledge_base: bool
    chroma_db:      bool


class ServiceStatus(BaseModel):
    """Response for GET /health."""

    status:    str            # "ok" | "degraded"
    service:   str
    version:   str
    artifacts: ArtifactStatus


class AvailableTestsResponse(BaseModel):
    """Response for GET /api/v1/pipeline/available-tests."""

    available_test_ids: list[int]
    note:               str


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — health sub-schema
# ─────────────────────────────────────────────────────────────────────────────

class StatusCounts(BaseModel):
    Stable:    int
    Degrading: int
    Critical:  int


class HealthSummary(BaseModel):
    mean_health_score:   float | None = Field(
        None, description="Mean health score across all snapshots (0–100)."
    )
    final_health_score:  float | None = Field(
        None, description="Health score of the final (most recent) snapshot (0–100)."
    )
    final_health_status: str | None   = Field(
        None, description="Health status of the final snapshot: Stable | Degrading | Critical."
    )
    health_trend:        str | None   = Field(
        None, description="Overall trend direction: Improving | Stable | Degrading."
    )
    status_counts:       StatusCounts | None = Field(
        None, description="Count of snapshots in each status bucket."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — fault sub-schema
# ─────────────────────────────────────────────────────────────────────────────

class FaultSummary(BaseModel):
    final_rf_pred:      int | None    = Field(None, description="0=Normal, 1=Faulty")
    predicted_fault:    str | None    = Field(None, description="'Normal' or 'Faulty'")
    normal_probability: float | None  = Field(None, description="RF P(Normal) for final snapshot")
    fault_probability:  float | None  = Field(None, description="RF P(Faulty) for final snapshot")
    fault_confidence:   float | None  = Field(None, description="Confidence in predicted class")
    label_disclaimer:   str           = Field(
        "RF trained on heuristic temporal labels (final 20% of run = Faulty). "
        "Probabilities are not calibrated Bayesian fault probabilities.",
        description="Disclaimer about label quality.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — anomaly sub-schema
# ─────────────────────────────────────────────────────────────────────────────

class AnomalySummary(BaseModel):
    anomaly_score:       float | None = Field(
        None,
        description=(
            "IF decision score for the final snapshot. "
            "HIGHER = MORE ANOMALOUS (decision_scores = -score_samples)."
        ),
    )
    anomaly_prediction:  int | None   = Field(None, description="1=Anomaly, 0=Normal")
    anomaly_label:       str | None   = Field(None, description="'Normal' or 'Anomaly'")
    n_anomaly_predicted: int | None   = Field(
        None, description="Total anomaly-predicted snapshots across the full run."
    )
    score_semantics: str = Field(
        "decision_scores() = -score_samples(). HIGHER = MORE ANOMALOUS. "
        "anomaly_pred: 1=anomaly, 0=normal.",
        description="Score interpretation.",
    )
    label_disclaimer: str = Field(
        "IF trained on Normal-only data (Phase 6). "
        "Evaluation uses heuristic temporal labels — not verified fault-onset annotations.",
        description="Disclaimer about label quality.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — execution sub-schema
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionLogEntry(BaseModel):
    agent:      str
    status:     str
    duration_s: float | None = None
    error:      str | None   = None


class ExecutionSummary(BaseModel):
    agents_executed:  list[str]
    execution_order:  list[str]
    execution_status: str
    errors:           list[str]
    execution_log:    list[ExecutionLogEntry]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — full response
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunResponse(BaseModel):
    """
    Response for POST /api/v1/pipeline/run and GET /api/v1/pipeline/{test_id}/summary.

    Maps directly to OrchestratorAgent.build_final_result() output.
    """

    test_id:   int
    health:    HealthSummary
    fault:     FaultSummary
    anomaly:   AnomalySummary
    execution: ExecutionSummary


# ─────────────────────────────────────────────────────────────────────────────
# Health timeseries
# ─────────────────────────────────────────────────────────────────────────────

class HealthSnapshotPoint(BaseModel):
    """
    A single row from HealthMonitor.score_dataset() output DataFrame.

    Columns mapped directly from confirmed source code inspection.
    """

    snapshot_index:            int
    timestamp:                 str | None  = Field(None, description="ISO 8601 timestamp string")
    label:                     int | None  = Field(None, description="Heuristic label: 0=Normal, 1=Faulty")
    label_name:                str | None  = None
    composite_rms:             float
    anomaly_score:             float
    anomaly_pred:              int
    rf_pred:                   int
    prob_normal:               float
    prob_faulty:               float
    health_rms_component:      float
    health_anomaly_component:  float
    health_fault_component:    float
    health_score:              float
    health_status:             str
    explanation:               str | None  = None


class TimeseriesResponse(BaseModel):
    """Response for GET /api/v1/pipeline/{test_id}/timeseries."""

    test_id:          int
    n_snapshots:      int
    fault_start:      int = Field(
        description=(
            "Heuristic label boundary snapshot index. "
            "NOT a verified fault-onset annotation."
        )
    )
    snapshots:        list[HealthSnapshotPoint]
    disclaimer:       str = Field(
        default=(
            "fault_start marks the heuristic temporal label boundary (final 20% of run = Faulty). "
            "This is not a verified fault-onset timestamp."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────────────────────────────────────

class RAGResult(BaseModel):
    """
    A single retrieved chunk. Maps directly to RetrievalResult.to_dict() output.
    """

    content:       str
    source:        str
    title:         str
    section:       str
    chunk_index:   int
    distance:      float  = Field(description="Cosine distance (0=identical, 2=orthogonal)")
    relevance_pct: float  = Field(
        description="Approximate relevance percentage: max(0, (1 - distance/2)) * 100"
    )


class RAGQueryResponse(BaseModel):
    """Response for POST /api/v1/rag/query."""

    query:           str
    count:           int
    results:         list[RAGResult]
    disclaimer:      str = Field(
        default=(
            "Retrieval only. No LLM generation. "
            "Source: knowledge_base/ Markdown documents. "
            "Results are retrieved passages, not generated answers."
        )
    )
    embedding_model: str = Field(default="nomic-embed-text (Ollama)")
