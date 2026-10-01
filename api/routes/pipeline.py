"""
api/routes/pipeline.py — Phase 11: Pipeline Execution and Result Routes

Endpoints:
  POST /api/v1/pipeline/run
      Execute the full Phase 8 agentic pipeline and cache the result.
      Returns the complete PipelineRunResponse.

  GET  /api/v1/pipeline/{test_id}/summary
      Return the CACHED pipeline result for test_id.
      Returns 409 if the pipeline has not been run yet for this test_id.
      Does NOT silently trigger a new pipeline run.

  GET  /api/v1/pipeline/{test_id}/timeseries
      Return the per-snapshot health score timeseries from Phase 7.
      Caches the result on first access.

  GET  /api/v1/pipeline/{test_id}/fault
      Return the Phase 5 RF fault diagnosis result.
      Requires the pipeline to have been run first (409 otherwise).

  GET  /api/v1/pipeline/{test_id}/anomaly
      Return the Phase 6 IF anomaly detection result.
      Requires the pipeline to have been run first (409 otherwise).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import get_pipeline_service
from api.schemas import (
    PipelineRunRequest,
    PipelineRunResponse,
    HealthSummary,
    FaultSummary,
    AnomalySummary,
    ExecutionSummary,
    ExecutionLogEntry,
    StatusCounts,
    TimeseriesResponse,
    HealthSnapshotPoint,
)

router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_pipeline_response(result: dict, test_id: int) -> PipelineRunResponse:
    """Convert build_final_result() dict to PipelineRunResponse."""

    h = result.get("health") or {}
    f = result.get("fault")  or {}
    a = result.get("anomaly") or {}
    e = result.get("execution") or {}

    sc = h.get("status_counts") or {}
    status_counts = StatusCounts(
        Stable    = sc.get("Stable",    0),
        Degrading = sc.get("Degrading", 0),
        Critical  = sc.get("Critical",  0),
    )

    log_entries = [
        ExecutionLogEntry(
            agent      = entry.get("agent",      ""),
            status     = entry.get("status",     ""),
            duration_s = entry.get("duration_s"),
            error      = entry.get("error"),
        )
        for entry in (e.get("execution_log") or [])
    ]

    return PipelineRunResponse(
        test_id = test_id,
        health  = HealthSummary(
            mean_health_score   = h.get("mean_health_score"),
            final_health_score  = h.get("final_health_score"),
            final_health_status = h.get("final_health_status"),
            health_trend        = h.get("health_trend"),
            status_counts       = status_counts,
        ),
        fault = FaultSummary(
            final_rf_pred       = f.get("final_rf_pred"),
            predicted_fault     = f.get("predicted_fault"),
            normal_probability  = f.get("normal_probability"),
            fault_probability   = f.get("fault_probability"),
            fault_confidence    = f.get("fault_confidence"),
        ),
        anomaly = AnomalySummary(
            anomaly_score       = a.get("anomaly_score"),
            anomaly_prediction  = a.get("anomaly_prediction"),
            anomaly_label       = a.get("anomaly_label"),
            n_anomaly_predicted = a.get("n_anomaly_predicted"),
        ),
        execution = ExecutionSummary(
            agents_executed  = e.get("agents_executed",  []),
            execution_order  = e.get("execution_order",  []),
            execution_status = e.get("execution_status", "UNKNOWN"),
            errors           = e.get("errors",           []),
            execution_log    = log_entries,
        ),
    )


def _require_models(test_id: int, service) -> None:
    """Raise 404 if model artifacts for test_id do not exist."""
    if not service._model_artifacts_exist(test_id):
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for test {test_id}. "
                   f"Available tests: {service.get_available_tests()}",
        )


def _require_cached(test_id: int, service) -> dict:
    """
    Return cached pipeline result or raise 409 (Conflict).

    The client must call POST /run first.
    """
    result = service.get_cached_pipeline(test_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No pipeline result cached for test {test_id}. "
                "Call POST /api/v1/pipeline/run first."
            ),
        )
    return result


# ── POST /api/v1/pipeline/run ─────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=PipelineRunResponse,
    summary="Execute the agentic pipeline",
    description=(
        "Run the full Phase 8 agentic pipeline (DataHealthAgent → FaultDiagnosisAgent → AnomalyAgent). "
        "Results are cached in memory for subsequent GET requests. "
        "This operation may take 10–60 seconds depending on dataset size."
    ),
    responses={
        400: {"description": "Invalid or unavailable test_id"},
        404: {"description": "Model artifacts not found for test_id"},
        500: {"description": "Internal pipeline error"},
    },
)
async def run_pipeline(
    request: PipelineRunRequest,
    service=Depends(get_pipeline_service),
) -> PipelineRunResponse:
    test_id = request.test_id

    # Validate artifact availability before attempting run
    if not service._model_artifacts_exist(test_id):
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for test {test_id}. "
                   f"Available tests: {service.get_available_tests()}",
        )

    result = service.run_pipeline(test_id)

    if "error" in result:
        # Determine appropriate HTTP status
        error_msg = result["error"]
        if "not found" in error_msg.lower() or "artifact" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    return _build_pipeline_response(result, test_id)


# ── GET /api/v1/pipeline/{test_id}/summary ────────────────────────────────────

@router.get(
    "/{test_id}/summary",
    response_model=PipelineRunResponse,
    summary="Get cached pipeline summary",
    description=(
        "Returns the cached pipeline result for test_id. "
        "Returns 409 if POST /run has not been called yet for this test. "
        "Does NOT trigger a new pipeline run."
    ),
    responses={
        404: {"description": "No models for test_id"},
        409: {"description": "Pipeline not yet run for this test_id"},
    },
)
async def get_summary(
    test_id: int,
    service=Depends(get_pipeline_service),
) -> PipelineRunResponse:
    _require_models(test_id, service)
    result = _require_cached(test_id, service)
    return _build_pipeline_response(result, test_id)


# ── GET /api/v1/pipeline/{test_id}/timeseries ─────────────────────────────────

@router.get(
    "/{test_id}/timeseries",
    response_model=TimeseriesResponse,
    summary="Per-snapshot health score timeseries",
    description=(
        "Returns the complete health score timeline computed by HealthMonitor.score_dataset(). "
        "Cached on first access. May take 10–30 seconds on first call."
    ),
    responses={
        404: {"description": "Models or feature CSV not found"},
        500: {"description": "Scoring computation failed"},
    },
)
async def get_timeseries(
    test_id: int,
    service=Depends(get_pipeline_service),
) -> TimeseriesResponse:
    if not service._model_artifacts_exist(test_id):
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for test {test_id}.",
        )

    ts = service.get_health_timeseries(test_id)

    if "error" in ts:
        code = 404 if "not found" in ts["error"].lower() else 500
        raise HTTPException(status_code=code, detail=ts["error"])

    # Convert records to HealthSnapshotPoint list
    snapshots = [HealthSnapshotPoint(**row) for row in ts["records"]]

    return TimeseriesResponse(
        test_id      = test_id,
        n_snapshots  = ts["n_snapshots"],
        fault_start  = ts["fault_start"],
        snapshots    = snapshots,
    )


# ── GET /api/v1/pipeline/{test_id}/fault ─────────────────────────────────────

@router.get(
    "/{test_id}/fault",
    response_model=FaultSummary,
    summary="Random Forest fault diagnosis result",
    description=(
        "Returns the Phase 5 RF fault diagnosis result for the final snapshot. "
        "Requires POST /run to have been called first."
    ),
    responses={
        404: {"description": "No models for test_id"},
        409: {"description": "Pipeline not yet run"},
    },
)
async def get_fault(
    test_id: int,
    service=Depends(get_pipeline_service),
) -> FaultSummary:
    _require_models(test_id, service)
    result = _require_cached(test_id, service)
    f = result.get("fault") or {}
    return FaultSummary(
        final_rf_pred      = f.get("final_rf_pred"),
        predicted_fault    = f.get("predicted_fault"),
        normal_probability = f.get("normal_probability"),
        fault_probability  = f.get("fault_probability"),
        fault_confidence   = f.get("fault_confidence"),
    )


# ── GET /api/v1/pipeline/{test_id}/anomaly ───────────────────────────────────

@router.get(
    "/{test_id}/anomaly",
    response_model=AnomalySummary,
    summary="Isolation Forest anomaly detection result",
    description=(
        "Returns the Phase 6 IF anomaly detection result for the final snapshot. "
        "Requires POST /run to have been called first. "
        "Score semantics: HIGHER = MORE ANOMALOUS."
    ),
    responses={
        404: {"description": "No models for test_id"},
        409: {"description": "Pipeline not yet run"},
    },
)
async def get_anomaly(
    test_id: int,
    service=Depends(get_pipeline_service),
) -> AnomalySummary:
    _require_models(test_id, service)
    result = _require_cached(test_id, service)
    a = result.get("anomaly") or {}
    return AnomalySummary(
        anomaly_score       = a.get("anomaly_score"),
        anomaly_prediction  = a.get("anomaly_prediction"),
        anomaly_label       = a.get("anomaly_label"),
        n_anomaly_predicted = a.get("n_anomaly_predicted"),
    )
