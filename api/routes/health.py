"""
api/routes/health.py — Phase 11: Service Health Endpoint

GET /health
    Returns API liveness status and artifact availability.
    Does NOT load any models — only checks file existence.
    Safe to call frequently.

GET /api/v1/pipeline/available-tests
    Lists test IDs for which all model artifacts are present.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.config import settings
from api.dependencies import get_pipeline_service
from api.schemas import ServiceStatus, ArtifactStatus, AvailableTestsResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=ServiceStatus,
    summary="Service health check",
    description=(
        "Returns API liveness status and checks whether required model artifacts "
        "exist on disk. Does not load models. Safe to call frequently."
    ),
    tags=["Health"],
)
async def health_check(service=Depends(get_pipeline_service)) -> ServiceStatus:
    raw = service.artifact_status()
    all_ok = all(raw.values())
    return ServiceStatus(
        status    = "ok" if all_ok else "degraded",
        service   = settings.app_name,
        version   = settings.app_version,
        artifacts = ArtifactStatus(**raw),
    )


@router.get(
    "/api/v1/pipeline/available-tests",
    response_model=AvailableTestsResponse,
    summary="List test IDs with trained models",
    tags=["Pipeline"],
)
async def available_tests(service=Depends(get_pipeline_service)) -> AvailableTestsResponse:
    ids = service.get_available_tests()
    return AvailableTestsResponse(
        available_test_ids = ids,
        note = (
            "Test IDs for which all required model artifacts exist. "
            "Currently only Test 1 has trained models (Phase 5 and Phase 6)."
        ),
    )
