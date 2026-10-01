"""
api/dependencies.py — Phase 11: FastAPI Dependency Providers

Provides get_pipeline_service() as a FastAPI Depends() provider.

The singleton PipelineService module is returned directly.
Tests can override this via app.dependency_overrides to inject mocks.

Keeping this thin makes the injection point stable and easy to test.
"""

from __future__ import annotations

import api.services.pipeline_service as _svc


def get_pipeline_service():
    """
    FastAPI dependency that returns the pipeline service module.

    Usage in routes:
        from api.dependencies import get_pipeline_service
        from fastapi import Depends

        @router.post("...")
        async def some_route(service=Depends(get_pipeline_service)):
            result = service.run_pipeline(test_id)
    """
    return _svc
