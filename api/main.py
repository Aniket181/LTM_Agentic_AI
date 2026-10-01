"""
api/main.py — Phase 11: FastAPI Application Factory

LAUNCH:
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

ARCHITECTURE:
  This file assembles the FastAPI application.
  No ML logic lives here.
  All pipeline logic is in api/services/pipeline_service.py
  All route handlers are in api/routes/

INTERACTIVE DOCS:
  http://127.0.0.1:8000/docs   (Swagger UI)
  http://127.0.0.1:8000/redoc  (ReDoc)

CORS:
  Configurable via CORS_ORIGINS env var.
  Defaults to Streamlit's local port (8501).
  Never uses allow_origins=["*"].

ERROR HANDLING:
  Global handlers ensure no Python tracebacks, filesystem paths, or
  internal details leak to API clients.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root and src/ are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.config import settings
from api.routes import health as health_router
from api.routes import pipeline as pipeline_router
from api.routes import rag as rag_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separating the factory from the module-level `app` makes it easier
    to test (inject mocks via app.dependency_overrides).
    """
    application = FastAPI(
        title       = settings.app_name,
        version     = settings.app_version,
        description = (
            "REST API for the Agentic AI-Based Industrial Equipment Health Monitoring "
            "and Fault Diagnosis system. "
            "Exposes Phase 5 (RF Fault), Phase 6 (IF Anomaly), "
            "Phase 7 (Health Score), Phase 8 (Orchestration), "
            "and Phase 9 (RAG Knowledge) via a versioned REST interface. "
            "\n\n"
            "**⚠ Important:** All fault labels are heuristic temporal labels. "
            "Results are not verified fault-onset annotations and are NOT "
            "suitable for safety-critical decisions."
        ),
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = False,
        allow_methods     = ["GET", "POST"],
        allow_headers     = ["Content-Type"],
    )

    # ── Global exception handlers ─────────────────────────────────────────────
    @application.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        # Never expose filesystem paths to clients
        logger.error(f"FileNotFoundError at {request.url}: {exc}")
        return JSONResponse(
            status_code=404,
            content={"error": "Required resource not found.", "detail": None},
        )

    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"ValueError at {request.url}: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid input.", "detail": str(exc)},
        )

    @application.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        # Log full details server-side; return sanitized response to client
        logger.exception(f"Unhandled exception at {request.url}")
        return JSONResponse(
            status_code=500,
            content={
                "error":  "Internal server error.",
                "detail": "Check server logs for details.",
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health_router.router)
    application.include_router(pipeline_router.router)
    application.include_router(rag_router.router)

    return application


# Module-level app instance — used by uvicorn
app = create_app()
