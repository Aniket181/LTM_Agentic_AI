"""
api/config.py — Phase 11: FastAPI Application Settings

Loads configuration from environment variables (via .env file).
Falls back to safe defaults when variables are not set.

No ML logic lives here. No model paths are stored here.
Model paths are provided by dashboard/config.py which is reused directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ and project root are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env if present (silent if missing — environment variables take precedence)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "")
    if not raw.strip():
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


class APISettings:
    """
    Central API configuration.  Reads from environment at import time.

    Instantiate once at module level; import the singleton `settings`.
    """

    app_name:    str       = _env("APP_NAME",    "Industrial Equipment Health API")
    app_version: str       = _env("APP_VERSION", "1.0.0")
    api_host:    str       = _env("API_HOST",    "127.0.0.1")
    api_port:    int       = int(_env("API_PORT", "8000"))
    log_level:   str       = _env("LOG_LEVEL",   "INFO")

    # CORS — default to Streamlit's local port; configurable via CORS_ORIGINS env var
    # NEVER use "*" by default.
    cors_origins: list[str] = _env_list(
        "CORS_ORIGINS",
        [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
    )


# Module-level singleton — import and use directly
settings = APISettings()
