"""
dashboard/config.py — Phase 10: Dashboard Path Configuration

Central source of truth for all filesystem paths used by the dashboard.
No ML logic lives here.

Test availability is determined dynamically by checking which model
artifacts exist on disk. This means Test 2/3 will become available
automatically once their models are trained in a future phase.
"""

from __future__ import annotations

from pathlib import Path

# ── Project Root ───────────────────────────────────────────────────────────────
# dashboard/ is one level below the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────────
FEATURES_DIR        = PROJECT_ROOT / "data" / "features"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
CHROMA_PATH         = PROJECT_ROOT / "chroma_db"

# ── Model artifact paths ────────────────────────────────────────────────────────
MODELS_DIR  = PROJECT_ROOT / "models"
SCALERS_DIR = PROJECT_ROOT / "models" / "scaler"


def _model_artifacts_exist(test_id: int) -> bool:
    """Return True if all 4 required model artifacts exist for test_id."""
    paths = [
        MODELS_DIR  / f"random_forest_test{test_id}.joblib",
        SCALERS_DIR / f"phase5_test{test_id}_scaler.pkl",
        MODELS_DIR  / f"isolation_forest_test{test_id}.joblib",
        SCALERS_DIR / f"phase6_test{test_id}_scaler.pkl",
    ]
    return all(p.exists() for p in paths)


def get_available_test_ids() -> list[int]:
    """
    Return the list of test IDs for which all model artifacts exist.

    Currently only Test 1 has trained models. When Phase 11+ trains
    models for Test 2 and 3, they will automatically appear here.
    """
    return [tid for tid in [1, 2, 3] if _model_artifacts_exist(tid)]


def get_artifact_paths(test_id: int) -> dict[str, Path]:
    """Return a dict of all model artifact paths for a given test_id."""
    return {
        "rf_model":   MODELS_DIR  / f"random_forest_test{test_id}.joblib",
        "rf_scaler":  SCALERS_DIR / f"phase5_test{test_id}_scaler.pkl",
        "if_model":   MODELS_DIR  / f"isolation_forest_test{test_id}.joblib",
        "if_scaler":  SCALERS_DIR / f"phase6_test{test_id}_scaler.pkl",
        "feature_csv": FEATURES_DIR / f"test{test_id}_features.csv",
    }
