"""
anomaly_detector.py — Phase 6: Isolation Forest Anomaly Detection

PURPOSE:
    Unsupervised anomaly detection on bearing vibration feature snapshots.
    The Isolation Forest is trained EXCLUSIVELY on Normal-period data. Faulty-
    period samples are never used during model fitting.

TRAINING PROTOCOL:
    The anomaly detector is trained exclusively on the Normal training period.
    Faulty-period samples are not used during fitting.

    Normal period (heuristic: first 80% of run snapshots):
        First 70% of Normal snapshots → anomaly model training
        Remaining 30% of Normal snapshots → evaluation only

    Faulty period (heuristic: final 20% of run snapshots):
        100% → evaluation only (never used for fitting)

LABEL DISCLAIMER:
    Evaluation uses heuristic temporal labels derived from the dataset's
    run-to-failure structure and should not be interpreted as verified
    fault-onset ground truth.

SCORE SEMANTICS (clearly documented):
    IsolationForest.score_samples() returns:
        More negative → more anomalous (outlier)
        Less negative / positive → more normal (inlier)
    IsolationForest.predict() returns:
        +1 → inlier (normal)
        -1 → outlier (anomaly)

    This module exposes:
        anomaly_score:  HIGHER = MORE ANOMALOUS
                        Computed as: -score_samples() (negation)
                        So a high anomaly_score means the model considers the
                        snapshot more anomalous.
        anomaly_pred:   +1 = anomaly    (IsolationForest pred == -1)
                        0  = normal     (IsolationForest pred == +1)

    This sign convention is explicitly documented everywhere to prevent
    silent score-semantic reversal bugs.

LEAKAGE PREVENTION:
    1. StandardScaler fitted ONLY on the Normal training partition.
    2. All other partitions are only transformed, never fitted.
    3. Faulty samples are never used during IsolationForest.fit().
    4. No random shuffling of time-series rows.
    5. No metadata columns used as features.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Metadata columns produced by Phase 3 — never used as features
METADATA_COLS: frozenset[str] = frozenset({
    "test_id", "source_file", "timestamp", "n_channels",
    "snapshot_index", "label", "label_name", "label_method", "fault_start_idx",
})

# Within the Normal period, this fraction goes to model training
NORMAL_TRAIN_FRACTION: float = 0.70

# Default IsolationForest parameters
IF_PARAMS: dict = {
    "n_estimators":  200,
    "contamination": "auto",
    "max_samples":   "auto",
    "max_features":  1.0,
    "random_state":  42,
    "n_jobs":        -1,
}


# ── Return types ──────────────────────────────────────────────────────────────

class AnomalyChronologicalSplit(NamedTuple):
    """Partitions produced by the Phase 6 Normal-only training protocol."""
    X_train:          np.ndarray   # Normal training data (scaled)
    X_eval_normal:    np.ndarray   # Normal evaluation data (scaled)
    X_eval_faulty:    np.ndarray   # Faulty evaluation data (scaled)
    X_eval_all:       np.ndarray   # All evaluation data (scaled), sorted by snapshot_index
    y_eval_all:       np.ndarray   # Heuristic labels for all eval rows {0,1}
    meta_train:       pd.DataFrame
    meta_eval_all:    pd.DataFrame
    feature_names:    list[str]
    scaler:           StandardScaler  # fitted ONLY on X_train
    n_normal_train:   int
    n_normal_eval:    int
    n_faulty_eval:    int
    fault_start_idx:  int


class AnomalyEvaluationResult(NamedTuple):
    """Structured metrics from anomaly evaluation against heuristic labels."""
    accuracy:          float
    precision_normal:  float
    recall_normal:     float
    f1_normal:         float
    precision_faulty:  float
    recall_faulty:     float
    f1_faulty:         float
    f1_macro:          float
    f1_weighted:       float
    n_predicted_anomaly: int
    n_predicted_normal:  int
    n_actual_faulty:     int
    n_actual_normal:     int
    confusion_matrix:    list[list[int]]
    classification_report_str: str
    n_samples:           int


# ── Split function ────────────────────────────────────────────────────────────

def anomaly_chronological_split(
    df: pd.DataFrame,
    normal_train_fraction: float = NORMAL_TRAIN_FRACTION,
) -> AnomalyChronologicalSplit:
    """
    Produce the Phase 6 Normal-only anomaly detection split.

    Algorithm:
        1. Sort all rows by snapshot_index (chronological).
        2. Separate Normal (label=0) and Faulty (label=1) rows.
        3. Within the Normal period:
               First normal_train_fraction → model training
               Remaining                  → evaluation
        4. Faulty period:
               100%                        → evaluation only
        5. Fit StandardScaler ONLY on X_train (Normal training rows).
        6. Transform all other partitions using train statistics.

    IMPORTANT:
        Faulty samples are never used during IsolationForest.fit() or
        during StandardScaler.fit().

    Args:
        df:                    Phase 3 feature DataFrame (all columns).
        normal_train_fraction: Fraction of Normal period used for training.

    Returns:
        AnomalyChronologicalSplit namedtuple.

    Raises:
        ValueError: if either class is absent, or if training partition is empty.
    """
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    df = df.sort_values("snapshot_index", ascending=True).reset_index(drop=True)

    fault_start_idx = int(df["fault_start_idx"].iloc[0])
    normal_df = df[df["label"] == 0].reset_index(drop=True)
    faulty_df = df[df["label"] == 1].reset_index(drop=True)

    if len(normal_df) == 0:
        raise ValueError("No Normal (label=0) samples found in dataset.")
    if len(faulty_df) == 0:
        raise ValueError("No Faulty (label=1) samples found in dataset.")

    # Split Normal period chronologically
    n_normal_train = int(len(normal_df) * normal_train_fraction)
    n_normal_train = max(1, min(n_normal_train, len(normal_df) - 1))

    normal_train = normal_df.iloc[:n_normal_train]
    normal_eval  = normal_df.iloc[n_normal_train:]

    # Faulty: all evaluation
    # Combined evaluation set (Normal_eval + all Faulty), sorted by snapshot_index
    eval_df = pd.concat([normal_eval, faulty_df], ignore_index=True)
    eval_df = eval_df.sort_values("snapshot_index").reset_index(drop=True)

    # Raw arrays
    X_train_raw      = normal_train[feature_cols].values.astype(np.float64)
    X_eval_normal_raw = normal_eval[feature_cols].values.astype(np.float64)
    X_eval_faulty_raw = faulty_df[feature_cols].values.astype(np.float64)
    X_eval_all_raw   = eval_df[feature_cols].values.astype(np.float64)
    y_eval_all       = eval_df["label"].values.astype(int)

    # Fit scaler ONLY on Normal training data
    scaler = StandardScaler()
    X_train       = scaler.fit_transform(X_train_raw)
    X_eval_normal = scaler.transform(X_eval_normal_raw)
    X_eval_faulty = scaler.transform(X_eval_faulty_raw)
    X_eval_all    = scaler.transform(X_eval_all_raw)

    # Metadata
    meta_cols = [c for c in df.columns if c in METADATA_COLS]
    meta_train    = normal_train[meta_cols].copy()
    meta_eval_all = eval_df[meta_cols].copy()

    return AnomalyChronologicalSplit(
        X_train=X_train,
        X_eval_normal=X_eval_normal,
        X_eval_faulty=X_eval_faulty,
        X_eval_all=X_eval_all,
        y_eval_all=y_eval_all,
        meta_train=meta_train,
        meta_eval_all=meta_eval_all,
        feature_names=feature_cols,
        scaler=scaler,
        n_normal_train=n_normal_train,
        n_normal_eval=len(normal_eval),
        n_faulty_eval=len(faulty_df),
        fault_start_idx=fault_start_idx,
    )


# ── Detector class ────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Isolation Forest anomaly detector for Phase 6.

    SCORE SEMANTICS:
        anomaly_score  = -score_samples()  →  HIGHER means MORE ANOMALOUS
        anomaly_pred   = +1 → anomaly  |  0 → normal
        (IsolationForest's raw predict: -1 → outlier, +1 → inlier)

    Accepts pre-scaled X arrays (scaler managed externally by
    anomaly_chronological_split). Does not apply any additional scaling.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params     = {**IF_PARAMS, **(params or {})}
        self.model      = IsolationForest(**self.params)
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, feature_names: list[str]) -> dict:
        """
        Fit the Isolation Forest on Normal-period training data.

        Args:
            X_train:       Pre-scaled Normal-only feature matrix.
            feature_names: Ordered feature column names.

        Returns:
            Dict with fitting summary.

        Raises:
            ValueError: if X_train is empty, or contains NaN/Inf,
                        or has wrong dimensionality.
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2-D, got shape {X_train.shape}.")
        if X_train.shape[0] == 0:
            raise ValueError("X_train is empty — cannot fit on zero samples.")
        if X_train.shape[1] == 0:
            raise ValueError("X_train has zero features.")
        if not np.isfinite(X_train).all():
            n_bad = (~np.isfinite(X_train)).sum()
            raise ValueError(
                f"X_train contains {n_bad} non-finite value(s) (NaN or Inf). "
                "Clean the data before fitting."
            )

        self.feature_names = list(feature_names)
        self.model.fit(X_train)
        self.is_fitted = True

        # Score the training data for reference statistics
        train_scores = self.decision_scores(X_train)
        return {
            "n_train_samples": int(X_train.shape[0]),
            "n_features":      int(X_train.shape[1]),
            "score_mean":      float(np.mean(train_scores)),
            "score_std":       float(np.std(train_scores)),
            "score_min":       float(np.min(train_scores)),
            "score_max":       float(np.max(train_scores)),
        }

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for each sample.

        Returns:
            Integer array where +1 = anomaly, 0 = normal.
            (Converts IsolationForest's -1/+1 to 1/0 for clarity.)

        Raises:
            RuntimeError: if model is not fitted.
        """
        self._require_fitted()
        raw = self.model.predict(X)   # +1 = inlier, -1 = outlier
        # Convert: -1 → 1 (anomaly), +1 → 0 (normal)
        return np.where(raw == -1, 1, 0).astype(int)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores where HIGHER = MORE ANOMALOUS.

        Computed as: -score_samples()
        (IsolationForest.score_samples() returns lower values for outliers.)

        Raises:
            RuntimeError: if model is not fitted.
        """
        self._require_fitted()
        return -self.model.score_samples(X)  # negate so higher = more anomalous

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Raw IsolationForest score_samples().
        More negative = more anomalous. Less negative / positive = more normal.
        Exposed for transparency; prefer decision_scores() for consistent semantics.
        """
        self._require_fitted()
        return self.model.score_samples(X)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_eval: np.ndarray,
        y_heuristic: np.ndarray,
    ) -> AnomalyEvaluationResult:
        """
        Evaluate anomaly predictions against heuristic temporal labels.

        IMPORTANT:
            These metrics measure agreement between Isolation Forest predictions
            and the heuristic temporal labels (final 20% = Faulty). They are NOT
            verified fault-onset metrics.

        Args:
            X_eval:       Pre-scaled evaluation feature matrix.
            y_heuristic:  Heuristic labels {0=Normal, 1=Faulty}.

        Returns:
            AnomalyEvaluationResult with full per-class and aggregate metrics.
        """
        self._require_fitted()

        y_pred = self.predict(X_eval)   # 1 = anomaly, 0 = normal

        cm = confusion_matrix(y_heuristic, y_pred, labels=[0, 1])
        report_str = classification_report(
            y_heuristic, y_pred,
            target_names=["Normal", "Faulty"],
            zero_division=0,
        )

        prec = precision_score(y_heuristic, y_pred, average=None, labels=[0, 1],
                               zero_division=0)
        rec  = recall_score(y_heuristic, y_pred, average=None, labels=[0, 1],
                            zero_division=0)
        f1   = f1_score(y_heuristic, y_pred, average=None, labels=[0, 1],
                        zero_division=0)

        return AnomalyEvaluationResult(
            accuracy=float(accuracy_score(y_heuristic, y_pred)),
            precision_normal=float(prec[0]),
            recall_normal=float(rec[0]),
            f1_normal=float(f1[0]),
            precision_faulty=float(prec[1]),
            recall_faulty=float(rec[1]),
            f1_faulty=float(f1[1]),
            f1_macro=float(f1_score(y_heuristic, y_pred, average="macro",
                                    zero_division=0)),
            f1_weighted=float(f1_score(y_heuristic, y_pred, average="weighted",
                                       zero_division=0)),
            n_predicted_anomaly=int((y_pred == 1).sum()),
            n_predicted_normal=int((y_pred == 0).sum()),
            n_actual_faulty=int((y_heuristic == 1).sum()),
            n_actual_normal=int((y_heuristic == 0).sum()),
            confusion_matrix=cm.tolist(),
            classification_report_str=report_str,
            n_samples=int(len(y_heuristic)),
        )

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save fitted model and feature names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        """Load a previously saved model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        inst = cls()
        inst.model         = data["model"]
        inst.feature_names = data["feature_names"]
        inst.is_fitted     = True
        return inst

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "AnomalyDetector must be fitted before calling this method. "
                "Call fit(X_train, feature_names) first."
            )
