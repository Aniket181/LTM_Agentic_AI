"""
anomaly_model.py — Isolation Forest Anomaly Detection

Detects anomalous bearing behavior without requiring fault labels.

Design:
  - Trained on Normal-period data only (unsupervised).
  - Scores each new observation: negative score → more anomalous.
  - Normalized to 0–1 range: 0 = highly anomalous, 1 = normal.
  - Threshold-based decision: below threshold → ANOMALOUS.

How Normal data is defined:
  - Use training split features where label == NORMAL (0).
  - This trains the Isolation Forest on healthy bearing behavior.
  - Any deviation from this normal profile is flagged as anomalous.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from loguru import logger

from industrial_health.data.preprocess import LABEL_NORMAL, get_feature_columns


# Default Isolation Forest hyperparameters
IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": 0.05,  # expected fraction of outliers in training data
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

ANOMALY_STATUS_NORMAL = "NORMAL"
ANOMALY_STATUS_ANOMALOUS = "ANOMALOUS"


class AnomalyDetectionModel:
    """
    Isolation Forest-based bearing anomaly detector.

    Trained on Normal-period data.
    Scores new observations for deviation from normal behavior.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or IF_PARAMS
        self.model = IsolationForest(**self.params)
        self.feature_cols: list[str] = []
        self.is_trained: bool = False
        self._score_min: float = -1.0
        self._score_max: float = 0.5

    def train(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "label",
    ) -> dict:
        """
        Train Isolation Forest on Normal-period training data.

        Args:
            train_df: Training DataFrame (may contain both Normal + Faulty).
            feature_cols: Feature columns (auto-detected if None).
            label_col: Label column name (0=Normal, 1=Faulty).

        Returns:
            Dictionary with training info.
        """
        if feature_cols is None:
            feature_cols = get_feature_columns(train_df)
        self.feature_cols = feature_cols

        # Train ONLY on Normal data
        normal_df = train_df[train_df[label_col] == LABEL_NORMAL]
        if len(normal_df) == 0:
            raise ValueError("No Normal-labeled samples in training data.")

        X_train = normal_df[feature_cols].values
        logger.info(
            f"Training Isolation Forest on {len(X_train)} Normal samples, "
            f"{len(feature_cols)} features"
        )

        self.model.fit(X_train)
        self.is_trained = True

        # Calibrate score range on training data
        train_scores = self.model.score_samples(X_train)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())
        logger.info(
            f"Score range on Normal training data: "
            f"[{self._score_min:.4f}, {self._score_max:.4f}]"
        )

        return {
            "n_normal_train": len(X_train),
            "score_min": self._score_min,
            "score_max": self._score_max,
        }

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw Isolation Forest score to 0–1 range.
        0 = highly anomalous, 1 = normal.

        Raw score from IsolationForest.score_samples():
          More negative → more anomalous.
          Typical range: -0.7 to +0.2 (varies by data).
        """
        score_range = self._score_max - self._score_min
        if score_range < 1e-10:
            return 1.0
        normalized = (raw_score - self._score_min) / score_range
        return float(np.clip(normalized, 0.0, 1.0))

    def score(self, features: dict | np.ndarray) -> dict:
        """
        Compute anomaly score for a single observation.

        Args:
            features: Feature dict or 1D numpy array.

        Returns:
            Dictionary with anomaly_score, anomaly_status, raw_score.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring.")

        if isinstance(features, dict):
            X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        else:
            X = np.array(features).reshape(1, -1)

        raw_score = float(self.model.score_samples(X)[0])
        pred = int(self.model.predict(X)[0])  # 1=normal, -1=anomaly

        normalized = self._normalize_score(raw_score)

        return {
            "raw_score":       raw_score,
            "anomaly_score":   normalized,         # 0=anomalous, 1=normal
            "anomaly_status":  ANOMALY_STATUS_NORMAL if pred == 1 else ANOMALY_STATUS_ANOMALOUS,
            "is_anomaly":      pred == -1,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all rows in a DataFrame.

        Returns:
            DataFrame with additional columns: raw_score, anomaly_score, anomaly_status.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring.")

        X = df[self.feature_cols].values
        raw_scores = self.model.score_samples(X)
        preds = self.model.predict(X)

        result = df.copy()
        result["raw_score"] = raw_scores
        result["anomaly_score"] = [self._normalize_score(s) for s in raw_scores]
        result["anomaly_status"] = [
            ANOMALY_STATUS_NORMAL if p == 1 else ANOMALY_STATUS_ANOMALOUS
            for p in preds
        ]
        result["is_anomaly"] = preds == -1

        n_anomalies = int((preds == -1).sum())
        logger.info(
            f"Scored {len(df)} samples: {n_anomalies} anomalies detected "
            f"({100*n_anomalies/len(df):.1f}%)"
        )
        return result

    def save(self, path: Path) -> None:
        """Save the trained anomaly model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "score_min": self._score_min,
                "score_max": self._score_max,
            },
            path,
        )
        logger.info(f"Anomaly model saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetectionModel":
        """Load a saved anomaly model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        instance._score_min = data["score_min"]
        instance._score_max = data["score_max"]
        instance.is_trained = True
        logger.info(f"Anomaly model loaded: {path}")
        return instance
