"""
fault_model.py — Random Forest Fault Diagnosis Model

Classifies bearing condition as Normal (0) or Faulty (1)
using extracted time/frequency domain features.

Design decisions:
  - Random Forest: robust, interpretable, works well with tabular features
  - No hyperparameter tuning initially — use sensible defaults
  - Feature importance is exposed for RCA (Root Cause Analysis)
  - Model saved/loaded via joblib
  - Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

All metrics shown are MEASURED results, never assumed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger

from industrial_health.data.preprocess import LABEL_NAMES, get_feature_columns


# Default Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",  # handles class imbalance (more Normal than Faulty)
}


class FaultDiagnosisModel:
    """
    Random Forest-based bearing fault classifier.

    Attributes:
        model: Fitted RandomForestClassifier.
        feature_cols: Feature column names used during training.
        is_trained: Whether the model has been trained.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or RF_PARAMS
        self.model = RandomForestClassifier(**self.params)
        self.feature_cols: list[str] = []
        self.is_trained: bool = False

    def train(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: str = "label",
    ) -> dict:
        """
        Train the Random Forest on the training feature DataFrame.

        Args:
            train_df: DataFrame with features + label column.
            feature_cols: Columns to use as features (auto-detected if None).
            label_col: Name of the label column.

        Returns:
            Dictionary with train accuracy and feature importance.
        """
        if feature_cols is None:
            feature_cols = get_feature_columns(train_df)
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values

        logger.info(f"Training Random Forest on {len(X_train)} samples, {len(feature_cols)} features")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        logger.info(f"Train accuracy: {train_acc:.4f}")

        return {
            "train_accuracy": train_acc,
            "n_train_samples": len(X_train),
            "n_features": len(feature_cols),
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        label_col: str = "label",
    ) -> dict:
        """
        Evaluate the trained model on test data.
        All metrics are MEASURED — never assumed.

        Args:
            test_df: DataFrame with features + label column.
            label_col: Name of the label column.

        Returns:
            Dictionary with all evaluation metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        X_test = test_df[self.feature_cols].values
        y_test = test_df[label_col].values

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        metrics = {
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score":         float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
                zero_division=0,
            ),
            "n_test_samples": len(X_test),
        }

        logger.info(
            f"Test evaluation — Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        return metrics

    def predict(self, features: dict | pd.DataFrame) -> dict:
        """
        Run inference on a single feature dictionary or a DataFrame row.

        Args:
            features: Feature dict or single-row DataFrame.

        Returns:
            Dictionary with fault_class, confidence, and probabilities.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        if isinstance(features, dict):
            X = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        else:
            X = features[self.feature_cols].values.reshape(1, -1)

        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]

        return {
            "fault_class":      pred,
            "fault_label":      LABEL_NAMES[pred],
            "confidence":       float(proba[pred]),
            "prob_normal":      float(proba[0]),
            "prob_faulty":      float(proba[1]) if len(proba) > 1 else 0.0,
        }

    def get_feature_importance(self, top_n: int = 15) -> list[dict]:
        """
        Return top-N most important features.
        Used by RCA Agent for evidence-based root cause analysis.

        Args:
            top_n: Number of top features to return.

        Returns:
            List of {feature, importance} dicts sorted by importance descending.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        return [
            {
                "feature":    self.feature_cols[i],
                "importance": float(importances[i]),
            }
            for i in indices
        ]

    def save(self, path: Path) -> None:
        """Save the trained model and feature column names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_cols": self.feature_cols},
            path,
        )
        logger.info(f"Fault model saved: {path}")

    @classmethod
    def load(cls, path: Path) -> "FaultDiagnosisModel":
        """Load a saved model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_cols = data["feature_cols"]
        instance.is_trained = True
        logger.info(f"Fault model loaded: {path}")
        return instance
