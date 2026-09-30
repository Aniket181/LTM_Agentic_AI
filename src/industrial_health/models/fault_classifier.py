"""
fault_classifier.py — Phase 5: Random Forest Fault Diagnosis Classifier

PURPOSE:
    Train and evaluate a Random Forest classifier for bearing fault diagnosis
    using the Phase 5 Stratified Chronological Split protocol.

SPLIT PROTOCOL (Phase 5 Experimental Split — separate from Phase 4 pipeline):
    The Normal and Faulty periods are partitioned independently to ensure both
    classes are represented in training and evaluation.

    Within each heuristic state period, the training portion chronologically
    precedes the corresponding testing portion. The Normal and Faulty periods
    are partitioned independently to ensure both classes are represented in
    training and evaluation.

LABEL DISCLAIMER:
    The binary Normal/Faulty labels are HEURISTIC TEMPORAL LABELS based on the
    final 20% of each run. They are not precise fault-onset annotations.
    All results should be interpreted as performance under this heuristic protocol.

LEAKAGE PREVENTION:
    1. StandardScaler is fitted ONLY on the Phase 5 X_train subset.
       Validation and test arrays are only transformed using training statistics.
    2. Phase 4 scaler is NOT reused — its fitting partition differs.
    3. No random shuffling is applied to time-series rows.
    4. No metadata columns (timestamp, source_file, etc.) are used as features.
    5. The final test set is never touched until final evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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

# Metadata columns from Phase 3 — never used as model features
METADATA_COLS: frozenset[str] = frozenset({
    "test_id", "source_file", "timestamp", "n_channels",
    "snapshot_index", "label", "label_name", "label_method", "fault_start_idx",
})

LABEL_NAMES: dict[int, str] = {0: "Normal", 1: "Faulty"}

# Phase 5 default Random Forest configuration
RF_PARAMS: dict = {
    "n_estimators":    200,
    "max_depth":       None,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":    "sqrt",
    "random_state":    42,
    "n_jobs":          -1,
    "class_weight":    "balanced",
}

# Within each class period, 70% goes to train, 30% to test
WITHIN_CLASS_TRAIN_FRACTION: float = 0.70


# ── Return type ───────────────────────────────────────────────────────────────

class StratifiedChronologicalSplit(NamedTuple):
    """All partitions produced by the Phase 5 experimental split."""
    X_train:       np.ndarray   # shape (n_train, n_features)
    X_test:        np.ndarray   # shape (n_test,  n_features)
    y_train:       np.ndarray   # int {0,1}
    y_test:        np.ndarray   # int {0,1}
    meta_train:    pd.DataFrame
    meta_test:     pd.DataFrame
    feature_names: list[str]
    scaler:        StandardScaler  # fitted ONLY on X_train
    n_normal_train: int
    n_faulty_train: int
    n_normal_test:  int
    n_faulty_test:  int
    fault_start_idx: int


class EvaluationResult(NamedTuple):
    """Structured metrics from model evaluation."""
    accuracy:        float
    precision_normal: float
    recall_normal:   float
    f1_normal:       float
    precision_faulty: float
    recall_faulty:   float
    f1_faulty:       float
    f1_macro:        float
    f1_weighted:     float
    confusion_matrix: list[list[int]]
    classification_report_str: str
    n_samples:       int


# ── Split function ────────────────────────────────────────────────────────────

def stratified_chronological_split(
    df: pd.DataFrame,
    within_class_train_fraction: float = WITHIN_CLASS_TRAIN_FRACTION,
) -> StratifiedChronologicalSplit:
    """
    Apply the Phase 5 Stratified Chronological Split.

    Algorithm:
        1. Sort all rows by snapshot_index (ascending chronological order).
        2. Separate Normal (label=0) and Faulty (label=1) rows.
        3. Within the Normal period: first 70% → train, remaining 30% → test.
        4. Within the Faulty period:  first 70% → train, remaining 30% → test.
        5. Combine Normal_train + Faulty_train → X_train.
        6. Combine Normal_test  + Faulty_test  → X_test.
        7. Fit StandardScaler ONLY on X_train.
        8. Transform X_test using the train-fitted scaler.

    IMPORTANT:
        Within each heuristic state period, the training portion
        chronologically precedes the corresponding testing portion.
        The Normal and Faulty periods are partitioned independently to
        ensure both classes are represented in training and evaluation.

    Args:
        df: Phase 3 feature DataFrame with all columns including metadata.
        within_class_train_fraction: Train fraction within each class period.

    Returns:
        StratifiedChronologicalSplit namedtuple.

    Raises:
        ValueError: if either class is absent, or if any partition is empty.
    """
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Sort chronologically
    df = df.sort_values("snapshot_index", ascending=True).reset_index(drop=True)

    fault_start_idx = int(df["fault_start_idx"].iloc[0])

    # Separate by class
    normal_df = df[df["label"] == 0].reset_index(drop=True)
    faulty_df = df[df["label"] == 1].reset_index(drop=True)

    if len(normal_df) == 0:
        raise ValueError("No Normal samples found in dataset.")
    if len(faulty_df) == 0:
        raise ValueError("No Faulty samples found in dataset.")

    # Split each class independently (first fraction → train, rest → test)
    n_normal_train = int(len(normal_df) * within_class_train_fraction)
    n_faulty_train = int(len(faulty_df) * within_class_train_fraction)

    # Clamp to ensure at least 1 sample in each partition
    n_normal_train = max(1, min(n_normal_train, len(normal_df) - 1))
    n_faulty_train = max(1, min(n_faulty_train, len(faulty_df) - 1))

    normal_train = normal_df.iloc[:n_normal_train]
    normal_test  = normal_df.iloc[n_normal_train:]
    faulty_train = faulty_df.iloc[:n_faulty_train]
    faulty_test  = faulty_df.iloc[n_faulty_train:]

    # Combine — train: all normal_train + all faulty_train
    # Sort combined train by snapshot_index to preserve internal ordering
    train_df = pd.concat([normal_train, faulty_train], ignore_index=True)
    test_df  = pd.concat([normal_test,  faulty_test],  ignore_index=True)
    train_df = train_df.sort_values("snapshot_index").reset_index(drop=True)
    test_df  = test_df.sort_values("snapshot_index").reset_index(drop=True)

    # Extract arrays
    X_train_raw = train_df[feature_cols].values.astype(np.float64)
    X_test_raw  = test_df[feature_cols].values.astype(np.float64)
    y_train     = train_df["label"].values.astype(int)
    y_test      = test_df["label"].values.astype(int)

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform on train
    X_test  = scaler.transform(X_test_raw)         # transform only on test

    # Metadata
    meta_cols = [c for c in df.columns if c in METADATA_COLS]
    meta_train = train_df[meta_cols].copy()
    meta_test  = test_df[meta_cols].copy()

    return StratifiedChronologicalSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        meta_train=meta_train,
        meta_test=meta_test,
        feature_names=feature_cols,
        scaler=scaler,
        n_normal_train=n_normal_train,
        n_faulty_train=n_faulty_train,
        n_normal_test=len(normal_test),
        n_faulty_test=len(faulty_test),
        fault_start_idx=fault_start_idx,
    )


def validate_split(split: StratifiedChronologicalSplit) -> None:
    """
    Sanity-check the split before training. Raises ValueError on any violation.
    Prints a clear pre-training report.
    """
    errors: list[str] = []

    # Both classes present in train
    train_classes = set(np.unique(split.y_train))
    if 0 not in train_classes:
        errors.append("TRAIN has no Normal (0) samples.")
    if 1 not in train_classes:
        errors.append("TRAIN has no Faulty (1) samples.")

    # Both classes present in test
    test_classes = set(np.unique(split.y_test))
    if 0 not in test_classes:
        errors.append("TEST has no Normal (0) samples.")
    if 1 not in test_classes:
        errors.append("TEST has no Faulty (1) samples.")

    # No row overlap (via snapshot_index)
    train_idx = set(split.meta_train["snapshot_index"])
    test_idx  = set(split.meta_test["snapshot_index"])
    overlap   = train_idx & test_idx
    if overlap:
        errors.append(f"snapshot_index overlap between train/test: {len(overlap)} rows.")

    # No source_file overlap
    train_files = set(split.meta_train["source_file"])
    test_files  = set(split.meta_test["source_file"])
    file_overlap = train_files & test_files
    if file_overlap:
        errors.append(f"source_file overlap: {len(file_overlap)} files.")

    # Shape consistency
    if split.X_train.shape[0] != len(split.y_train):
        errors.append("X_train and y_train row counts differ.")
    if split.X_test.shape[0] != len(split.y_test):
        errors.append("X_test and y_test row counts differ.")
    if split.X_train.shape[1] != split.X_test.shape[1]:
        errors.append("X_train and X_test feature counts differ.")

    # No NaN/Inf
    if np.isnan(split.X_train).any():
        errors.append("NaN found in X_train.")
    if np.isinf(split.X_train).any():
        errors.append("Inf found in X_train.")
    if np.isnan(split.X_test).any():
        errors.append("NaN found in X_test.")
    if np.isinf(split.X_test).any():
        errors.append("Inf found in X_test.")

    if errors:
        msg = "\n".join(f"  ✗ {e}" for e in errors)
        raise ValueError(f"Split validation FAILED:\n{msg}")


# ── Model class ───────────────────────────────────────────────────────────────

class FaultClassifier:
    """
    Random Forest fault classifier for Phase 5.

    Accepts pre-scaled X arrays (from StratifiedChronologicalSplit).
    Does not apply any additional scaling — scaler is managed externally.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params        = {**RF_PARAMS, **(params or {})}
        self.model         = RandomForestClassifier(**self.params)
        self.feature_names: list[str] = []
        self.is_trained:   bool = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> dict:
        """
        Fit the Random Forest on pre-scaled training arrays.

        Args:
            X_train:       Scaled training feature matrix.
            y_train:       Integer label vector {0,1}.
            feature_names: Ordered list of feature column names.

        Returns:
            Dict with training accuracy and class distribution.

        Raises:
            ValueError: if only one class is present in y_train.
        """
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError(
                f"Training requires both classes (0=Normal, 1=Faulty). "
                f"Only class(es) {classes.tolist()} found in y_train."
            )

        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_pred = self.model.predict(X_train)
        train_acc  = float(accuracy_score(y_train, train_pred))

        class_dist = {
            LABEL_NAMES[c]: int((y_train == c).sum()) for c in [0, 1]
        }

        return {
            "train_accuracy":  train_acc,
            "n_train_samples": int(len(y_train)),
            "n_features":      int(len(feature_names)),
            "class_distribution": class_dist,
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationResult:
        """
        Evaluate on the test partition.

        Returns per-class and aggregate metrics.
        All results are MEASURED — not assumed.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)

        # Per-class metrics (label-specific, not macro)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report_str = classification_report(
            y_test, y_pred,
            target_names=["Normal", "Faulty"],
            zero_division=0,
        )

        prec_per_class = precision_score(y_test, y_pred, average=None,
                                         labels=[0, 1], zero_division=0)
        rec_per_class  = recall_score(y_test, y_pred, average=None,
                                      labels=[0, 1], zero_division=0)
        f1_per_class   = f1_score(y_test, y_pred, average=None,
                                  labels=[0, 1], zero_division=0)

        return EvaluationResult(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision_normal=float(prec_per_class[0]),
            recall_normal=float(rec_per_class[0]),
            f1_normal=float(f1_per_class[0]),
            precision_faulty=float(prec_per_class[1]),
            recall_faulty=float(rec_per_class[1]),
            f1_faulty=float(f1_per_class[1]),
            f1_macro=float(f1_score(y_test, y_pred, average="macro",
                                    zero_division=0)),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted",
                                       zero_division=0)),
            confusion_matrix=cm.tolist(),
            classification_report_str=report_str,
            n_samples=int(len(y_test)),
        )

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top-N features sorted by importance descending."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first.")

        importances = self.model.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]

        return pd.DataFrame({
            "feature":    [self.feature_names[i] for i in indices],
            "importance": [float(importances[i])  for i in indices],
            "rank":       list(range(1, top_n + 1)),
        })

    def save(self, path: Path) -> None:
        """Save the trained model and feature names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "FaultClassifier":
        """Load a previously saved model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        inst = cls()
        inst.model         = data["model"]
        inst.feature_names = data["feature_names"]
        inst.is_trained    = True
        return inst
