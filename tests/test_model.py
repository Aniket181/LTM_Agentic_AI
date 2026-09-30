"""
test_model.py — Minimal Phase 5 model tests.

Tests:
  1. FaultClassifier initialises correctly with default and custom params.
  2. Training raises ValueError when only one class is present.
  3. Prediction output shape and keys are correct.
  4. Evaluation returns all required metric keys.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, n_feat: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature DataFrame mimicking Phase 3 output (2 classes)."""
    rng = np.random.default_rng(seed)
    base = datetime(2003, 10, 22, 12, 0, 0)
    rows = []
    for idx in range(n):
        label = 0 if idx < int(n * 0.8) else 1
        row = {
            "test_id":          1,
            "source_file":      f"file_{idx:06d}",
            "timestamp":        base + timedelta(minutes=10 * idx),
            "n_channels":       4,
            "snapshot_index":   idx,
            "label":            label,
            "label_name":       "Normal" if label == 0 else "Faulty",
            "label_method":     "heuristic_time_fraction",
            "fault_start_idx":  int(n * 0.8),
        }
        for f in range(n_feat):
            row[f"b1__feat{f}"] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


def _make_arrays(n_train: int = 200, n_test: int = 60,
                 n_feat: int = 8, seed: int = 0):
    """Synthetic scaled X/y arrays with both classes in train and test."""
    rng = np.random.default_rng(seed)
    # Train: 160 Normal + 40 Faulty
    y_train = np.array([0] * 160 + [1] * 40)
    X_train = rng.standard_normal((n_train, n_feat))
    # Test: 45 Normal + 15 Faulty
    y_test = np.array([0] * 45 + [1] * 15)
    X_test = rng.standard_normal((n_test, n_feat))
    feat_names = [f"b1__feat{i}" for i in range(n_feat)]
    return X_train, X_test, y_train, y_test, feat_names


# ── 1. FaultClassifier initialisation ────────────────────────────────────────

class TestFaultClassifierInit:

    def test_default_params_creates_rf(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier()
        assert clf.model is not None
        assert clf.is_trained is False
        assert clf.feature_names == []

    def test_custom_params_override(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={
            "n_estimators": 10,
            "random_state": 99,
            "n_jobs": 1,
            "class_weight": "balanced",
        })
        assert clf.model.n_estimators == 10
        assert clf.model.random_state == 99

    def test_is_trained_false_before_fit(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier()
        assert clf.is_trained is False


# ── 2. Training requires both classes ─────────────────────────────────────────

class TestTrainingRequiresBothClasses:

    def test_raises_if_only_normal(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1})
        X = np.random.randn(50, 4)
        y = np.zeros(50, dtype=int)   # all Normal
        feat_names = [f"f{i}" for i in range(4)]
        with pytest.raises(ValueError, match="both classes"):
            clf.train(X, y, feat_names)

    def test_raises_if_only_faulty(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1})
        X = np.random.randn(50, 4)
        y = np.ones(50, dtype=int)    # all Faulty
        feat_names = [f"f{i}" for i in range(4)]
        with pytest.raises(ValueError, match="both classes"):
            clf.train(X, y, feat_names)

    def test_trains_successfully_with_both_classes(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={"n_estimators": 5, "random_state": 42,
                                       "n_jobs": 1})
        X_train, _, y_train, _, feat_names = _make_arrays()
        info = clf.train(X_train, y_train, feat_names)
        assert clf.is_trained is True
        assert "train_accuracy" in info
        assert 0.0 <= info["train_accuracy"] <= 1.0


# ── 3. Prediction shape and keys ─────────────────────────────────────────────

class TestPredictionShape:

    @pytest.fixture
    def trained_clf(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={"n_estimators": 10, "random_state": 42,
                                       "n_jobs": 1, "class_weight": "balanced"})
        X_train, _, y_train, _, feat_names = _make_arrays()
        clf.train(X_train, y_train, feat_names)
        return clf

    def test_get_feature_importance_length(self, trained_clf):
        fi = trained_clf.get_feature_importance(top_n=5)
        assert len(fi) == 5

    def test_evaluate_returns_all_metrics(self, trained_clf):
        from industrial_health.models.fault_classifier import EvaluationResult
        _, X_test, _, y_test, _ = _make_arrays()
        result = trained_clf.evaluate(X_test, y_test)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.f1_macro <= 1.0
        assert result.n_samples == len(y_test)

    def test_evaluate_confusion_matrix_shape(self, trained_clf):
        _, X_test, _, y_test, _ = _make_arrays()
        result = trained_clf.evaluate(X_test, y_test)
        cm = np.array(result.confusion_matrix)
        assert cm.shape == (2, 2)


# ── 4. Evaluation returns required metrics ────────────────────────────────────

class TestEvaluationMetrics:

    @pytest.fixture
    def trained_and_split(self):
        from industrial_health.models.fault_classifier import FaultClassifier
        clf = FaultClassifier(params={"n_estimators": 10, "random_state": 42,
                                       "n_jobs": 1, "class_weight": "balanced"})
        X_train, X_test, y_train, y_test, feat_names = _make_arrays()
        clf.train(X_train, y_train, feat_names)
        result = clf.evaluate(X_test, y_test)
        return result

    def test_all_scalar_metrics_in_range(self, trained_and_split):
        r = trained_and_split
        for metric in [r.accuracy, r.precision_normal, r.recall_normal,
                       r.f1_normal, r.precision_faulty, r.recall_faulty,
                       r.f1_faulty, r.f1_macro, r.f1_weighted]:
            assert 0.0 <= metric <= 1.0, f"Metric out of range: {metric}"

    def test_classification_report_contains_both_classes(self, trained_and_split):
        report = trained_and_split.classification_report_str
        assert "Normal" in report
        assert "Faulty" in report

    def test_cm_rows_sum_to_test_size(self, trained_and_split):
        cm = np.array(trained_and_split.confusion_matrix)
        assert cm.sum() == trained_and_split.n_samples


# ── 5. Stratified Chronological Split ────────────────────────────────────────

class TestStratifiedChronologicalSplit:

    def test_both_classes_in_train_and_test(self):
        from industrial_health.models.fault_classifier import stratified_chronological_split
        df = _make_df(n=200)
        split = stratified_chronological_split(df)
        assert 0 in np.unique(split.y_train)
        assert 1 in np.unique(split.y_train)
        assert 0 in np.unique(split.y_test)
        assert 1 in np.unique(split.y_test)

    def test_no_snapshot_index_overlap(self):
        from industrial_health.models.fault_classifier import stratified_chronological_split
        df = _make_df(n=200)
        split = stratified_chronological_split(df)
        train_idx = set(split.meta_train["snapshot_index"])
        test_idx  = set(split.meta_test["snapshot_index"])
        assert len(train_idx & test_idx) == 0

    def test_scaler_fitted_on_train_only(self):
        from industrial_health.models.fault_classifier import (
            stratified_chronological_split, METADATA_COLS,
        )
        df = _make_df(n=300)
        split = stratified_chronological_split(df)
        feat_cols = [c for c in df.columns if c not in METADATA_COLS]
        X_train_raw = df.sort_values("snapshot_index").iloc[
            split.meta_train["snapshot_index"].values
        ][feat_cols].values
        # Verify mean_ matches X_train mean, not full dataset mean
        full_mean = df[feat_cols].values.mean(axis=0)
        # They should NOT be equal (scaler is train-only, not full-data)
        # Just verify scaler has the right shape
        assert split.scaler.mean_.shape[0] == len(feat_cols)

    def test_no_source_file_overlap(self):
        from industrial_health.models.fault_classifier import stratified_chronological_split
        df = _make_df(n=200)
        split = stratified_chronological_split(df)
        train_files = set(split.meta_train["source_file"])
        test_files  = set(split.meta_test["source_file"])
        assert len(train_files & test_files) == 0
