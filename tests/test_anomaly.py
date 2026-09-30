"""
test_anomaly.py — Minimal Phase 6 tests for AnomalyDetector.

Tests:
  1. Detector initialization
  2. Detector cannot predict before fit
  3. Successful fit on valid data
  4. Prediction shape and value range
  5. Decision/score output shape
  6. Anomaly predictions contain only expected values (0, 1)
  7. Training data validation (empty, NaN, Inf)
  8. No-Faulty-training protocol (anomaly_chronological_split)
  9. Scaler train-only behavior in anomaly_chronological_split
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, n_feat: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic Phase-3-like DataFrame with Normal (0) and Faulty (1)."""
    rng = np.random.default_rng(seed)
    base = datetime(2003, 10, 22, 12, 0, 0)
    fault_start = int(n * 0.8)
    rows = []
    for idx in range(n):
        label = 0 if idx < fault_start else 1
        row = {
            "test_id":          1,
            "source_file":      f"file_{idx:06d}",
            "timestamp":        base + timedelta(minutes=10 * idx),
            "n_channels":       4,
            "snapshot_index":   idx,
            "label":            label,
            "label_name":       "Normal" if label == 0 else "Faulty",
            "label_method":     "heuristic_time_fraction",
            "fault_start_idx":  fault_start,
        }
        for f in range(n_feat):
            row[f"b1__feat{f}"] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


def _make_X(n: int = 150, n_feat: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, n_feat))


def _fitted_detector(n: int = 150, n_feat: int = 8):
    from industrial_health.models.anomaly_detector import AnomalyDetector
    det = AnomalyDetector(params={"n_estimators": 10, "random_state": 42,
                                   "n_jobs": 1, "contamination": "auto"})
    X = _make_X(n, n_feat)
    feat_names = [f"f{i}" for i in range(n_feat)]
    det.fit(X, feat_names)
    return det, X, feat_names


# ── 1. Initialization ─────────────────────────────────────────────────────────

class TestAnomalyDetectorInit:

    def test_default_params(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector, IF_PARAMS
        det = AnomalyDetector()
        assert det.model is not None
        assert det.is_fitted is False
        assert det.feature_names == []

    def test_custom_params_applied(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 50, "random_state": 7,
                                       "n_jobs": 1, "contamination": "auto"})
        assert det.model.n_estimators == 50
        assert det.model.random_state == 7

    def test_is_fitted_false_before_fit(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        assert det.is_fitted is False


# ── 2. Cannot predict before fit ─────────────────────────────────────────────

class TestCannotPredictBeforeFit:

    def test_predict_raises_if_not_fitted(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        X = _make_X(10, 4)
        with pytest.raises(RuntimeError, match="fitted"):
            det.predict(X)

    def test_decision_scores_raises_if_not_fitted(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        X = _make_X(10, 4)
        with pytest.raises(RuntimeError, match="fitted"):
            det.decision_scores(X)

    def test_evaluate_raises_if_not_fitted(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector()
        X = _make_X(20, 4)
        y = np.zeros(20, dtype=int)
        with pytest.raises(RuntimeError, match="fitted"):
            det.evaluate(X, y)


# ── 3. Successful fit on valid data ──────────────────────────────────────────

class TestSuccessfulFit:

    def test_fit_returns_summary_dict(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1, "contamination": "auto"})
        X = _make_X(100, 6)
        info = det.fit(X, [f"f{i}" for i in range(6)])
        assert "n_train_samples" in info
        assert "n_features" in info
        assert info["n_train_samples"] == 100
        assert info["n_features"] == 6

    def test_is_fitted_after_fit(self):
        det, _, _ = _fitted_detector()
        assert det.is_fitted is True

    def test_feature_names_stored(self):
        det, _, feat_names = _fitted_detector(n_feat=5)
        assert det.feature_names == feat_names


# ── 4. Prediction shape ───────────────────────────────────────────────────────

class TestPredictionShape:

    def test_predict_output_shape_matches_input(self):
        det, X_train, _ = _fitted_detector(n=100, n_feat=8)
        X_new = _make_X(40, 8, seed=99)
        preds = det.predict(X_new)
        assert preds.shape == (40,)

    def test_predict_on_training_data_shape(self):
        det, X_train, _ = _fitted_detector(n=80, n_feat=6)
        preds = det.predict(X_train)
        assert preds.shape == (80,)


# ── 5. Score output shape ─────────────────────────────────────────────────────

class TestScoreOutputShape:

    def test_decision_scores_shape(self):
        det, X_train, _ = _fitted_detector(n=80, n_feat=6)
        X_new = _make_X(30, 6, seed=5)
        scores = det.decision_scores(X_new)
        assert scores.shape == (30,)

    def test_score_samples_shape(self):
        det, X_train, _ = _fitted_detector(n=80, n_feat=6)
        scores = det.score_samples(X_train)
        assert scores.shape == (80,)

    def test_decision_scores_are_negation_of_score_samples(self):
        """anomaly_score = -score_samples() — verify the negation."""
        det, X_train, _ = _fitted_detector(n=80, n_feat=6)
        X_new = _make_X(20, 6, seed=11)
        raw   = det.score_samples(X_new)
        anom  = det.decision_scores(X_new)
        np.testing.assert_allclose(anom, -raw, atol=1e-12)


# ── 6. Anomaly predictions contain only 0 and 1 ──────────────────────────────

class TestPredictionValues:

    def test_predictions_are_0_or_1(self):
        det, X_train, _ = _fitted_detector(n=100, n_feat=8)
        X_new = _make_X(50, 8, seed=22)
        preds = det.predict(X_new)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0, 1}), f"Unexpected values: {unique_vals}"

    def test_prediction_dtype_is_int(self):
        det, X_train, _ = _fitted_detector()
        preds = det.predict(X_train)
        assert np.issubdtype(preds.dtype, np.integer)


# ── 7. Training data validation ───────────────────────────────────────────────

class TestTrainingDataValidation:

    def test_raises_on_empty_X_train(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1, "contamination": "auto"})
        X_empty = np.empty((0, 4))
        with pytest.raises(ValueError, match="empty"):
            det.fit(X_empty, ["f0", "f1", "f2", "f3"])

    def test_raises_on_nan_in_X_train(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1, "contamination": "auto"})
        X = _make_X(50, 4)
        X[3, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            det.fit(X, [f"f{i}" for i in range(4)])

    def test_raises_on_inf_in_X_train(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1, "contamination": "auto"})
        X = _make_X(50, 4)
        X[10, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            det.fit(X, [f"f{i}" for i in range(4)])

    def test_raises_on_1d_X_train(self):
        from industrial_health.models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(params={"n_estimators": 5, "random_state": 0,
                                       "n_jobs": 1, "contamination": "auto"})
        X = np.random.randn(50)  # 1D instead of 2D
        with pytest.raises(ValueError, match="2-D"):
            det.fit(X, ["f0"])


# ── 8. No-Faulty-training protocol ───────────────────────────────────────────

class TestNoFaultyTrainingProtocol:

    def test_training_partition_contains_no_faulty(self):
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=300)
        split = anomaly_chronological_split(df)
        train_labels = split.meta_train["label"].values
        assert (train_labels == 0).all(), (
            f"Faulty samples found in training partition: "
            f"{(train_labels == 1).sum()} rows with label=1"
        )

    def test_faulty_samples_only_in_eval(self):
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=300)
        split = anomaly_chronological_split(df)
        eval_labels = split.y_eval_all
        # Eval set must contain both Normal and Faulty
        assert 0 in np.unique(eval_labels)
        assert 1 in np.unique(eval_labels)

    def test_no_snapshot_index_overlap_between_train_and_eval(self):
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=300)
        split = anomaly_chronological_split(df)
        train_idx = set(split.meta_train["snapshot_index"].values)
        eval_idx  = set(split.meta_eval_all["snapshot_index"].values)
        overlap   = train_idx & eval_idx
        assert len(overlap) == 0, f"Overlap: {len(overlap)} indices"

    def test_total_coverage(self):
        """Training + eval rows must equal total rows in dataset."""
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=300)
        split = anomaly_chronological_split(df)
        total = split.n_normal_train + split.n_normal_eval + split.n_faulty_eval
        assert total == len(df)


# ── 9. Scaler train-only behavior ────────────────────────────────────────────

class TestScalerTrainOnly:

    def test_scaler_mean_matches_normal_train_partition(self):
        """
        scaler.mean_ must match the column means of the Normal training partition
        (not the full dataset or the eval set).
        """
        from industrial_health.models.anomaly_detector import (
            anomaly_chronological_split, METADATA_COLS,
        )
        df = _make_df(n=400, n_feat=4, seed=7)
        split = anomaly_chronological_split(df)
        feat_cols = split.feature_names

        # Recompute expected mean from the raw Normal training rows
        df_sorted = df.sort_values("snapshot_index").reset_index(drop=True)
        normal_df = df_sorted[df_sorted["label"] == 0].reset_index(drop=True)
        n_tr = split.n_normal_train
        X_train_raw = normal_df.iloc[:n_tr][feat_cols].values.astype(np.float64)
        expected_mean = X_train_raw.mean(axis=0)

        np.testing.assert_allclose(split.scaler.mean_, expected_mean, atol=1e-6)

    def test_x_train_scaled_mean_near_zero(self):
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=400, n_feat=4)
        split = anomaly_chronological_split(df)
        col_means = split.X_train.mean(axis=0)
        assert np.abs(col_means).max() < 0.05, (
            f"X_train column means not near 0: max|mean|={np.abs(col_means).max():.4f}"
        )

    def test_scaler_shape_matches_feature_count(self):
        from industrial_health.models.anomaly_detector import anomaly_chronological_split
        df = _make_df(n=300, n_feat=6)
        split = anomaly_chronological_split(df)
        assert split.scaler.mean_.shape[0] == len(split.feature_names)
