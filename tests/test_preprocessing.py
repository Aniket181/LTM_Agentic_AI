"""
test_preprocessing.py — Minimum automated tests for Phase 4 preprocessing.

Tests:
  1. chronological_split — correct ratio, no overlap, correct ordering
  2. metadata/feature separation — get_feature_columns excludes metadata
  3. scaler fitting — fitted only on train; test uses same statistics
  4. train/test temporal separation — train timestamps precede test timestamps
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Synthetic fixture ─────────────────────────────────────────────────────────

def _make_feature_df(n: int = 100, n_feat: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Create a minimal synthetic feature DataFrame that mimics Phase 3 output.
    snapshot_index is 0..n-1, timestamp increases by 10 minutes each step.
    First 80% labeled Normal (0), last 20% labeled Faulty (1).
    """
    rng = np.random.default_rng(seed)
    base_ts = datetime(2003, 10, 22, 12, 0, 0)

    feat_names = [f"b1__feat{i}" for i in range(n_feat)]
    rows = []
    for idx in range(n):
        row = {
            "test_id":          1,
            "source_file":      f"2003.10.22.{idx:06d}",
            "timestamp":        base_ts + timedelta(minutes=10 * idx),
            "n_channels":       4,
            "snapshot_index":   idx,
            "label":            0 if idx < int(n * 0.8) else 1,
            "label_name":       "Normal" if idx < int(n * 0.8) else "Faulty",
            "label_method":     "heuristic_time_fraction",
            "fault_start_idx":  int(n * 0.8),
        }
        for fn in feat_names:
            row[fn] = float(rng.standard_normal())
        rows.append(row)

    return pd.DataFrame(rows)


# ── Test 1: chronological_split ───────────────────────────────────────────────

class TestChronologicalSplit:

    def test_split_ratio_is_approximately_70_30(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df, train_fraction=0.70)
        assert len(train) == 70
        assert len(test)  == 30

    def test_no_snapshot_index_overlap(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        overlap = set(train["snapshot_index"]) & set(test["snapshot_index"])
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_train_indices_all_less_than_test_indices(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        assert train["snapshot_index"].max() < test["snapshot_index"].min()

    def test_no_shuffling_order_preserved(self):
        """Train rows must come chronologically before test rows."""
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_full_coverage_no_rows_dropped(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        assert len(train) + len(test) == len(df)


# ── Test 2: metadata/feature separation ──────────────────────────────────────

class TestFeatureColumnSeparation:

    def test_metadata_cols_excluded(self):
        from industrial_health.data.preprocessing import get_feature_columns
        df = _make_feature_df(n=10, n_feat=4)
        feat_cols = get_feature_columns(df)
        metadata = {
            "test_id", "source_file", "timestamp", "n_channels",
            "snapshot_index", "label", "label_name", "label_method",
            "fault_start_idx",
        }
        for mc in metadata:
            assert mc not in feat_cols, f"Metadata column leaked into features: {mc}"

    def test_feature_count_correct(self):
        from industrial_health.data.preprocessing import get_feature_columns
        df = _make_feature_df(n=10, n_feat=4)
        feat_cols = get_feature_columns(df)
        assert len(feat_cols) == 4

    def test_feature_names_contain_double_underscore(self):
        """Phase 3 feature naming convention: {channel}__{feature}."""
        from industrial_health.data.preprocessing import get_feature_columns
        df = _make_feature_df(n=10, n_feat=4)
        feat_cols = get_feature_columns(df)
        assert all("__" in c for c in feat_cols)


# ── Test 3: scaler fitting ────────────────────────────────────────────────────

class TestScalerFitting:

    def test_scaler_fitted_on_train_only(self):
        """
        Scaler parameters (mean_, scale_) must equal train column statistics.
        If scaler was accidentally fitted on all data, these will differ.
        """
        from industrial_health.data.preprocessing import (
            chronological_split, get_feature_columns, fit_and_scale,
        )
        df = _make_feature_df(n=200, n_feat=4, seed=7)
        train_df, test_df, _ = chronological_split(df, train_fraction=0.70)
        feat_cols = get_feature_columns(df)

        X_train, X_test, scaler = fit_and_scale(train_df, test_df, feat_cols)

        # Scaler mean should match train column means (not full dataset means)
        expected_means = train_df[feat_cols].values.mean(axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_means, atol=1e-6)

    def test_x_train_mean_approximately_zero(self):
        from industrial_health.data.preprocessing import (
            chronological_split, get_feature_columns, fit_and_scale,
        )
        df = _make_feature_df(n=200, n_feat=4)
        train_df, test_df, _ = chronological_split(df)
        feat_cols = get_feature_columns(df)
        X_train, _, _ = fit_and_scale(train_df, test_df, feat_cols)
        assert np.abs(X_train.mean(axis=0)).max() < 0.05

    def test_x_train_std_approximately_one(self):
        from industrial_health.data.preprocessing import (
            chronological_split, get_feature_columns, fit_and_scale,
        )
        df = _make_feature_df(n=200, n_feat=4)
        train_df, test_df, _ = chronological_split(df)
        feat_cols = get_feature_columns(df)
        X_train, _, _ = fit_and_scale(train_df, test_df, feat_cols)
        assert np.abs(X_train.std(axis=0) - 1.0).max() < 0.05

    def test_x_test_uses_train_scaler(self):
        """
        Manually verify X_test by re-applying the scaler to raw test features.
        The values must match exactly (no refitting on test data).
        """
        from industrial_health.data.preprocessing import (
            chronological_split, get_feature_columns, fit_and_scale,
        )
        df = _make_feature_df(n=200, n_feat=4)
        train_df, test_df, _ = chronological_split(df)
        feat_cols = get_feature_columns(df)
        X_train, X_test, scaler = fit_and_scale(train_df, test_df, feat_cols)

        X_test_expected = scaler.transform(test_df[feat_cols].values)
        np.testing.assert_allclose(X_test, X_test_expected, atol=1e-10)


# ── Test 4: train/test temporal separation ────────────────────────────────────

class TestTemporalSeparation:

    def test_no_source_file_in_both_partitions(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        overlap = set(train["source_file"]) & set(test["source_file"])
        assert len(overlap) == 0

    def test_train_max_timestamp_before_test_min_timestamp(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_label_info_preserved_in_both_partitions(self):
        """label, label_method, fault_start_idx must be present in both splits."""
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        for partition, name in [(train, "train"), (test, "test")]:
            for col in ["label", "label_method", "fault_start_idx"]:
                assert col in partition.columns, (
                    f"Column '{col}' missing from {name} partition"
                )

    def test_label_method_is_heuristic(self):
        from industrial_health.data.preprocessing import chronological_split
        df = _make_feature_df(n=100)
        train, test, _ = chronological_split(df)
        assert (train["label_method"] == "heuristic_time_fraction").all()
        assert (test["label_method"]  == "heuristic_time_fraction").all()
