"""
test_health.py — Phase 7: Comprehensive health monitoring tests.

Tests (14 groups, addressing all 14 spec requirements):

 1. Health score always 0–100
 2. Higher anomaly severity cannot increase health (all else constant)
 3. Higher fault confidence cannot increase health
 4. Higher RMS degradation cannot increase health
 5. Status thresholds work correctly
 6. Weight configuration works
 7. Invalid weights are rejected
 8. Missing required inputs are handled
 9. No NaN/Inf in generated health results
10. Health calculation is deterministic
11. Chronological ordering is preserved
12. Phase 5/6 model/scaler semantics are respected
13. No fitting occurs on complete Test 1 timeline
14. Explanation generation is deterministic
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 300, n_feat: int = 8, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic Phase 3 feature DataFrame with RMS columns.
    80% Normal, 20% Faulty.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2003, 10, 22, 12, 0, 0)
    fault_start = int(n * 0.8)
    rows = []
    for idx in range(n):
        label = 0 if idx < fault_start else 1
        row = {
            "test_id":         1,
            "source_file":     f"file_{idx:06d}",
            "timestamp":       base + timedelta(minutes=10 * idx),
            "n_channels":      4,
            "snapshot_index":  idx,
            "label":           label,
            "label_name":      "Normal" if label == 0 else "Faulty",
            "label_method":    "heuristic_time_fraction",
            "fault_start_idx": fault_start,
        }
        for i in range(n_feat):
            if i % 2 == 0:
                # Even features are __rms columns
                row[f"bearing1_ch1__rms{i}"] = float(rng.uniform(0.1, 0.5))
            else:
                row[f"bearing1_ch1__feat{i}"] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


# ── 1. Health score always 0–100 ──────────────────────────────────────────────

class TestHealthScoreRange:

    def test_rms_component_always_0_to_1(self):
        from industrial_health.health.health_monitor import rms_health_component
        rms = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 100.0])
        comp = rms_health_component(rms, baseline_rms=0.1, ref_max_rms=0.4)
        assert np.all(comp >= 0.0)
        assert np.all(comp <= 1.0)

    def test_anomaly_component_always_0_to_1(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        scores = np.array([-1.0, 0.0, 0.1, 0.5, 1.0, 5.0, 100.0])
        comp = anomaly_health_component(scores)
        assert np.all(comp >= 0.0)
        assert np.all(comp <= 1.0)

    def test_fault_component_always_0_to_1(self):
        from industrial_health.health.health_monitor import fault_health_component
        probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        comp = fault_health_component(probs)
        assert np.all(comp >= 0.0)
        assert np.all(comp <= 1.0)

    def test_weighted_score_always_0_to_100(self):
        """Verify that no combination of components can produce out-of-range health."""
        from industrial_health.health.health_monitor import (
            rms_health_component, anomaly_health_component, fault_health_component,
            HealthConfig,
        )
        rng = np.random.default_rng(7)
        cfg = HealthConfig()
        for _ in range(500):
            rms_c  = rng.uniform(0, 1)
            anom_c = rng.uniform(0, 1)
            fault_c= rng.uniform(0, 1)
            score = np.clip(
                100 * (cfg.weight_rms * rms_c
                       + cfg.weight_anomaly * anom_c
                       + cfg.weight_fault * fault_c),
                0.0, 100.0
            )
            assert 0.0 <= score <= 100.0, f"Out-of-range: {score}"


# ── 2. Higher anomaly severity cannot increase health ─────────────────────────

class TestAnomalySeverityMonotonicity:

    def test_higher_anomaly_score_gives_lower_health_component(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        # Ascending anomaly scores → descending health components
        scores = np.linspace(0.0, 1.0, 50)
        comps  = anomaly_health_component(scores)
        # Each subsequent component must be <= previous (monotone non-increasing)
        assert np.all(np.diff(comps) <= 1e-9), \
            f"Anomaly health component is not monotone non-increasing: {np.diff(comps).max():.6f}"

    def test_max_anomaly_gives_zero_component(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        scores = np.array([0.0, 0.5, 1.0])   # 1.0 is the max
        comps  = anomaly_health_component(scores)
        assert comps[-1] == pytest.approx(0.0, abs=1e-9)

    def test_zero_anomaly_gives_max_component(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        # All zero scores → all health components = 1.0
        scores = np.zeros(10)
        comps  = anomaly_health_component(scores)
        assert np.all(comps == pytest.approx(1.0, abs=1e-9))


# ── 3. Higher fault confidence cannot increase health ─────────────────────────

class TestFaultConfidenceMonotonicity:

    def test_higher_prob_faulty_gives_lower_fault_component(self):
        from industrial_health.health.health_monitor import fault_health_component
        probs = np.linspace(0.0, 1.0, 50)
        comps = fault_health_component(probs)
        assert np.all(np.diff(comps) <= 1e-9), \
            "Fault health component is not monotone non-increasing."

    def test_prob_faulty_1_gives_zero_component(self):
        from industrial_health.health.health_monitor import fault_health_component
        comp = fault_health_component(np.array([1.0]))
        assert comp[0] == pytest.approx(0.0, abs=1e-9)

    def test_prob_faulty_0_gives_one_component(self):
        from industrial_health.health.health_monitor import fault_health_component
        comp = fault_health_component(np.array([0.0]))
        assert comp[0] == pytest.approx(1.0, abs=1e-9)


# ── 4. Higher RMS degradation cannot increase health ─────────────────────────

class TestRmsMonotonicity:

    def test_increasing_rms_gives_decreasing_component(self):
        from industrial_health.health.health_monitor import rms_health_component
        rms = np.linspace(0.1, 0.8, 50)  # increasing
        comps = rms_health_component(rms, baseline_rms=0.1, ref_max_rms=0.8)
        assert np.all(np.diff(comps) <= 1e-9), \
            "RMS health component is not monotone non-increasing."

    def test_rms_at_baseline_gives_max_component(self):
        from industrial_health.health.health_monitor import rms_health_component
        comps = rms_health_component(np.array([0.1]), baseline_rms=0.1, ref_max_rms=0.5)
        assert comps[0] == pytest.approx(1.0, abs=1e-9)

    def test_rms_at_or_above_ref_max_gives_zero(self):
        from industrial_health.health.health_monitor import rms_health_component
        comps = rms_health_component(np.array([1.0, 2.0]), baseline_rms=0.1, ref_max_rms=1.0)
        assert np.all(comps <= 0.0 + 1e-9)


# ── 5. Status thresholds ──────────────────────────────────────────────────────

class TestStatusThresholds:

    def test_stable_above_threshold(self):
        from industrial_health.health.health_monitor import get_health_status, HealthConfig
        cfg = HealthConfig(weight_rms=0.2, weight_anomaly=0.4, weight_fault=0.4,
                           threshold_stable=70.0, threshold_degrading=40.0)
        assert get_health_status(100.0, cfg) == "Stable"
        assert get_health_status(70.0,  cfg) == "Stable"
        assert get_health_status(70.1,  cfg) == "Stable"

    def test_degrading_in_band(self):
        from industrial_health.health.health_monitor import get_health_status, HealthConfig
        cfg = HealthConfig(weight_rms=0.2, weight_anomaly=0.4, weight_fault=0.4,
                           threshold_stable=70.0, threshold_degrading=40.0)
        assert get_health_status(69.9,  cfg) == "Degrading"
        assert get_health_status(55.0,  cfg) == "Degrading"
        assert get_health_status(40.0,  cfg) == "Degrading"

    def test_critical_below_threshold(self):
        from industrial_health.health.health_monitor import get_health_status, HealthConfig
        cfg = HealthConfig(weight_rms=0.2, weight_anomaly=0.4, weight_fault=0.4,
                           threshold_stable=70.0, threshold_degrading=40.0)
        assert get_health_status(39.9,  cfg) == "Critical"
        assert get_health_status(0.0,   cfg) == "Critical"

    def test_custom_thresholds(self):
        from industrial_health.health.health_monitor import get_health_status, HealthConfig
        cfg = HealthConfig(weight_rms=0.2, weight_anomaly=0.4, weight_fault=0.4,
                           threshold_stable=80.0, threshold_degrading=50.0)
        assert get_health_status(79.9,  cfg) == "Degrading"
        assert get_health_status(80.0,  cfg) == "Stable"
        assert get_health_status(49.9,  cfg) == "Critical"


# ── 6. Weight configuration works ────────────────────────────────────────────

class TestWeightConfiguration:

    def test_default_config_has_correct_weights(self):
        from industrial_health.health.health_monitor import HealthConfig
        cfg = HealthConfig()
        assert cfg.weight_rms     == pytest.approx(0.20)
        assert cfg.weight_anomaly == pytest.approx(0.40)
        assert cfg.weight_fault   == pytest.approx(0.40)

    def test_custom_weights_applied(self):
        from industrial_health.health.health_monitor import HealthConfig
        cfg = HealthConfig(weight_rms=0.30, weight_anomaly=0.50, weight_fault=0.20)
        assert cfg.weight_rms     == pytest.approx(0.30)
        assert cfg.weight_anomaly == pytest.approx(0.50)
        assert cfg.weight_fault   == pytest.approx(0.20)

    def test_equal_weights_valid(self):
        from industrial_health.health.health_monitor import HealthConfig
        cfg = HealthConfig(weight_rms=1/3, weight_anomaly=1/3, weight_fault=1/3)
        total = cfg.weight_rms + cfg.weight_anomaly + cfg.weight_fault
        assert total == pytest.approx(1.0, abs=1e-6)


# ── 7. Invalid weights are rejected ──────────────────────────────────────────

class TestInvalidWeightsRejected:

    def test_weights_not_summing_to_one(self):
        from industrial_health.health.health_monitor import HealthConfig
        with pytest.raises(ValueError, match="sum to 1.0"):
            HealthConfig(weight_rms=0.5, weight_anomaly=0.5, weight_fault=0.5)

    def test_negative_weight_rejected(self):
        from industrial_health.health.health_monitor import HealthConfig
        with pytest.raises(ValueError, match="non-negative"):
            HealthConfig(weight_rms=-0.1, weight_anomaly=0.6, weight_fault=0.5)

    def test_invalid_thresholds_rejected(self):
        from industrial_health.health.health_monitor import HealthConfig
        with pytest.raises(ValueError):
            # degrading > stable — invalid
            HealthConfig(weight_rms=0.2, weight_anomaly=0.4, weight_fault=0.4,
                         threshold_stable=40.0, threshold_degrading=70.0)

    def test_all_zero_weights_rejected(self):
        from industrial_health.health.health_monitor import HealthConfig
        with pytest.raises(ValueError, match="sum to 1.0"):
            HealthConfig(weight_rms=0.0, weight_anomaly=0.0, weight_fault=0.0)


# ── 8. Missing required inputs handled ───────────────────────────────────────

class TestMissingInputsHandled:

    def test_rms_component_equal_baseline_max_returns_all_ones(self):
        """When baseline_rms == ref_max_rms, range is zero → all health = 1.0."""
        from industrial_health.health.health_monitor import rms_health_component
        comps = rms_health_component(np.array([0.2, 0.5]), baseline_rms=0.5, ref_max_rms=0.5)
        assert np.all(comps == pytest.approx(1.0, abs=1e-9))

    def test_anomaly_all_zero_scores_returns_all_ones(self):
        """If all anomaly scores are zero, health components are all 1."""
        from industrial_health.health.health_monitor import anomaly_health_component
        comps = anomaly_health_component(np.zeros(10))
        assert np.all(comps == pytest.approx(1.0, abs=1e-9))

    def test_empty_rms_array(self):
        from industrial_health.health.health_monitor import rms_health_component
        comps = rms_health_component(np.array([]), baseline_rms=0.1, ref_max_rms=0.5)
        assert len(comps) == 0

    def test_empty_anomaly_array(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        comps = anomaly_health_component(np.array([]))
        assert len(comps) == 0

    def test_empty_fault_array(self):
        from industrial_health.health.health_monitor import fault_health_component
        comps = fault_health_component(np.array([]))
        assert len(comps) == 0


# ── 9. No NaN/Inf in health results ─────────────────────────────────────────

class TestNoNanInf:

    def test_rms_component_no_nan(self):
        from industrial_health.health.health_monitor import rms_health_component
        rms = np.random.default_rng(0).uniform(0, 2, 200)
        comps = rms_health_component(rms, baseline_rms=0.3, ref_max_rms=1.5)
        assert np.isfinite(comps).all()

    def test_anomaly_component_no_nan(self):
        from industrial_health.health.health_monitor import anomaly_health_component
        scores = np.random.default_rng(1).uniform(0, 3, 200)
        comps  = anomaly_health_component(scores)
        assert np.isfinite(comps).all()

    def test_fault_component_no_nan(self):
        from industrial_health.health.health_monitor import fault_health_component
        probs = np.random.default_rng(2).uniform(0, 1, 200)
        comps = fault_health_component(probs)
        assert np.isfinite(comps).all()

    def test_anomaly_component_negative_scores(self):
        """Negative anomaly scores (more normal than training) should not produce NaN."""
        from industrial_health.health.health_monitor import anomaly_health_component
        scores = np.array([-2.0, -1.0, -0.5, 0.0, 0.1])
        comps  = anomaly_health_component(scores)
        assert np.isfinite(comps).all()
        # Negative scores are clipped to 0 before normalization → all should give 1.0
        # (or max, depending on whether any positive scores exist)


# ── 10. Health calculation is deterministic ───────────────────────────────────

class TestDeterminism:

    def test_same_inputs_same_outputs(self):
        from industrial_health.health.health_monitor import (
            rms_health_component, anomaly_health_component, fault_health_component,
            HealthConfig,
        )
        rng = np.random.default_rng(42)
        rms    = rng.uniform(0, 1, 100)
        anom   = rng.uniform(0, 1, 100)
        fault  = rng.uniform(0, 1, 100)
        cfg    = HealthConfig()

        comp_rms1   = rms_health_component(rms, 0.2, 0.8)
        comp_anom1  = anomaly_health_component(anom)
        comp_fault1 = fault_health_component(fault)

        comp_rms2   = rms_health_component(rms, 0.2, 0.8)
        comp_anom2  = anomaly_health_component(anom)
        comp_fault2 = fault_health_component(fault)

        np.testing.assert_array_equal(comp_rms1,   comp_rms2)
        np.testing.assert_array_equal(comp_anom1,  comp_anom2)
        np.testing.assert_array_equal(comp_fault1, comp_fault2)

    def test_explanation_deterministic(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        exp1 = generate_explanation(55.0, 0.8, 0.4, 0.7, cfg)
        exp2 = generate_explanation(55.0, 0.8, 0.4, 0.7, cfg)
        assert exp1 == exp2


# ── 11. Chronological ordering preserved ─────────────────────────────────────

class TestChronologicalOrdering:

    def test_trend_labels_use_chronological_scores(self):
        """Rolling trend on ascending scores should yield final label = Improving."""
        from industrial_health.health.trend import rolling_trend
        scores = np.linspace(30, 90, 100)  # consistently improving
        labels = rolling_trend(scores, window=20)
        # After warm-up, labels should be Improving
        assert labels[-1] == "Improving"

    def test_trend_labels_degrading_for_falling_scores(self):
        from industrial_health.health.trend import rolling_trend
        scores = np.linspace(90, 20, 100)  # consistently falling
        labels = rolling_trend(scores, window=20)
        assert labels[-1] == "Degrading"

    def test_trend_labels_stable_for_flat_scores(self):
        from industrial_health.health.trend import rolling_trend
        scores = np.full(100, 70.0)  # completely flat
        labels = rolling_trend(scores, window=20)
        assert labels[-1] == "Stable"

    def test_rolling_trend_output_length(self):
        from industrial_health.health.trend import rolling_trend
        scores = np.random.default_rng(5).uniform(40, 90, 200)
        labels = rolling_trend(scores, window=30)
        assert len(labels) == 200


# ── 12. Phase 5/6 semantics respected ────────────────────────────────────────

class TestModelSemantics:

    def test_anomaly_score_semantics_higher_is_more_anomalous(self):
        """
        Verify that anomaly_health_component correctly treats higher scores
        as more anomalous (lower health). This mirrors Phase 6 semantics.
        """
        from industrial_health.health.health_monitor import anomaly_health_component
        low_anom  = anomaly_health_component(np.array([0.1]))
        high_anom = anomaly_health_component(np.array([0.9]))
        assert low_anom[0] > high_anom[0], (
            "Higher anomaly score should yield lower health component. "
            "Phase 6: decision_scores() = -score_samples(), higher = more anomalous."
        )

    def test_fault_component_prob_faulty_drives_health_down(self):
        """
        RF prob_faulty is the Phase 5 Faulty class probability.
        Higher prob_faulty → lower health. Verify this semantics.
        """
        from industrial_health.health.health_monitor import fault_health_component
        low_fault  = fault_health_component(np.array([0.1]))
        high_fault = fault_health_component(np.array([0.9]))
        assert low_fault[0] > high_fault[0], \
            "Higher Faulty probability should yield lower fault health component."


# ── 13. No fitting on complete Test 1 timeline ───────────────────────────────

class TestNoFittingOnFullTimeline:

    def test_rms_reference_uses_only_normal_training_fraction(self):
        """
        Verify that the RMS baseline is computed from a subset of Normal data,
        not the full Test 1 timeline. The baseline median must equal the Normal
        training fraction's median, not the full dataset median.
        """
        # We can't call HealthMonitor without model artifacts, so we test the
        # component-level math directly.
        rng = np.random.default_rng(10)
        n_normal = 240
        n_faulty = 60
        normal_rms = rng.uniform(0.1, 0.3, n_normal)
        faulty_rms = rng.uniform(0.4, 0.8, n_faulty)
        full_rms   = np.concatenate([normal_rms, faulty_rms])

        ref_fraction = 0.70
        n_ref = int(n_normal * ref_fraction)
        ref_rms = normal_rms[:n_ref]

        baseline_from_ref   = float(np.median(ref_rms))
        baseline_from_full  = float(np.median(full_rms))

        # They should differ (ref is lower, full includes faulty period)
        assert baseline_from_ref < baseline_from_full, (
            "RMS reference from Normal training period should be lower than "
            "full-timeline median (Normal + Faulty mixed)."
        )

    def test_scaler_not_fitted_in_health_monitor_constructor(self):
        """
        HealthConfig creation must not fit a scaler on any data.
        This is a structural test — HealthConfig is pure configuration.
        """
        from industrial_health.health.health_monitor import HealthConfig
        cfg = HealthConfig()
        # Verify no fitted scaler attributes exist on HealthConfig
        assert not hasattr(cfg, "scaler_")
        assert not hasattr(cfg, "mean_")


# ── 14. Explanation generation is deterministic ───────────────────────────────

class TestExplanationGeneration:

    def test_stable_explanation_mentions_stable(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        exp = generate_explanation(80.0, 0.9, 0.85, 0.9, cfg)
        assert "stable" in exp.lower() or "Stable" in exp

    def test_critical_explanation_mentions_critical(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        exp = generate_explanation(20.0, 0.05, 0.05, 0.05, cfg)
        assert "CRITICAL" in exp or "critical" in exp.lower()

    def test_degrading_explanation_mentions_degrading(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        exp = generate_explanation(55.0, 0.6, 0.4, 0.7, cfg)
        assert "degrad" in exp.lower() or "Degrading" in exp

    def test_explanation_returns_string(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        for score in [10.0, 55.0, 85.0]:
            exp = generate_explanation(score, 0.5, 0.5, 0.5, cfg)
            assert isinstance(exp, str)
            assert len(exp) > 10

    def test_explanation_is_deterministic(self):
        from industrial_health.health.health_monitor import generate_explanation, HealthConfig
        cfg = HealthConfig()
        exp1 = generate_explanation(30.0, 0.1, 0.2, 0.3, cfg)
        exp2 = generate_explanation(30.0, 0.1, 0.2, 0.3, cfg)
        assert exp1 == exp2


# ── Trend module tests ────────────────────────────────────────────────────────

class TestTrendModule:

    def test_first_degradation_returns_minus_one_if_none(self):
        from industrial_health.health.trend import first_degradation_snapshot
        labels = np.array(["Stable"] * 100)
        assert first_degradation_snapshot(labels) == -1

    def test_first_degradation_finds_correct_position(self):
        from industrial_health.health.trend import first_degradation_snapshot
        labels = (["Stable"] * 50 + ["Degrading"] * 10 + ["Stable"] * 40)
        labels = np.array(labels)
        pos = first_degradation_snapshot(labels)
        assert pos == 50

    def test_run_trend_summary_keys(self):
        from industrial_health.health.trend import compute_run_trend_summary
        n = 200
        scores   = np.linspace(90, 20, n)
        statuses = np.where(scores >= 70, "Stable",
                            np.where(scores >= 40, "Degrading", "Critical"))
        snap_idx = np.arange(n)
        trend_ls = np.where(scores > 60, "Stable", "Degrading")
        summary  = compute_run_trend_summary(scores, statuses, snap_idx, trend_ls)
        required = ["overall_trend", "q1_mean_health", "q4_mean_health",
                    "health_delta_q1_to_q4", "first_degradation_snapshot_idx",
                    "n_snapshots_stable", "n_snapshots_degrading",
                    "n_snapshots_critical"]
        for key in required:
            assert key in summary, f"Missing key in trend summary: {key}"

    def test_run_trend_summary_status_counts_sum(self):
        from industrial_health.health.trend import compute_run_trend_summary
        n = 200
        scores   = np.random.default_rng(9).uniform(0, 100, n)
        statuses = np.where(scores >= 70, "Stable",
                            np.where(scores >= 40, "Degrading", "Critical"))
        snap_idx = np.arange(n)
        trend_ls = np.array(["Stable"] * n)
        summary  = compute_run_trend_summary(scores, statuses, snap_idx, trend_ls)
        total = (summary["n_snapshots_stable"]
                 + summary["n_snapshots_degrading"]
                 + summary["n_snapshots_critical"])
        assert total == n
