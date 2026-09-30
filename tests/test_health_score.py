"""
test_health_score.py — Unit tests for the health score calculator.
"""

import pytest
from industrial_health.health.health_score import (
    compute_health_score, get_status, get_risk_level, get_trend,
    _compute_anomaly_component, _compute_fault_component,
)


NORMAL_ANOMALY_RESULT = {
    "anomaly_score": 1.0,
    "anomaly_status": "NORMAL",
    "is_anomaly": False,
    "raw_score": 0.1,
}

ANOMALOUS_RESULT = {
    "anomaly_score": 0.1,
    "anomaly_status": "ANOMALOUS",
    "is_anomaly": True,
    "raw_score": -0.5,
}

NORMAL_FAULT_RESULT = {
    "fault_class": 0,
    "fault_label": "Normal",
    "confidence": 0.95,
    "prob_normal": 0.95,
    "prob_faulty": 0.05,
}

FAULTY_FAULT_RESULT = {
    "fault_class": 1,
    "fault_label": "Faulty",
    "confidence": 0.90,
    "prob_normal": 0.10,
    "prob_faulty": 0.90,
}


class TestHealthScoreComponents:

    def test_anomaly_component_normal(self):
        comp = _compute_anomaly_component(1.0)
        assert comp == 100.0

    def test_anomaly_component_anomalous(self):
        comp = _compute_anomaly_component(0.0)
        assert comp == 0.0

    def test_fault_component_normal_high_confidence(self):
        comp = _compute_fault_component(NORMAL_FAULT_RESULT)
        assert comp == pytest.approx(95.0, abs=1.0)

    def test_fault_component_faulty_high_confidence(self):
        comp = _compute_fault_component(FAULTY_FAULT_RESULT)
        # 1 - 0.90 = 0.10 → 10%
        assert comp == pytest.approx(10.0, abs=1.0)


class TestGetStatus:

    def test_healthy_range(self):
        assert get_status(95.0) == "HEALTHY"
        assert get_status(90.0) == "HEALTHY"

    def test_normal_range(self):
        assert get_status(89.0) == "NORMAL"
        assert get_status(70.0) == "NORMAL"

    def test_degraded_range(self):
        assert get_status(69.0) == "DEGRADED"
        assert get_status(40.0) == "DEGRADED"

    def test_critical_range(self):
        assert get_status(39.0) == "CRITICAL"
        assert get_status(0.0) == "CRITICAL"


class TestGetTrend:

    def test_stable_single_value(self):
        assert get_trend([80.0]) == "STABLE"

    def test_degrading(self):
        scores = [90, 85, 80, 75, 70]
        assert get_trend(scores) == "DEGRADING"

    def test_improving(self):
        scores = [50, 60, 70, 80, 90]
        assert get_trend(scores) == "IMPROVING"

    def test_stable(self):
        scores = [80, 80, 81, 80, 80]
        assert get_trend(scores) == "STABLE"


class TestComputeHealthScore:

    def test_healthy_equipment(self):
        result = compute_health_score(
            anomaly_result=NORMAL_ANOMALY_RESULT,
            fault_result=NORMAL_FAULT_RESULT,
        )
        assert result["health_score"] > 80
        assert result["status"] in ("HEALTHY", "NORMAL")

    def test_critical_equipment(self):
        result = compute_health_score(
            anomaly_result=ANOMALOUS_RESULT,
            fault_result=FAULTY_FAULT_RESULT,
        )
        assert result["health_score"] < 50

    def test_output_keys_present(self):
        result = compute_health_score(
            anomaly_result=NORMAL_ANOMALY_RESULT,
            fault_result=NORMAL_FAULT_RESULT,
        )
        required_keys = [
            "health_score", "status", "risk_level", "trend",
            "components", "weights", "anomaly_status", "fault_label",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_score_within_range(self):
        result = compute_health_score(
            anomaly_result=NORMAL_ANOMALY_RESULT,
            fault_result=NORMAL_FAULT_RESULT,
        )
        assert 0 <= result["health_score"] <= 100
