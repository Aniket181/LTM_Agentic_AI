"""
test_features.py — Unit tests for the feature extractor.
"""

import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestTimeDomainFeatures:
    """Test time-domain feature extraction with synthetic signals."""

    def _make_signal(self, n=20480, freq=100, fs=20000, noise=0.01):
        """Create a synthetic sinusoidal signal."""
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * freq * t) + noise * np.random.randn(n)

    def test_rms_positive(self):
        from industrial_health.features.extractor import _time_domain_features
        signal = self._make_signal()
        features = _time_domain_features(signal)
        assert features["rms"] > 0

    def test_kurtosis_sinusoid(self):
        """Pure sinusoid kurtosis should be ~1.5 (Fisher=True subtracts 3)."""
        from industrial_health.features.extractor import _time_domain_features
        signal = np.sin(np.linspace(0, 100 * np.pi, 20480))
        features = _time_domain_features(signal)
        # Fisher kurtosis of sinusoid ≈ -1.5
        assert -2.0 < features["kurtosis"] < 0.0

    def test_crest_factor_positive(self):
        from industrial_health.features.extractor import _time_domain_features
        signal = self._make_signal()
        features = _time_domain_features(signal)
        assert features["crest_factor"] > 0

    def test_all_features_present(self):
        from industrial_health.features.extractor import _time_domain_features
        signal = self._make_signal()
        features = _time_domain_features(signal)
        expected = ["mean", "rms", "std", "variance", "kurtosis", "skewness",
                    "peak", "peak_to_peak", "crest_factor", "shape_factor"]
        for key in expected:
            assert key in features, f"Missing feature: {key}"

    def test_impulsive_signal_high_kurtosis(self):
        """An impulsive signal should have kurtosis much higher than sinusoid."""
        from industrial_health.features.extractor import _time_domain_features
        signal = np.zeros(20480)
        signal[::100] = 5.0  # periodic impulses
        features = _time_domain_features(signal)
        assert features["kurtosis"] > 10, "Expected high kurtosis for impulsive signal"


class TestFrequencyDomainFeatures:

    def test_dominant_freq_detection(self):
        """Dominant frequency should match injected sinusoid frequency."""
        from industrial_health.features.extractor import _frequency_domain_features
        fs = 20000
        freq = 1000
        t = np.arange(20480) / fs
        signal = np.sin(2 * np.pi * freq * t)
        features = _frequency_domain_features(signal, fs=fs)
        assert abs(features["dominant_freq"] - freq) < 50, (
            f"Expected ~{freq} Hz, got {features['dominant_freq']:.1f} Hz"
        )

    def test_spectral_energy_positive(self):
        from industrial_health.features.extractor import _frequency_domain_features
        signal = np.random.randn(20480)
        features = _frequency_domain_features(signal)
        assert features["spectral_energy"] > 0


class TestExtractFeaturesFromArray:

    def test_output_shape(self):
        """Feature dict should have n_channels × n_features entries."""
        from industrial_health.features.extractor import extract_features_from_array
        data = np.random.randn(20480, 4)
        cols = ["bearing1", "bearing2", "bearing3", "bearing4"]
        features = extract_features_from_array(data, cols, include_freq=True)
        # 4 channels × (10 time + 3 freq) = 52 features + 'filename' excluded here
        assert len(features) == 4 * 13

    def test_column_mismatch_raises(self):
        from industrial_health.features.extractor import extract_features_from_array
        data = np.random.randn(20480, 4)
        cols = ["only_one"]  # wrong number
        with pytest.raises(ValueError):
            extract_features_from_array(data, cols)
