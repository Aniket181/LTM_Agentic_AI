"""
extractor.py — Signal Feature Extraction

Converts raw vibration waveforms (20,480 samples per channel)
into a compact feature vector per snapshot.

Time-domain features (per channel):
    - mean:          Average signal level
    - rms:           Root Mean Square — overall vibration energy
    - std:           Standard deviation — signal spread
    - variance:      Signal variability
    - kurtosis:      Impulsiveness — rises sharply with bearing faults
    - skewness:      Signal asymmetry
    - peak:          Maximum absolute value
    - peak_to_peak:  Max - Min amplitude
    - crest_factor:  Peak / RMS — detects impulsive faults early
    - shape_factor:  RMS / Mean(abs) — signal shape indicator

Frequency-domain features (per channel, via FFT):
    - dominant_freq:   Frequency with highest spectral energy
    - spectral_energy: Total energy in frequency spectrum
    - spectral_centroid: Weighted mean frequency

Usage:
    from industrial_health.features.extractor import extract_features_from_file
    features = extract_features_from_file(filepath, columns)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from loguru import logger


# Sampling frequency for IMS dataset (Hz)
SAMPLING_FREQ = 20_000  # 20 kHz


def _time_domain_features(signal: np.ndarray) -> dict:
    """
    Compute time-domain statistical features for a 1D signal array.

    Args:
        signal: 1D numpy array of vibration values.

    Returns:
        Dictionary of feature_name → float value.
    """
    signal = signal.astype(np.float64)
    abs_signal = np.abs(signal)
    peak = float(np.max(abs_signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    mean_abs = float(np.mean(abs_signal))

    features = {
        "mean":          float(np.mean(signal)),
        "rms":           rms,
        "std":           float(np.std(signal)),
        "variance":      float(np.var(signal)),
        "kurtosis":      float(scipy_kurtosis(signal, fisher=True)),
        "skewness":      float(scipy_skew(signal)),
        "peak":          peak,
        "peak_to_peak":  float(np.max(signal) - np.min(signal)),
        "crest_factor":  float(peak / rms) if rms > 1e-10 else 0.0,
        "shape_factor":  float(rms / mean_abs) if mean_abs > 1e-10 else 0.0,
    }
    return features


def _frequency_domain_features(
    signal: np.ndarray,
    fs: int = SAMPLING_FREQ,
) -> dict:
    """
    Compute frequency-domain features using FFT.

    Args:
        signal: 1D numpy array of vibration values.
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature_name → float value.
    """
    signal = signal.astype(np.float64)
    n = len(signal)

    # Compute FFT magnitude spectrum (one-sided)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    spectral_energy = float(np.sum(fft_mag ** 2))
    if spectral_energy > 1e-10:
        dominant_idx = int(np.argmax(fft_mag))
        dominant_freq = float(freqs[dominant_idx])
        # Spectral centroid: weighted mean frequency
        spectral_centroid = float(np.sum(freqs * fft_mag) / np.sum(fft_mag))
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0

    features = {
        "dominant_freq":     dominant_freq,
        "spectral_energy":   spectral_energy,
        "spectral_centroid": spectral_centroid,
    }
    return features


def extract_features_from_array(
    data: np.ndarray,
    column_names: list[str],
    include_freq: bool = True,
    fs: int = SAMPLING_FREQ,
) -> dict:
    """
    Extract features from a 2D data array (rows=samples, cols=channels).

    Args:
        data:         2D numpy array (n_samples × n_channels).
        column_names: Channel names matching the columns.
        include_freq: If True, include frequency-domain features.
        fs:           Sampling frequency in Hz.

    Returns:
        Flat dictionary: {channel_feature: value, ...}
    """
    if data.shape[1] != len(column_names):
        raise ValueError(
            f"data has {data.shape[1]} columns but {len(column_names)} names provided"
        )

    all_features = {}
    for i, col in enumerate(column_names):
        signal = data[:, i]
        td = _time_domain_features(signal)
        for feat_name, val in td.items():
            all_features[f"{col}__{feat_name}"] = val

        if include_freq:
            fd = _frequency_domain_features(signal, fs=fs)
            for feat_name, val in fd.items():
                all_features[f"{col}__{feat_name}"] = val

    return all_features


def extract_features_from_file(
    filepath: Path,
    column_names: list[str],
    include_freq: bool = True,
    fs: int = SAMPLING_FREQ,
) -> dict:
    """
    Load a single IMS data file and extract all features.

    Args:
        filepath:     Path to the raw data file.
        column_names: Channel names for this test.
        include_freq: Include FFT-based features.
        fs:           Sampling frequency.

    Returns:
        Dictionary of features including 'timestamp' and 'filename'.
    """
    data = pd.read_csv(filepath, sep=r"\s+", header=None, dtype=np.float32).values

    features = extract_features_from_array(
        data, column_names, include_freq=include_freq, fs=fs
    )
    features["filename"] = filepath.name
    return features


def extract_features_for_test(
    archive_path: Path,
    test_id: int,
    include_freq: bool = True,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract features for ALL files in a test run.
    This is the main entry point for building the feature dataset.

    Args:
        archive_path: Path to archive/ directory.
        test_id:      1, 2, or 3.
        include_freq: Include FFT features.
        max_files:    Limit files for quick testing (None = all files).

    Returns:
        DataFrame: one row per file, columns = features + timestamp + test_id.
    """
    from industrial_health.data.loader import list_test_files, _parse_timestamp, _get_columns

    files = list_test_files(archive_path, test_id)
    if max_files:
        files = files[:max_files]

    columns = _get_columns(test_id)
    records = []

    logger.info(f"Extracting features from {len(files)} files for test {test_id}...")

    for i, filepath in enumerate(files):
        if i % 100 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(files)} files...")
        try:
            features = extract_features_from_file(
                filepath, columns, include_freq=include_freq
            )
            ts = _parse_timestamp(filepath.name)
            features["timestamp"] = ts
            features["test_id"] = test_id
            records.append(features)
        except Exception as e:
            logger.warning(f"Skipping {filepath.name}: {e}")

    if not records:
        raise RuntimeError(f"No features extracted for test {test_id}")

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"Test {test_id}: extracted {df.shape[0]} snapshots × {df.shape[1]} features"
    )
    return df


def get_feature_names(column_names: list[str], include_freq: bool = True) -> list[str]:
    """Return the list of feature names that would be generated."""
    td_names = ["mean", "rms", "std", "variance", "kurtosis", "skewness",
                "peak", "peak_to_peak", "crest_factor", "shape_factor"]
    fd_names = ["dominant_freq", "spectral_energy", "spectral_centroid"]

    names = []
    for col in column_names:
        for feat in td_names:
            names.append(f"{col}__{feat}")
        if include_freq:
            for feat in fd_names:
                names.append(f"{col}__{feat}")
    return names
