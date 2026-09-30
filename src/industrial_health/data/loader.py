"""
loader.py — IMS Bearing Dataset Loader

Reads the raw IMS bearing dataset from the archive/ folder.
Each file is a tab-delimited text file with no headers.
Filename = timestamp (YYYY.MM.DD.HH.MM.SS).

Dataset structure:
  archive/
    1st_test/1st_test/   → 8-column files (4 bearings × 2 channels each)
    2nd_test/2nd_test/   → 4-column files (4 bearings × 1 channel each)
    3rd_test/4th_test/txt/ → 4-column files

IMS fault outcomes (from official documentation):
  Test 1: Bearing 3 outer race, Bearing 4 roller element
  Test 2: Bearing 1 outer race
  Test 3: Bearing 3 outer race
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Column definitions per test ──────────────────────────────────────────────

# Test 1: 8 channels (4 bearings × 2 accelerometers each)
COLUMNS_TEST1 = [
    "bearing1_ch1", "bearing1_ch2",
    "bearing2_ch1", "bearing2_ch2",
    "bearing3_ch1", "bearing3_ch2",
    "bearing4_ch1", "bearing4_ch2",
]

# Test 2 & 3: 4 channels (4 bearings × 1 accelerometer each)
COLUMNS_TEST2 = ["bearing1", "bearing2", "bearing3", "bearing4"]
COLUMNS_TEST3 = ["bearing1", "bearing2", "bearing3", "bearing4"]

# ── Known fault outcomes from IMS documentation ──────────────────────────────
IMS_FAULT_OUTCOMES = {
    "test1": {
        "bearing3": "outer_race_fault",
        "bearing4": "roller_element_fault",
    },
    "test2": {
        "bearing1": "outer_race_fault",
    },
    "test3": {
        "bearing3": "outer_race_fault",
    },
}

# Fraction of test duration considered as "faulty" period
# (last FAULT_FRACTION of timestamps → labeled Faulty)
FAULT_FRACTION = 0.20


def _parse_timestamp(filename: str) -> Optional[datetime]:
    """Parse IMS filename (YYYY.MM.DD.HH.MM.SS) to datetime."""
    try:
        return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return None


def _get_test_path(archive_path: Path, test_id: int) -> Path:
    """Return the actual data directory for a given test ID."""
    paths = {
        1: archive_path / "1st_test" / "1st_test",
        2: archive_path / "2nd_test" / "2nd_test",
        3: archive_path / "3rd_test" / "4th_test" / "txt",
    }
    if test_id not in paths:
        raise ValueError(f"test_id must be 1, 2, or 3. Got: {test_id}")
    return paths[test_id]


def _get_columns(test_id: int) -> list:
    """Return column names for a given test ID."""
    mapping = {1: COLUMNS_TEST1, 2: COLUMNS_TEST2, 3: COLUMNS_TEST3}
    return mapping[test_id]


def load_single_file(
    filepath: Path,
    columns: list,
    n_cols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a single IMS data file into a DataFrame.

    Args:
        filepath: Path to the data file.
        columns:  Column names to assign.
        n_cols:   Expected number of columns (used for validation).

    Returns:
        DataFrame with shape (20480, n_cols) and named columns.

    Raises:
        ValueError if file has unexpected column count.
    """
    try:
        df = pd.read_csv(filepath, sep=r"\s+", header=None, dtype=np.float32)
    except Exception as e:
        raise IOError(f"Cannot read file {filepath}: {e}") from e

    actual_cols = df.shape[1]
    if n_cols and actual_cols != n_cols:
        raise ValueError(
            f"Expected {n_cols} columns but got {actual_cols} in {filepath.name}"
        )

    if actual_cols != len(columns):
        # Trim or pad columns list to match actual data
        columns = columns[:actual_cols]

    df.columns = columns
    return df


def list_test_files(archive_path: Path, test_id: int) -> list[Path]:
    """
    List all data files for a given test, sorted chronologically.

    Args:
        archive_path: Path to the archive/ directory.
        test_id: 1, 2, or 3.

    Returns:
        List of Path objects sorted by timestamp filename.
    """
    test_dir = _get_test_path(archive_path, test_id)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    files = []
    for f in test_dir.iterdir():
        if f.is_file() and _parse_timestamp(f.name) is not None:
            files.append(f)

    if not files:
        raise FileNotFoundError(f"No timestamped data files found in: {test_dir}")

    # Sort chronologically by filename (timestamp)
    files.sort(key=lambda p: p.name)
    logger.info(f"Test {test_id}: found {len(files)} files in {test_dir}")
    return files


def load_test_summary(
    archive_path: Path,
    test_id: int,
    stat: str = "rms",
) -> pd.DataFrame:
    """
    Load all files for a test and compute a single statistic per file per channel.
    This is the lightweight summary used for EDA and degradation analysis.

    Args:
        archive_path: Path to archive/ directory.
        test_id: 1, 2, or 3.
        stat: Statistic to compute per file: 'rms', 'mean', 'std', 'kurtosis'.

    Returns:
        DataFrame with columns [timestamp, channel1, channel2, ...].
        One row per file (one row per 10-minute snapshot).
    """
    files = list_test_files(archive_path, test_id)
    columns = _get_columns(test_id)

    records = []
    for filepath in files:
        ts = _parse_timestamp(filepath.name)
        try:
            df = load_single_file(filepath, columns)
        except (IOError, ValueError) as e:
            logger.warning(f"Skipping {filepath.name}: {e}")
            continue

        row = {"timestamp": ts}
        if stat == "rms":
            for col in df.columns:
                row[col] = float(np.sqrt(np.mean(df[col].values ** 2)))
        elif stat == "mean":
            for col in df.columns:
                row[col] = float(df[col].mean())
        elif stat == "std":
            for col in df.columns:
                row[col] = float(df[col].std())
        elif stat == "kurtosis":
            from scipy.stats import kurtosis
            for col in df.columns:
                row[col] = float(kurtosis(df[col].values))
        else:
            raise ValueError(f"Unknown stat: {stat}. Use: rms, mean, std, kurtosis")

        records.append(row)

    if not records:
        raise RuntimeError(f"No valid files loaded for test {test_id}")

    summary = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Test {test_id} summary: {summary.shape[0]} snapshots loaded (stat={stat})")
    return summary


def get_test_info(test_id: int) -> dict:
    """Return metadata about a given IMS test."""
    info = {
        1: {
            "test_id": 1,
            "n_channels": 8,
            "columns": COLUMNS_TEST1,
            "n_bearings": 4,
            "channels_per_bearing": 2,
            "date_range": "2003-10-22 to 2003-11-25",
            "known_faults": IMS_FAULT_OUTCOMES["test1"],
            "notes": "8-channel; Bearing 3 outer race; Bearing 4 roller element",
        },
        2: {
            "test_id": 2,
            "n_channels": 4,
            "columns": COLUMNS_TEST2,
            "n_bearings": 4,
            "channels_per_bearing": 1,
            "date_range": "2004-02-12 to 2004-02-19",
            "known_faults": IMS_FAULT_OUTCOMES["test2"],
            "notes": "4-channel; Bearing 1 outer race failure",
        },
        3: {
            "test_id": 3,
            "n_channels": 4,
            "columns": COLUMNS_TEST3,
            "n_bearings": 4,
            "channels_per_bearing": 1,
            "date_range": "2004-03-04 to 2004-04-04",
            "known_faults": IMS_FAULT_OUTCOMES["test3"],
            "notes": "4-channel; Bearing 3 outer race failure",
        },
    }
    if test_id not in info:
        raise ValueError(f"test_id must be 1, 2, or 3. Got: {test_id}")
    return info[test_id]
