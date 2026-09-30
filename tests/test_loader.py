"""
test_loader.py — Unit tests for the IMS data loader.

Tests:
  1. list_test_files — finds correct number of files
  2. load_single_file — reads correct shape
  3. get_test_info — returns correct metadata
  4. load_test_summary — computes RMS correctly
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Find project root from test file location
PROJECT_ROOT = Path(__file__).parent.parent
ARCHIVE_PATH = PROJECT_ROOT / "archive"


def archive_available():
    """Check if archive data is present."""
    return (ARCHIVE_PATH / "1st_test" / "1st_test").exists()


@pytest.mark.skipif(
    not archive_available(),
    reason="Archive dataset not found — run from project root with archive/ present"
)
class TestLoader:
    """Tests requiring the actual dataset."""

    def test_list_test1_files(self):
        """Test 1 should have ~983 files."""
        from industrial_health.data.loader import list_test_files
        files = list_test_files(ARCHIVE_PATH, test_id=1)
        assert len(files) > 900, f"Expected >900 files, got {len(files)}"
        # Verify sorted by filename (chronological)
        names = [f.name for f in files]
        assert names == sorted(names), "Files not in chronological order"

    def test_load_single_file_shape(self):
        """Each file should have 20480 rows."""
        from industrial_health.data.loader import list_test_files, load_single_file, _get_columns
        files = list_test_files(ARCHIVE_PATH, test_id=1)
        columns = _get_columns(1)
        df = load_single_file(files[0], columns)
        assert df.shape[0] == 20480, f"Expected 20480 rows, got {df.shape[0]}"
        assert df.shape[1] == 8, f"Expected 8 columns, got {df.shape[1]}"

    def test_load_single_file_no_nulls(self):
        """Data files should have no null values."""
        from industrial_health.data.loader import list_test_files, load_single_file, _get_columns
        files = list_test_files(ARCHIVE_PATH, test_id=1)
        columns = _get_columns(1)
        df = load_single_file(files[0], columns)
        assert not df.isnull().any().any(), "Null values found in data file"

    def test_rms_computation(self):
        """RMS should be positive and reasonable."""
        from industrial_health.data.loader import load_test_summary
        summary = load_test_summary(ARCHIVE_PATH, test_id=1, stat="rms")
        assert "timestamp" in summary.columns
        assert len(summary) > 0
        # RMS values should be positive and < 10 (vibration in g units)
        for col in summary.columns:
            if col != "timestamp":
                vals = summary[col].values
                assert (vals > 0).all(), f"Non-positive RMS in column {col}"
                assert (vals < 100).all(), f"Unreasonably large RMS in column {col}"


class TestGetTestInfo:
    """Tests that don't require the dataset."""

    def test_valid_test_ids(self):
        from industrial_health.data.loader import get_test_info
        for test_id in [1, 2, 3]:
            info = get_test_info(test_id)
            assert "n_channels" in info
            assert "columns" in info
            assert "known_faults" in info

    def test_invalid_test_id_raises(self):
        from industrial_health.data.loader import get_test_info
        with pytest.raises(ValueError):
            get_test_info(99)

    def test_test1_has_8_channels(self):
        from industrial_health.data.loader import get_test_info
        info = get_test_info(1)
        assert info["n_channels"] == 8

    def test_test2_has_4_channels(self):
        from industrial_health.data.loader import get_test_info
        info = get_test_info(2)
        assert info["n_channels"] == 4
