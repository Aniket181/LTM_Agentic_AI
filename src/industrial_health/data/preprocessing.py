"""
preprocessing.py — Phase 4: Preprocessing & Chronological Data Splitting

PURPOSE:
    Load Phase 3 feature CSV files, apply a leakage-safe chronological 70/30
    split, fit a StandardScaler on the training partition only, and save the
    scaled X/y arrays plus metadata for later use in Phase 5 model training.

LABELING DISCLAIMER:
    The binary Normal/Faulty labels at this stage are HEURISTIC TEMPORAL
    LABELS based on the final 20% of each run's snapshot sequence.
    Explicit fault-onset annotations are NOT available in the IMS dataset.
    These labels must not be described as verified ground truth.

LEAKAGE PREVENTION:
    1. Split is always chronological (by snapshot_index rank, ascending).
       Random shuffling is never applied to time-series data.
    2. StandardScaler is fitted exclusively on X_train.
       X_test is only transformed — never used to compute scaling parameters.
    3. Each test is processed independently.
       Test 1, 2, 3 have different channel schemas and must NOT be concatenated.

SPLIT RULE:
    Sort by snapshot_index (ascending → chronological).
    Train = first TRAIN_FRACTION (0.70) of rows.
    Test  = remaining rows (0.30).
    The split boundary falls between consecutive snapshot indices with no overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_FRACTION: float = 0.70

# All metadata columns produced by Phase 3 extraction
METADATA_COLS: frozenset[str] = frozenset({
    "test_id", "source_file", "timestamp", "n_channels",
    "snapshot_index", "label", "label_name", "label_method", "fault_start_idx",
})

# Columns that form the label vector for ML
LABEL_COL: str = "label"

LABEL_DISCLAIMER: str = (
    "The binary Normal/Faulty labels used at this stage are heuristic temporal "
    "labels based on the final 20% of each run because explicit fault-onset "
    "annotations are not available in the current dataset."
)


# ── Return type ───────────────────────────────────────────────────────────────

class PreprocessedSplit(NamedTuple):
    """All outputs for one test after preprocessing."""
    test_id:         int
    X_train:         np.ndarray      # shape (n_train, n_features)
    X_test:          np.ndarray      # shape (n_test,  n_features)
    y_train:         np.ndarray      # shape (n_train,) — int {0,1}
    y_test:          np.ndarray      # shape (n_test,)  — int {0,1}
    meta_train:      pd.DataFrame    # metadata rows aligned with X_train/y_train
    meta_test:       pd.DataFrame    # metadata rows aligned with X_test/y_test
    feature_names:   list[str]       # ordered list of feature column names
    scaler:          StandardScaler  # fitted on X_train only
    split_idx:       int             # row index of first test row (in sorted df)
    n_features:      int
    n_train:         int
    n_test:          int


# ── Core functions ────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return feature column names from a feature DataFrame.
    Excludes all metadata and label columns produced by Phase 3.
    """
    return [c for c in df.columns if c not in METADATA_COLS]


def load_feature_csv(path: Path) -> pd.DataFrame:
    """
    Load a Phase 3 feature CSV with correct dtypes.
    Parses timestamp as datetime. Validates required columns exist.

    Args:
        path: Path to test{N}_features.csv

    Returns:
        DataFrame with correct dtypes.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError:        if required metadata columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Validate required metadata
    missing = {"snapshot_index", "timestamp", "label", "label_method"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    return df


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split DataFrame into train and test using chronological ordering.

    Sorting rule: ascending snapshot_index (integer rank assigned by Phase 3
    in chronological order). This is equivalent to timestamp ordering and is
    more robust to duplicate or missing timestamps.

    Args:
        df:             Feature DataFrame with snapshot_index column.
        train_fraction: Fraction of rows assigned to training set (default 0.70).

    Returns:
        (train_df, test_df, split_idx)
        split_idx is the first row index of the test partition.

    Raises:
        ValueError: if snapshot_index is not monotonically increasing.
    """
    df = df.sort_values("snapshot_index", ascending=True).reset_index(drop=True)

    if not df["snapshot_index"].is_monotonic_increasing:
        raise ValueError(
            "snapshot_index is not monotonically increasing after sort. "
            "Phase 3 extraction may have produced duplicate indices."
        )

    split_idx = int(len(df) * train_fraction)

    # Ensure at least 1 row in each partition
    split_idx = max(1, min(split_idx, len(df) - 1))

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df, split_idx


def fit_and_scale(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on X_train and transform both X_train and X_test.

    LEAKAGE PREVENTION:
        scaler.fit() is called ONLY on train_df.
        test_df is ONLY transformed — its statistics never influence the scaler.

    Args:
        train_df:     Training partition DataFrame.
        test_df:      Test partition DataFrame.
        feature_cols: List of feature column names to scale.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
        All arrays are float64.
    """
    X_train_raw = train_df[feature_cols].values.astype(np.float64)
    X_test_raw  = test_df[feature_cols].values.astype(np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test_raw)         # transform only on test

    return X_train_scaled, X_test_scaled, scaler


def preprocess_test(
    feature_csv_path: Path,
    test_id: int,
    train_fraction: float = TRAIN_FRACTION,
) -> PreprocessedSplit:
    """
    Full preprocessing pipeline for one IMS test.

    Steps:
        1. Load Phase 3 feature CSV.
        2. Sort chronologically by snapshot_index.
        3. Separate feature columns from metadata columns.
        4. Apply 70/30 chronological split.
        5. Extract y_train, y_test from label column.
        6. Fit StandardScaler on X_train, transform both.
        7. Return PreprocessedSplit namedtuple.

    Args:
        feature_csv_path: Path to test{N}_features.csv
        test_id:          1, 2, or 3 (used for labeling in output)
        train_fraction:   Train partition size (default 0.70)

    Returns:
        PreprocessedSplit with all arrays, metadata, and fitted scaler.
    """
    df = load_feature_csv(feature_csv_path)
    feature_cols = get_feature_columns(df)

    train_df, test_df, split_idx = chronological_split(df, train_fraction)

    # Separate X and y
    y_train = train_df[LABEL_COL].values.astype(int)
    y_test  = test_df[LABEL_COL].values.astype(int)

    X_train, X_test, scaler = fit_and_scale(train_df, test_df, feature_cols)

    # Metadata: all non-feature columns, aligned with X/y rows
    meta_cols  = [c for c in df.columns if c in METADATA_COLS]
    meta_train = train_df[meta_cols].copy()
    meta_test  = test_df[meta_cols].copy()

    return PreprocessedSplit(
        test_id=test_id,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        meta_train=meta_train,
        meta_test=meta_test,
        feature_names=feature_cols,
        scaler=scaler,
        split_idx=split_idx,
        n_features=len(feature_cols),
        n_train=len(y_train),
        n_test=len(y_test),
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_preprocessed(
    split: PreprocessedSplit,
    output_dir: Path,
    scaler_dir: Path,
) -> dict[str, Path]:
    """
    Save all preprocessing outputs for one test.

    Output structure:
        output_dir/test{N}/
            X_train.npy          — scaled training features
            X_test.npy           — scaled test features
            y_train.npy          — training labels (int)
            y_test.npy           — test labels (int)
            meta_train.csv       — training metadata rows
            meta_test.csv        — test metadata rows
            feature_names.txt    — one feature name per line
            split_config.json    — split parameters and sizes

        scaler_dir/
            test{N}_scaler.pkl   — fitted StandardScaler

    Returns:
        Dict mapping name → saved Path.
    """
    import json

    tid = split.test_id
    out = Path(output_dir) / f"test{tid}"
    out.mkdir(parents=True, exist_ok=True)
    Path(scaler_dir).mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # Arrays
    np.save(out / "X_train.npy", split.X_train)
    np.save(out / "X_test.npy",  split.X_test)
    np.save(out / "y_train.npy", split.y_train)
    np.save(out / "y_test.npy",  split.y_test)
    saved["X_train"] = out / "X_train.npy"
    saved["X_test"]  = out / "X_test.npy"
    saved["y_train"] = out / "y_train.npy"
    saved["y_test"]  = out / "y_test.npy"

    # Metadata CSVs
    split.meta_train.to_csv(out / "meta_train.csv", index=False)
    split.meta_test.to_csv(out / "meta_test.csv",  index=False)
    saved["meta_train"] = out / "meta_train.csv"
    saved["meta_test"]  = out / "meta_test.csv"

    # Feature names
    feat_path = out / "feature_names.txt"
    feat_path.write_text("\n".join(split.feature_names))
    saved["feature_names"] = feat_path

    # Split config JSON
    config = {
        "test_id":           split.test_id,
        "n_total":           split.n_train + split.n_test,
        "n_train":           split.n_train,
        "n_test":            split.n_test,
        "split_idx":         split.split_idx,
        "train_fraction":    TRAIN_FRACTION,
        "n_features":        split.n_features,
        "split_method":      "chronological_snapshot_index",
        "scaler":            "sklearn.preprocessing.StandardScaler",
        "scaler_fit":        "train_partition_only",
        "label_col":         LABEL_COL,
        "label_encoding":    {"0": "Normal", "1": "Faulty"},
        "label_disclaimer":  LABEL_DISCLAIMER,
        "metadata_cols":     list(METADATA_COLS),
    }
    cfg_path = out / "split_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    saved["split_config"] = cfg_path

    # Scaler
    scaler_path = Path(scaler_dir) / f"test{tid}_scaler.pkl"
    joblib.dump(split.scaler, scaler_path)
    saved["scaler"] = scaler_path

    return saved


def load_preprocessed(
    output_dir: Path,
    scaler_dir: Path,
    test_id: int,
) -> dict:
    """
    Load previously saved preprocessing outputs for one test.

    Returns a dict with keys: X_train, X_test, y_train, y_test,
    meta_train, meta_test, feature_names, scaler, config.
    """
    import json

    out = Path(output_dir) / f"test{test_id}"

    result = {
        "X_train":      np.load(out / "X_train.npy"),
        "X_test":       np.load(out / "X_test.npy"),
        "y_train":      np.load(out / "y_train.npy"),
        "y_test":       np.load(out / "y_test.npy"),
        "meta_train":   pd.read_csv(out / "meta_train.csv", parse_dates=["timestamp"]),
        "meta_test":    pd.read_csv(out / "meta_test.csv",  parse_dates=["timestamp"]),
        "feature_names": (out / "feature_names.txt").read_text().splitlines(),
        "scaler":       joblib.load(Path(scaler_dir) / f"test{test_id}_scaler.pkl"),
        "config":       json.loads((out / "split_config.json").read_text()),
    }
    return result
