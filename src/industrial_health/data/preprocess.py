"""
preprocess.py — IMS Data Preprocessing

Handles:
  - Time-based labeling (Normal vs Faulty) using known IMS fault outcomes
  - Train/test splitting (ALWAYS time-based — never random — to prevent leakage)
  - Feature normalization (StandardScaler fitted on train data only)
  - Saving/loading processed datasets

Labeling strategy:
  - IMS dataset has NO explicit fault labels in the files.
  - We use known failure outcomes from the IMS documentation.
  - Last FAULT_FRACTION (20%) of each test's timeline = "Faulty" period.
  - The affected bearing channels are labeled Faulty; others remain Normal.
  - This is the standard academic approach for IMS dataset.

Train/test split strategy:
  - DO NOT use random splitting on time-series data.
  - Use temporal split: first TRAIN_FRACTION of timeline → train, rest → test.
  - This prevents data leakage between time windows.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger

from industrial_health.data.loader import FAULT_FRACTION

# Fraction of timeline used for training (time-based split)
TRAIN_FRACTION = 0.70

# Label constants
LABEL_NORMAL = 0
LABEL_FAULTY = 1
LABEL_NAMES = {LABEL_NORMAL: "Normal", LABEL_FAULTY: "Faulty"}


def assign_labels(
    feature_df: pd.DataFrame,
    test_id: int,
    fault_fraction: float = FAULT_FRACTION,
) -> pd.DataFrame:
    """
    Assign binary fault labels to a feature DataFrame based on time position.

    Strategy:
        - Sort by timestamp (should already be sorted from loader).
        - Last `fault_fraction` of all timestamps → label = Faulty (1).
        - Remaining → Normal (0).
        - This is a simplification; future work can use bearing-specific labels.

    Args:
        feature_df: DataFrame with 'timestamp' column + feature columns.
        test_id: 1, 2, or 3 (used for logging only here).
        fault_fraction: Fraction of tail timestamps to label as Faulty.

    Returns:
        DataFrame with a new 'label' column (0=Normal, 1=Faulty).
    """
    df = feature_df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("feature_df must contain a 'timestamp' column")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    cutoff_idx = int(n * (1 - fault_fraction))

    df["label"] = LABEL_NORMAL
    df.loc[cutoff_idx:, "label"] = LABEL_FAULTY

    n_normal = (df["label"] == LABEL_NORMAL).sum()
    n_faulty = (df["label"] == LABEL_FAULTY).sum()
    logger.info(
        f"Test {test_id} labeling: {n_normal} Normal, {n_faulty} Faulty "
        f"({fault_fraction*100:.0f}% fault fraction)"
    )
    return df


def time_based_split(
    feature_df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-sorted feature DataFrame into train and test sets.

    IMPORTANT: Always time-based. Never random.
    Train = first `train_fraction` of the timeline.
    Test  = remaining `1 - train_fraction` of the timeline.

    Args:
        feature_df: DataFrame sorted by timestamp with 'timestamp' column.
        train_fraction: Fraction of timeline for training (default 0.70).

    Returns:
        (train_df, test_df) — both DataFrames.
    """
    df = feature_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * train_fraction)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        f"Time-based split: train={len(train_df)} rows, test={len(test_df)} rows "
        f"(split at index {split_idx}/{len(df)})"
    )
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (exclude timestamp, label, test_id)."""
    exclude = {"timestamp", "label", "test_id", "bearing_id"}
    return [col for col in df.columns if col not in exclude]


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    scaler_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on train features and transform both train and test.

    IMPORTANT:
        Scaler is fitted on TRAIN data only.
        Test data is only transformed (never used to fit the scaler).
        This prevents test information leaking into normalization.

    Args:
        train_df: Training feature DataFrame.
        test_df: Testing feature DataFrame.
        feature_cols: Columns to normalize. If None, auto-detected.
        scaler_path: If provided, save the fitted scaler here.

    Returns:
        (train_df_normalized, test_df_normalized, fitted_scaler)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Fit ONLY on train
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
    # Transform test using train statistics
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    if scaler_path:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    logger.info(f"Normalized {len(feature_cols)} features (fitted on {len(train_df)} train rows)")
    return train_df, test_df, scaler


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save a processed DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved dataset: {path} ({df.shape[0]} rows × {df.shape[1]} cols)")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a processed DataFrame from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded dataset: {path} ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df
