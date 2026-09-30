"""
analysis_pipeline.py — End-to-End Analysis Pipeline

Ties together all components:
  1. Feature extraction from raw IMS files
  2. Labeling + train/test split
  3. Model training (Random Forest + Isolation Forest)
  4. Evaluation
  5. Saving models and feature datasets

This is the main script to run during Phase 3–5 training.

Usage:
  python scripts/train_pipeline.py --test_id 1
  OR import and call directly:
    from industrial_health.pipeline.analysis_pipeline import AnalysisPipeline
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from industrial_health.data.loader import get_test_info
from industrial_health.data.preprocess import (
    assign_labels, time_based_split, normalize_features,
    save_dataset, load_dataset, get_feature_columns,
)
from industrial_health.features.extractor import extract_features_for_test
from industrial_health.models.fault_model import FaultDiagnosisModel
from industrial_health.models.anomaly_model import AnomalyDetectionModel


class AnalysisPipeline:
    """
    End-to-end pipeline: raw data → features → labels → trained models.

    Args:
        archive_path:  Path to archive/ directory.
        data_path:     Path to data/ directory (processed + features).
        models_path:   Path to models/ directory.
    """

    def __init__(
        self,
        archive_path: Path,
        data_path: Path,
        models_path: Path,
    ):
        self.archive_path = Path(archive_path)
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def extract_features(
        self,
        test_id: int,
        include_freq: bool = True,
        max_files: Optional[int] = None,
        force: bool = False,
    ) -> object:
        """
        Extract features for a test. Uses cache if available.

        Args:
            test_id: 1, 2, or 3.
            include_freq: Include FFT features.
            max_files: Limit files (for quick testing).
            force: Re-extract even if cache exists.

        Returns:
            Feature DataFrame.
        """
        import pandas as pd

        features_dir = self.data_path / "features"
        cache_path = features_dir / f"test{test_id}_features.csv"

        if cache_path.exists() and not force:
            logger.info(f"Loading cached features: {cache_path}")
            return load_dataset(cache_path)

        logger.info(f"Extracting features for test {test_id}...")
        df = extract_features_for_test(
            archive_path=self.archive_path,
            test_id=test_id,
            include_freq=include_freq,
            max_files=max_files,
        )

        # Assign labels
        df = assign_labels(df, test_id=test_id)

        # Save
        save_dataset(df, cache_path)
        return df

    def train_models(
        self,
        test_id: int,
        include_freq: bool = True,
    ) -> dict:
        """
        Full training pipeline for a given test:
          1. Extract / load features
          2. Label + split
          3. Normalize
          4. Train Random Forest + Isolation Forest
          5. Evaluate + save models

        Args:
            test_id: 1, 2, or 3.
            include_freq: Use FFT features.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"=== Training pipeline for IMS Test {test_id} ===")

        # Step 1: Features
        df = self.extract_features(test_id=test_id, include_freq=include_freq)

        # Step 2: Train/test split (time-based)
        train_df, test_df = time_based_split(df)

        # Step 3: Normalize (fit on train only)
        feat_cols = get_feature_columns(train_df)
        train_df, test_df, scaler = normalize_features(
            train_df, test_df,
            feature_cols=feat_cols,
            scaler_path=self.models_path / "feature_scaler.pkl",
        )

        # Save processed splits
        processed_dir = self.data_path / "processed"
        save_dataset(train_df, processed_dir / f"test{test_id}_train.csv")
        save_dataset(test_df,  processed_dir / f"test{test_id}_test.csv")

        # Step 4a: Train Random Forest
        fault_model = FaultDiagnosisModel()
        train_info = fault_model.train(train_df, feature_cols=feat_cols)
        fault_metrics = fault_model.evaluate(test_df)
        fault_model.save(self.models_path / "fault_model.pkl")

        # Step 4b: Train Isolation Forest (on Normal data only)
        anomaly_model = AnomalyDetectionModel()
        anomaly_info = anomaly_model.train(train_df, feature_cols=feat_cols)
        anomaly_model.save(self.models_path / "anomaly_model.pkl")

        results = {
            "test_id":            test_id,
            "n_train":            len(train_df),
            "n_test":             len(test_df),
            "n_features":         len(feat_cols),
            "fault_train_info":   train_info,
            "fault_metrics":      fault_metrics,
            "anomaly_train_info": anomaly_info,
        }

        logger.info("=== Training complete ===")
        logger.info(f"Fault Model — Accuracy: {fault_metrics['accuracy']:.4f}, "
                    f"F1: {fault_metrics['f1_score']:.4f}")
        return results
