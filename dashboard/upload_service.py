"""
dashboard/upload_service.py
Handles user CSV data upload, validation, and feature extraction.
Integrates with the existing pipeline by creating temporary feature CSVs.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from industrial_health.features.extractor import extract_features_from_array
from industrial_health.agents import OrchestratorAgent
from dashboard.config import MODELS_DIR, SCALERS_DIR

# The pipeline expects 8 channels
EXPECTED_CHANNELS = 8
# Approximate number of rows for a single snapshot (IMS dataset)
EXPECTED_ROWS = 20_480

def validate_uploaded_csv(file_obj: io.BytesIO) -> dict[str, Any]:
    """
    Validate the uploaded CSV file.
    Must have exactly 8 numeric columns.
    """
    try:
        # Read the file
        df = pd.read_csv(file_obj)
        
        # Validate columns
        if len(df.columns) != EXPECTED_CHANNELS:
            return {
                "valid": False,
                "error": f"Expected {EXPECTED_CHANNELS} numeric vibration channels but found {len(df.columns)}."
            }
            
        # Validate numeric
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
            return {
                "valid": False,
                "error": "All columns must contain numeric sensor values."
            }
            
        # Validate missing/NaN
        missing = df.isna().sum().sum()
        if missing > 0:
            return {
                "valid": False,
                "error": f"Found {missing} missing/NaN values. Please clean the data."
            }
            
        # Validate sufficient rows (at least 1000 for meaningful FFT)
        if len(df) < 1000:
            return {
                "valid": False,
                "error": f"Insufficient rows. Expected ~{EXPECTED_ROWS} samples per snapshot, got {len(df)}."
            }
            
        return {
            "valid": True,
            "rows": len(df),
            "channels": len(df.columns),
            "missing": missing,
            "df": df
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to parse CSV file: {str(e)}"
        }

def process_upload_and_run(df_raw: pd.DataFrame, file_name: str) -> dict[str, Any]:
    """
    Extracts features from the raw uploaded data,
    creates a temporary feature directory,
    and runs the OrchestratorAgent using the existing models.
    """
    try:
        # Step 1: Feature Extraction
        # Generate the standard feature column names expected by the models
        # test1_features.csv uses: bearing1_ch1, bearing1_ch2, ..., bearing4_ch2
        channel_names = [
            "bearing1_ch1", "bearing1_ch2",
            "bearing2_ch1", "bearing2_ch2",
            "bearing3_ch1", "bearing3_ch2",
            "bearing4_ch1", "bearing4_ch2"
        ]
        
        data_array = df_raw.values
        features = extract_features_from_array(
            data=data_array,
            column_names=channel_names,
            include_freq=True
        )
        
        # Step 2: Create a mock single-snapshot DataFrame matching Phase 3 schema
        row_data = {
            "test_id": 99,
            "source_file": file_name,
            "timestamp": pd.Timestamp.now(),
            "n_channels": EXPECTED_CHANNELS,
        }
        
        # Flatten the features dictionary into the row
        for feat_key, feat_val in features.items():
            row_data[feat_key] = feat_val
                
        # Add required suffix columns
        row_data["snapshot_index"] = 0
        row_data["label"] = "Unknown"
        row_data["label_name"] = "Unknown"
        row_data["label_method"] = "Upload"
        row_data["fault_start_idx"] = -1
        
        df_features = pd.DataFrame([row_data])
        
        # Step 3: Write to a temporary directory to feed the existing pipeline
        # The Orchestrator Agent reads test_id, so we use test_id=1 to load the test1 models,
        # but we place a fake 'test1_features.csv' in the temporary directory!
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv = tmp_path / "test1_features.csv"
            
            # To trick the pipeline into processing our row using test 1 models,
            # we write it as test1_features.csv, but keep the test_id column as 1
            df_features["test_id"] = 1
            df_features.to_csv(feature_csv, index=False)
            
            # Step 4: Run the orchestrator
            # Note: OrchestratorAgent uses models_dir and scalers_dir which we point to the real ones.
            orch = OrchestratorAgent(
                models_dir=MODELS_DIR,
                features_dir=tmp_path,
                scalers_dir=SCALERS_DIR,
                test_id=1,  # Uses Test 1 models
            )
            
            state = orch.run()
            
            if state.status == "FAILED":
                return {
                    "error": "Agentic pipeline failed during analysis.",
                    "details": state.errors
                }
                
            result = orch.build_final_result(state)
            
            # Add timeseries records directly since we don't cache HealthMonitor for uploads
            from industrial_health.health.health_monitor import HealthMonitor
            
            paths = {
                "rf_model": MODELS_DIR / "random_forest_test1.joblib",
                "rf_scaler": SCALERS_DIR / "phase5_test1_scaler.pkl",
                "if_model": MODELS_DIR / "isolation_forest_test1.joblib",
                "if_scaler": SCALERS_DIR / "phase6_test1_scaler.pkl",
            }
            
            monitor = HealthMonitor(
                rf_model_path=paths["rf_model"],
                rf_scaler_path=paths["rf_scaler"],
                if_model_path=paths["if_model"],
                if_scaler_path=paths["if_scaler"]
            )
            
            ts_df = monitor.score_dataset(df_features)
            
            result["timeseries"] = {
                "records": ts_df.to_dict(orient="records"),
                "fault_start": -1,
                "n_snapshots": 1
            }
            
            return result
            
    except Exception as e:
        import traceback
        return {
            "error": f"Upload processing failed: {str(e)}",
            "details": [traceback.format_exc()]
        }
