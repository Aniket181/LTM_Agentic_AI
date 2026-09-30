"""
validate_preprocessing.py — Phase 4 Validation

Loads the preprocessed outputs and runs all 14 checks.
Produces a structured pass/fail report.

Usage:
    python scripts/validate_preprocessing.py --tests 1 2 3
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from industrial_health.data.preprocessing import LABEL_DISCLAIMER, METADATA_COLS

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_DIR    = PROJECT_ROOT / "models" / "scaler"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

TRAIN_FRACTION = 0.70
TOL_SPLIT      = 0.03   # allow ±3% deviation from target split
TOL_MEAN       = 0.10   # scaled train mean must be within ±0.10 of 0
TOL_STD        = 0.15   # scaled train std  must be within 1.0 ± 0.15


def check(condition: bool, label: str, detail: str = "") -> bool:
    mark = "✓" if condition else "✗"
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return condition


def validate_test(test_id: int) -> bool:
    out_dir = PROCESSED_DIR / f"test{test_id}"
    scaler_path = SCALER_DIR / f"test{test_id}_scaler.pkl"
    feat_csv    = FEATURES_DIR / f"test{test_id}_features.csv"

    print(f"\n{'─'*60}")
    print(f"  TEST {test_id}")
    print(f"{'─'*60}")

    all_ok = True

    # ── File existence ────────────────────────────────────────────────────────
    required_files = [
        out_dir / "X_train.npy",    out_dir / "X_test.npy",
        out_dir / "y_train.npy",    out_dir / "y_test.npy",
        out_dir / "meta_train.csv", out_dir / "meta_test.csv",
        out_dir / "feature_names.txt", out_dir / "split_config.json",
        scaler_path,
    ]
    all_exist = all(f.exists() for f in required_files)
    missing   = [f.name for f in required_files if not f.exists()]
    all_ok &= check(all_exist, "All output files exist",
                    f"missing: {missing}" if missing else "")
    if not all_exist:
        print("  Cannot continue — output files missing.")
        return False

    # ── Load ──────────────────────────────────────────────────────────────────
    X_train = np.load(out_dir / "X_train.npy")
    X_test  = np.load(out_dir / "X_test.npy")
    y_train = np.load(out_dir / "y_train.npy")
    y_test  = np.load(out_dir / "y_test.npy")
    meta_train = pd.read_csv(out_dir / "meta_train.csv", parse_dates=["timestamp"])
    meta_test  = pd.read_csv(out_dir / "meta_test.csv",  parse_dates=["timestamp"])
    feat_names = (out_dir / "feature_names.txt").read_text().splitlines()
    config     = json.loads((out_dir / "split_config.json").read_text())
    scaler: StandardScaler = joblib.load(scaler_path)

    n_total = X_train.shape[0] + X_test.shape[0]
    n_feat  = X_train.shape[1]

    # ── 1. Train/test row counts ≈ 70/30 ─────────────────────────────────────
    actual_tr = X_train.shape[0] / n_total
    all_ok &= check(
        abs(actual_tr - TRAIN_FRACTION) <= TOL_SPLIT,
        f"Train fraction ≈ {TRAIN_FRACTION:.0%}",
        f"actual={actual_tr:.3f}  (±{TOL_SPLIT:.2f} tolerance)",
    )

    # ── 2. No source-file overlap ─────────────────────────────────────────────
    if "source_file" in meta_train.columns and "source_file" in meta_test.columns:
        overlap = set(meta_train["source_file"]) & set(meta_test["source_file"])
        all_ok &= check(len(overlap) == 0, "No source-file overlap between train/test",
                        f"overlapping: {overlap}" if overlap else "")
    else:
        check(False, "source_file column present in metadata", "MISSING")
        all_ok = False

    # ── 3. No snapshot-index overlap ─────────────────────────────────────────
    if "snapshot_index" in meta_train.columns:
        idx_overlap = set(meta_train["snapshot_index"]) & set(meta_test["snapshot_index"])
        all_ok &= check(len(idx_overlap) == 0, "No snapshot_index overlap",
                        f"overlapping: {len(idx_overlap)} indices" if idx_overlap else "")
    else:
        check(False, "snapshot_index present in metadata", "MISSING")
        all_ok = False

    # ── 4. Training timestamps precede test timestamps ────────────────────────
    if "timestamp" in meta_train.columns:
        last_train_ts  = meta_train["timestamp"].max()
        first_test_ts  = meta_test["timestamp"].min()
        ts_ok = last_train_ts < first_test_ts
        all_ok &= check(ts_ok, "Training timestamps precede test timestamps",
                        f"last_train={last_train_ts}  first_test={first_test_ts}")
    else:
        check(False, "timestamp column present in metadata", "MISSING")
        all_ok = False

    # ── 5. No NaN in X_train or X_test ───────────────────────────────────────
    nan_tr = np.isnan(X_train).sum()
    nan_te = np.isnan(X_test).sum()
    all_ok &= check(nan_tr == 0 and nan_te == 0, "No NaN in X_train or X_test",
                    f"train_nan={nan_tr}  test_nan={nan_te}")

    # ── 6. No Inf in X_train or X_test ───────────────────────────────────────
    inf_tr = np.isinf(X_train).sum()
    inf_te = np.isinf(X_test).sum()
    all_ok &= check(inf_tr == 0 and inf_te == 0, "No Inf in X_train or X_test",
                    f"train_inf={inf_tr}  test_inf={inf_te}")

    # ── 7. Feature dimensions correct ────────────────────────────────────────
    feats_match = (X_train.shape[1] == X_test.shape[1] == len(feat_names))
    all_ok &= check(feats_match, "Feature dimension consistent across X_train/X_test/feature_names",
                    f"X_train.shape[1]={X_train.shape[1]}  "
                    f"X_test.shape[1]={X_test.shape[1]}  "
                    f"len(feat_names)={len(feat_names)}")

    # ── 8. Metadata row count matches X rows ─────────────────────────────────
    meta_ok = (len(meta_train) == X_train.shape[0]) and (len(meta_test) == X_test.shape[0])
    all_ok &= check(meta_ok, "Metadata row counts match X array rows",
                    f"meta_train={len(meta_train)} vs X_train={X_train.shape[0]}  "
                    f"meta_test={len(meta_test)} vs X_test={X_test.shape[0]}")

    # ── 9. Label metadata preserved ──────────────────────────────────────────
    label_meta_ok = ("label" in meta_train.columns and
                     "label_method" in meta_train.columns and
                     "fault_start_idx" in meta_train.columns)
    all_ok &= check(label_meta_ok, "label, label_method, fault_start_idx in metadata")
    if "label_method" in meta_train.columns:
        method = meta_train["label_method"].iloc[0]
        all_ok &= check(method == "heuristic_time_fraction",
                        "label_method = 'heuristic_time_fraction'",
                        f"found: '{method}'")

    # ── 10. Scaler fitted only on train ──────────────────────────────────────
    # Verify scaler.mean_ was computed from X_train shape
    scaler_fitted = hasattr(scaler, "mean_") and scaler.mean_.shape[0] == n_feat
    all_ok &= check(scaler_fitted, "Scaler is fitted and has correct feature count",
                    f"scaler.mean_.shape={scaler.mean_.shape}  n_feat={n_feat}")

    # ── 11. Scaled X_train approx mean=0, std=1 ──────────────────────────────
    tr_means = X_train.mean(axis=0)
    tr_stds  = X_train.std(axis=0)
    mean_ok  = (np.abs(tr_means) < TOL_MEAN).all()
    std_ok   = (np.abs(tr_stds - 1.0) < TOL_STD).all()
    all_ok &= check(mean_ok,
                    f"X_train column means ≈ 0 (all |mean| < {TOL_MEAN})",
                    f"max|mean|={np.abs(tr_means).max():.4f}")
    all_ok &= check(std_ok,
                    f"X_train column stds ≈ 1 (all |std-1| < {TOL_STD})",
                    f"max|std-1|={np.abs(tr_stds - 1.0).max():.4f}")

    # ── 12. X_test uses same scaler ──────────────────────────────────────────
    # Re-transform X_test_raw using loaded scaler and compare
    # Load raw test features from Phase 3 CSV
    raw_df = pd.read_csv(feat_csv, parse_dates=["timestamp"])
    raw_df = raw_df.sort_values("snapshot_index").reset_index(drop=True)
    n_train_rows = X_train.shape[0]
    feat_names_ordered = (out_dir / "feature_names.txt").read_text().splitlines()
    X_test_raw = raw_df.iloc[n_train_rows:][feat_names_ordered].values.astype(np.float64)
    X_test_recomputed = scaler.transform(X_test_raw)
    same_transform = np.allclose(X_test, X_test_recomputed, atol=1e-6)
    all_ok &= check(same_transform,
                    "X_test matches re-transformation using saved scaler",
                    "(verifies scaler was saved and loaded correctly)")

    # ── 13. Label distributions reported (informational) ─────────────────────
    n0_tr = int((y_train == 0).sum())
    n1_tr = int((y_train == 1).sum())
    n0_te = int((y_test  == 0).sum())
    n1_te = int((y_test  == 1).sum())
    print(f"  [i] Label distribution:")
    print(f"      Train: Normal={n0_tr}  Faulty={n1_tr}  "
          f"({n1_tr/(n0_tr+n1_tr)*100:.1f}% faulty in train)")
    print(f"      Test:  Normal={n0_te}  Faulty={n1_te}  "
          f"({n1_te/(n0_te+n1_te)*100:.1f}% faulty in test)")

    # ── 14. Heuristic label disclaimer present in config ─────────────────────
    disclaimer_ok = "label_disclaimer" in config and len(config["label_disclaimer"]) > 10
    all_ok &= check(disclaimer_ok,
                    "Label disclaimer present in split_config.json")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Validation — check preprocessing outputs"
    )
    parser.add_argument(
        "--tests", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3]
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PHASE 4 VALIDATION")
    print("=" * 60)
    print(f"\n  LABEL DISCLAIMER:")
    print(f"    {LABEL_DISCLAIMER}")

    test_results = {}
    for tid in args.tests:
        test_results[tid] = validate_test(tid)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 4 VALIDATION")
    print(f"{'='*60}")
    for tid, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"\n  TEST {tid}: {status}")

    overall = all(test_results.values())
    print(f"\n{'='*60}")
    if overall:
        print("  VALIDATION PASSED")
    else:
        print("  VALIDATION FAILED — review ✗ items above")
    print(f"{'='*60}\n")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
