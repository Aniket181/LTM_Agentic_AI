"""
run_preprocessing.py — Phase 4: Preprocessing Pipeline CLI

Loads Phase 3 feature CSVs, applies chronological 70/30 split,
fits StandardScaler on training partition only, and saves all outputs.

Usage:
    python scripts/run_preprocessing.py --tests 1 2 3
    python scripts/run_preprocessing.py --tests 1
    python scripts/run_preprocessing.py --train_fraction 0.70

Output:
    data/processed/test{N}/
        X_train.npy, X_test.npy
        y_train.npy, y_test.npy
        meta_train.csv, meta_test.csv
        feature_names.txt, split_config.json
    models/scaler/
        test{N}_scaler.pkl

DO NOT run Phase 5 (model training) until validation passes.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_health.data.preprocessing import (
    preprocess_test,
    save_preprocessed,
    TRAIN_FRACTION,
    LABEL_DISCLAIMER,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Preprocessing & chronological data splitting"
    )
    parser.add_argument(
        "--tests", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3],
        help="Which test IDs to preprocess (default: all three)"
    )
    parser.add_argument(
        "--train_fraction", type=float, default=TRAIN_FRACTION,
        help=f"Train partition fraction (default: {TRAIN_FRACTION})"
    )
    args = parser.parse_args()

    features_dir  = PROJECT_ROOT / "data" / "features"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    scaler_dir    = PROJECT_ROOT / "models" / "scaler"

    print("\n" + "=" * 65)
    print("  PHASE 4 — PREPROCESSING & CHRONOLOGICAL DATA SPLITTING")
    print("=" * 65)
    print(f"  Tests:          {args.tests}")
    print(f"  Train fraction: {args.train_fraction:.2f}")
    print(f"  Test  fraction: {1 - args.train_fraction:.2f}")
    print(f"  Split method:   chronological (by snapshot_index rank)")
    print(f"  Scaler:         StandardScaler, fitted on train only")
    print(f"\n  LABELING DISCLAIMER:")
    print(f"    {LABEL_DISCLAIMER}")
    print()

    results = []

    for test_id in args.tests:
        csv_path = features_dir / f"test{test_id}_features.csv"

        print(f"{'─'*65}")
        print(f"  Processing Test {test_id}  ←  {csv_path.name}")
        print(f"{'─'*65}")

        if not csv_path.exists():
            print(f"  ERROR: {csv_path} not found. Run Phase 3 first.")
            sys.exit(1)

        split = preprocess_test(
            feature_csv_path=csv_path,
            test_id=test_id,
            train_fraction=args.train_fraction,
        )

        saved = save_preprocessed(split, processed_dir, scaler_dir)

        # ── Print per-test summary ────────────────────────────────────────────
        n_total = split.n_train + split.n_test
        actual_train_pct = split.n_train / n_total * 100
        actual_test_pct  = split.n_test  / n_total * 100

        y_n_train = (split.y_train == 0).sum()
        y_f_train = (split.y_train == 1).sum()
        y_n_test  = (split.y_test  == 0).sum()
        y_f_test  = (split.y_test  == 1).sum()

        scaler_mean_check = split.scaler.mean_
        scaler_std_check  = split.scaler.scale_

        print(f"  Rows total:     {n_total}")
        print(f"  Train rows:     {split.n_train}  ({actual_train_pct:.1f}%)")
        print(f"  Test  rows:     {split.n_test}   ({actual_test_pct:.1f}%)")
        print(f"  Split at index: {split.split_idx}")
        print(f"  Features:       {split.n_features}")
        print(f"  X_train shape:  {split.X_train.shape}")
        print(f"  X_test  shape:  {split.X_test.shape}")
        print(f"  y_train:        Normal={y_n_train}  Faulty={y_f_train}")
        print(f"  y_test:         Normal={y_n_test}   Faulty={y_f_test}")
        print(f"  Scaler means:   min={scaler_mean_check.min():.4f}  "
              f"max={scaler_mean_check.max():.4f}")
        print(f"  Scaler stds:    min={scaler_std_check.min():.4f}  "
              f"max={scaler_std_check.max():.4f}")

        print(f"\n  Saved files:")
        for name, path in saved.items():
            sz = path.stat().st_size
            print(f"    {name:<16}  {path.relative_to(PROJECT_ROOT)}  "
                  f"({sz:,} bytes)")

        results.append({
            "test_id":  test_id,
            "n_train":  split.n_train,
            "n_test":   split.n_test,
            "n_feat":   split.n_features,
            "y_n_tr":   int(y_n_train),
            "y_f_tr":   int(y_f_train),
            "y_n_te":   int(y_n_test),
            "y_f_te":   int(y_f_test),
        })
        print()

    # ── Final cross-test summary ──────────────────────────────────────────────
    print("=" * 65)
    print("  PHASE 4 COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"  {'Test':<8} {'Train':>7} {'Test':>7} {'Features':>10} "
          f"{'y0_tr':>7} {'y1_tr':>7} {'y0_te':>7} {'y1_te':>7}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for r in results:
        print(f"  Test {r['test_id']:<3}  {r['n_train']:>7}  {r['n_test']:>7}  "
              f"{r['n_feat']:>10}  {r['y_n_tr']:>7}  {r['y_f_tr']:>7}  "
              f"{r['y_n_te']:>7}  {r['y_f_te']:>7}")

    print(f"\n  Output root:  data/processed/")
    print(f"  Scaler root:  models/scaler/")
    print(f"\n  LABEL DISCLAIMER:")
    print(f"    {LABEL_DISCLAIMER}")
    print(f"\n  NEXT STEP: run validate_preprocessing.py, then await Phase 5 approval.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
