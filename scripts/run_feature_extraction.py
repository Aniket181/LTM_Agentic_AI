"""
run_feature_extraction.py — Phase 3: Feature Extraction Pipeline

PURPOSE:
    Extract vibration signal features from all raw IMS files across all
    3 tests and save model-ready feature datasets to data/features/.

    This script does NOT train any model.
    This script does NOT apply normalization/scaling.
    This script does NOT perform train/test splits.
    It only extracts features + metadata + a documented heuristic label.

METHODOLOGY:
    - Each raw file (1 second @ 20 kHz → 20,480 rows) is read once,
      features are computed, then the raw array is discarded (incremental).
    - One output row per file (snapshot).
    - Features saved to CSV immediately after each test completes.

LABELING STRATEGY (HEURISTIC — documented explicitly):
    No explicit fault onset timestamps exist in the IMS dataset.
    Label assignment uses a configurable time-position heuristic:

        label = 0 (Normal)   →   snapshots at positions [0, FAULT_START_FRACTION)
        label = 1 (Faulty)   →   snapshots at positions [FAULT_START_FRACTION, end]

    FAULT_START_FRACTION is configurable (default 0.80, i.e. last 20% = Faulty).
    This is the standard academic approach for IMS and must be treated as
    an approximation only — NOT as ground truth.

    For Test 1, BOTH Bearing 3 AND Bearing 4 fail.
    The heuristic applies uniformly across all channels (simplification).
    A future 3-class labeling (Normal / Degrading / Faulty) is possible.

METADATA COLUMNS PRESERVED (per row):
    - test_id         : int (1, 2, or 3)
    - snapshot_index  : int (chronological rank within this test, 0-based)
    - timestamp       : datetime (parsed from filename)
    - source_file     : str (original filename e.g. 2003.10.22.12.06.24)
    - n_channels      : int (8 for test1, 4 for tests 2&3)
    - label           : int (0=Normal, 1=Faulty — HEURISTIC)
    - label_name      : str ("Normal" or "Faulty")
    - label_method    : str (always "heuristic_time_fraction")
    - fault_start_idx : int (snapshot index at which Faulty period starts)

FEATURE NAMING:
    Features are named:  {channel_name}__{feature_name}
    Example:  bearing3_ch1__rms,  bearing3_ch1__kurtosis

    Test 1: 8 channels × 13 features = 104 feature columns
    Tests 2&3: 4 channels × 13 features = 52 feature columns

OUTPUT:
    data/features/test1_features.csv   (2,156 rows × 108+ cols)
    data/features/test2_features.csv   (  984 rows × 56+ cols)
    data/features/test3_features.csv   (6,324 rows × 56+ cols)
    data/features/extraction_report.txt

Usage:
    python scripts/run_feature_extraction.py
    python scripts/run_feature_extraction.py --tests 1 2 3
    python scripts/run_feature_extraction.py --tests 1 --max_files 50  # quick test
    python scripts/run_feature_extraction.py --fault_fraction 0.15     # custom heuristic
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLING_FREQ = 20_000  # Hz

# Column definitions (from loader.py — replicated here for clarity)
COLUMNS = {
    1: ["bearing1_ch1", "bearing1_ch2",
        "bearing2_ch1", "bearing2_ch2",
        "bearing3_ch1", "bearing3_ch2",
        "bearing4_ch1", "bearing4_ch2"],
    2: ["bearing1", "bearing2", "bearing3", "bearing4"],
    3: ["bearing1", "bearing2", "bearing3", "bearing4"],
}

# Test data paths
ARCHIVE = PROJECT_ROOT / "archive"
TEST_PATHS = {
    1: ARCHIVE / "1st_test" / "1st_test",
    2: ARCHIVE / "2nd_test" / "2nd_test",
    3: ARCHIVE / "3rd_test" / "4th_test" / "txt",
}

# Known fault outcomes from IMS official documentation
KNOWN_FAULTS = {
    1: {"bearing3": "outer_race_fault", "bearing4": "roller_element_fault"},
    2: {"bearing1": "outer_race_fault"},
    3: {"bearing3": "outer_race_fault"},
}

# Time-domain feature names (10 per channel)
TD_FEATURES = [
    "mean", "rms", "std", "variance",
    "kurtosis", "skewness",
    "peak", "peak_to_peak",
    "crest_factor", "shape_factor",
]

# Frequency-domain feature names (3 per channel)
FD_FEATURES = [
    "dominant_freq", "spectral_energy", "spectral_centroid",
]

ALL_FEATURES_PER_CHANNEL = TD_FEATURES + FD_FEATURES  # 13 total


# ── Feature computation ───────────────────────────────────────────────────────

def compute_time_domain(signal: np.ndarray) -> dict:
    """Compute 10 time-domain features for a 1D signal."""
    s = signal.astype(np.float64)
    abs_s = np.abs(s)
    peak = float(np.max(abs_s))
    rms  = float(np.sqrt(np.mean(s ** 2)))
    mean_abs = float(np.mean(abs_s))
    return {
        "mean":         float(np.mean(s)),
        "rms":          rms,
        "std":          float(np.std(s)),
        "variance":     float(np.var(s)),
        "kurtosis":     float(sp_kurtosis(s, fisher=True)),   # excess kurtosis
        "skewness":     float(sp_skew(s)),
        "peak":         peak,
        "peak_to_peak": float(np.max(s) - np.min(s)),
        "crest_factor": float(peak / rms)       if rms      > 1e-10 else 0.0,
        "shape_factor": float(rms / mean_abs)   if mean_abs > 1e-10 else 0.0,
    }


def compute_frequency_domain(signal: np.ndarray, fs: int = SAMPLING_FREQ) -> dict:
    """
    Compute 3 frequency-domain features via FFT.

    NOTE on dominant_freq:
        The DC bin (index 0, freq=0 Hz) represents the signal's mean offset.
        For vibration signals with a non-zero mean (common in IMS data),
        the DC bin would always dominate, making dominant_freq=0 meaningless.
        We skip index 0 and find the dominant AC frequency instead.
        This is standard practice in vibration signal processing.
    """
    s = signal.astype(np.float64)
    n = len(s)
    fft_mag = np.abs(np.fft.rfft(s))
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)
    spec_energy = float(np.sum(fft_mag ** 2))

    if spec_energy > 1e-10 and np.sum(fft_mag) > 1e-10:
        # Skip DC bin (index 0) — find dominant AC frequency
        ac_mag = fft_mag[1:]   # exclude DC
        ac_freqs = freqs[1:]
        if len(ac_mag) > 0:
            dom_freq  = float(ac_freqs[np.argmax(ac_mag)])
        else:
            dom_freq  = 0.0
        spec_cent = float(np.sum(freqs * fft_mag) / np.sum(fft_mag))
    else:
        dom_freq  = 0.0
        spec_cent = 0.0

    return {
        "dominant_freq":     dom_freq,
        "spectral_energy":   spec_energy,
        "spectral_centroid": spec_cent,
    }


def extract_features_from_file(
    filepath: Path,
    col_names: list[str],
    include_freq: bool = True,
    fs: int = SAMPLING_FREQ,
) -> tuple[dict, int, int]:
    """
    Load one IMS file and extract features for all channels.

    Returns:
        (features_dict, actual_col_count, n_nonfinite_replaced)

    The raw file data is immediately discarded after extraction.
    Non-finite values (NaN, Inf, -Inf) are replaced with 0.0 and counted.
    A feature value of 0.0 that was already legitimately 0 is NOT counted.
    """
    data = pd.read_csv(filepath, sep=r"\s+", header=None, dtype=np.float32).values
    actual_cols = data.shape[1]

    # Guard: if file has fewer columns than expected, use what's present
    cols_to_use = col_names[:actual_cols]

    row = {}
    for i, col in enumerate(cols_to_use):
        sig = data[:, i]
        td = compute_time_domain(sig)
        for feat, val in td.items():
            row[f"{col}__{feat}"] = val
        if include_freq:
            fd = compute_frequency_domain(sig, fs=fs)
            for feat, val in fd.items():
                row[f"{col}__{feat}"] = val

    # Replace only genuinely non-finite values (NaN, +Inf, -Inf)
    # and count them accurately.
    n_replaced = 0
    for k, v in row.items():
        if not np.isfinite(v):   # True ONLY for NaN, +inf, -inf
            row[k] = 0.0
            n_replaced += 1

    return row, actual_cols, n_replaced


def parse_timestamp(name: str):
    """Parse IMS filename to datetime, return None if it doesn't match."""
    try:
        return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return None


# ── Labeling ──────────────────────────────────────────────────────────────────

def assign_heuristic_labels(n_snapshots: int, fault_fraction: float) -> tuple[list, int]:
    """
    Assign binary heuristic labels based on snapshot position.

    HEURISTIC (not ground truth):
        snapshots [0, cutoff_idx)       → 0 = Normal
        snapshots [cutoff_idx, n)       → 1 = Faulty

    Args:
        n_snapshots:    Total number of chronologically ordered snapshots.
        fault_fraction: Fraction of tail snapshots to label Faulty.

    Returns:
        (labels: list of int, fault_start_idx: int)
    """
    cutoff = int(n_snapshots * (1.0 - fault_fraction))
    labels = [0] * cutoff + [1] * (n_snapshots - cutoff)
    return labels, cutoff


# ── Per-test extraction ───────────────────────────────────────────────────────

def extract_test(
    test_id: int,
    fault_fraction: float,
    include_freq: bool = True,
    max_files: int = None,
    report_lines: list = None,
) -> pd.DataFrame:
    """
    Extract features for all files in one IMS test.

    Files are processed one at a time (incremental) — raw data is never
    accumulated in memory. Only the feature vector per file is retained.

    Args:
        test_id:        1, 2, or 3.
        fault_fraction: Fraction of tail snapshots to label Faulty.
        include_freq:   Include FFT-based frequency features.
        max_files:      Limit number of files (None = all).
        report_lines:   Optional list to append report text to.

    Returns:
        DataFrame with metadata + features + labels.
    """
    test_path = TEST_PATHS[test_id]
    col_names = COLUMNS[test_id]

    if not test_path.exists():
        raise FileNotFoundError(f"Test {test_id} directory not found: {test_path}")

    # Collect and sort files chronologically
    all_files = sorted(
        [f for f in test_path.iterdir()
         if f.is_file() and parse_timestamp(f.name) is not None],
        key=lambda f: f.name,
    )

    if max_files:
        all_files = all_files[:max_files]

    n_files = len(all_files)
    print(f"\n{'='*65}")
    print(f"  TEST {test_id}: {n_files} files | path: {test_path}")
    print(f"  Columns:  {col_names}")
    print(f"  Features: {len(col_names)} channels × {len(ALL_FEATURES_PER_CHANNEL)} "
          f"= {len(col_names) * len(ALL_FEATURES_PER_CHANNEL)} feature columns")
    print(f"  Labeling: heuristic — last {fault_fraction*100:.0f}% = Faulty")
    print(f"{'='*65}")

    # ── Main extraction loop (incremental — one file at a time) ──────────────
    records      = []
    n_skipped    = 0
    n_bad_values = 0
    t_start      = time.perf_counter()

    for i, fpath in enumerate(all_files):
        if i % 200 == 0 and i > 0:
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed
            remaining = (n_files - i) / rate
            print(f"  [{i:>5}/{n_files}]  elapsed={elapsed:.0f}s  "
                  f"rate={rate:.1f} files/s  eta={remaining:.0f}s")

        ts = parse_timestamp(fpath.name)
        if ts is None:
            n_skipped += 1
            continue

        try:
            features, actual_cols, n_replaced = extract_features_from_file(
                fpath, col_names, include_freq=include_freq
            )
        except Exception as e:
            print(f"  WARNING: Skipping {fpath.name}: {e}")
            n_skipped += 1
            continue

        # Count only actual non-finite replacements (NaN/Inf → 0.0)
        # NOT legitimate 0.0 values (e.g. near-zero mean, zero-energy channel)
        n_bad_values += n_replaced

        # Metadata (set later after all records are in order)
        record = {
            "test_id":     test_id,
            "source_file": fpath.name,
            "timestamp":   ts,
            "n_channels":  actual_cols,
        }
        record.update(features)
        records.append(record)

    # Sort chronologically (should already be sorted, but ensure consistency)
    records.sort(key=lambda r: r["timestamp"])

    # Add snapshot_index (chronological rank within this test)
    for idx, rec in enumerate(records):
        rec["snapshot_index"] = idx

    # Assign heuristic labels
    n_valid = len(records)
    labels, fault_start_idx = assign_heuristic_labels(n_valid, fault_fraction)

    for rec, lbl in zip(records, labels):
        rec["label"]          = lbl
        rec["label_name"]     = "Normal" if lbl == 0 else "Faulty"
        rec["label_method"]   = "heuristic_time_fraction"
        rec["fault_start_idx"] = fault_start_idx

    df = pd.DataFrame(records)

    # Verify chronological ordering
    ts_sorted = df["timestamp"].is_monotonic_increasing
    elapsed = time.perf_counter() - t_start

    # Print summary
    n_normal = (df["label"] == 0).sum()
    n_faulty = (df["label"] == 1).sum()
    feat_cols = [c for c in df.columns
                 if c not in {"test_id","source_file","timestamp","n_channels",
                               "snapshot_index","label","label_name",
                               "label_method","fault_start_idx"}]

    print(f"\n  ✓ Extracted {n_valid} snapshots in {elapsed:.1f}s "
          f"({n_valid/elapsed:.1f} files/s)")
    print(f"  ✓ Skipped:  {n_skipped} files")
    print(f"  ✓ Features: {len(feat_cols)} columns per row")
    print(f"  ✓ Labels:   {n_normal} Normal | {n_faulty} Faulty "
          f"(cutoff @ idx {fault_start_idx})")
    print(f"  ✓ Chrono:   timestamp monotonic = {ts_sorted}")
    print(f"  ✓ Bad vals: {n_bad_values} (replaced with 0.0)")
    print(f"  ✓ NaN:      {df[feat_cols].isnull().sum().sum()}")
    print(f"  ✓ Inf:      {np.isinf(df[feat_cols].values).sum()}")

    if report_lines is not None:
        report_lines.extend([
            f"\nTEST {test_id}",
            f"  Path:            {test_path}",
            f"  Total files:     {n_files}",
            f"  Valid snapshots: {n_valid}",
            f"  Skipped files:   {n_skipped}",
            f"  Channels:        {actual_cols} ({col_names})",
            f"  Features/row:    {len(feat_cols)}",
            f"  Label Normal:    {n_normal}",
            f"  Label Faulty:    {n_faulty}",
            f"  Fault cutoff:    index {fault_start_idx} "
            f"(heuristic: last {fault_fraction*100:.0f}%)",
            f"  Chronological:   {ts_sorted}",
            f"  NaN values:      {df[feat_cols].isnull().sum().sum()}",
            f"  Inf values:      {np.isinf(df[feat_cols].values).sum()}",
            f"  First file:      {df['source_file'].iloc[0]}",
            f"  Last file:       {df['source_file'].iloc[-1]}",
            f"  First timestamp: {df['timestamp'].iloc[0]}",
            f"  Last timestamp:  {df['timestamp'].iloc[-1]}",
            f"  Duration (days): {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days}",
            f"  Elapsed:         {elapsed:.1f}s",
        ])

    return df


# ── Validation report ─────────────────────────────────────────────────────────

def print_feature_summary(df: pd.DataFrame, test_id: int, n_sample: int = 5):
    """Print feature name list, types, distribution summary."""
    feat_cols = [c for c in df.columns
                 if c not in {"test_id","source_file","timestamp","n_channels",
                               "snapshot_index","label","label_name",
                               "label_method","fault_start_idx"}]
    print(f"\n  Feature columns ({len(feat_cols)} total):")
    for i, name in enumerate(feat_cols):
        col = df[name]
        print(f"    [{i+1:>3}] {name:<40}  "
              f"min={col.min():.4f}  max={col.max():.4f}  "
              f"mean={col.mean():.4f}  std={col.std():.4f}")

    print(f"\n  Sample rows (first {n_sample}):")
    meta = ["snapshot_index", "timestamp", "label_name"]
    sample_cols = [c for c in meta if c in df.columns] + feat_cols[:4]
    print(df[sample_cols].head(n_sample).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Extract features from IMS bearing dataset."
    )
    parser.add_argument("--tests", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3], help="Which tests to process")
    parser.add_argument("--fault_fraction", type=float, default=0.20,
                        help="Fraction of tail snapshots to label Faulty (default 0.20)")
    parser.add_argument("--no_freq", action="store_true",
                        help="Exclude FFT frequency-domain features")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Limit files per test (for quick testing only)")
    args = parser.parse_args()

    include_freq = not args.no_freq
    features_per_ch = 10 + (3 if include_freq else 0)

    print("\n" + "=" * 65)
    print("  PHASE 3 — FEATURE EXTRACTION")
    print("=" * 65)
    print(f"  Tests to process:   {args.tests}")
    print(f"  Fault fraction:     {args.fault_fraction} "
          f"(last {args.fault_fraction*100:.0f}% = Faulty HEURISTIC)")
    print(f"  Include FFT:        {include_freq}")
    print(f"  Max files/test:     {args.max_files or 'ALL'}")
    print(f"  Features/channel:   {features_per_ch}")
    print(f"\n  LABELING NOTE:")
    print(f"    No explicit fault onset timestamps exist in IMS dataset.")
    print(f"    Labels are HEURISTIC only — last {args.fault_fraction*100:.0f}% of each")
    print(f"    test's snapshot sequence is labeled Faulty.")
    print(f"    Do NOT treat these as ground-truth labels.")
    print(f"    Known fault outcomes from IMS documentation:")
    for tid, faults in KNOWN_FAULTS.items():
        if tid in args.tests:
            print(f"      Test {tid}: {faults}")

    output_dir = PROJECT_ROOT / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "PHASE 3 EXTRACTION REPORT",
        "=" * 65,
        f"Generated:      {datetime.now().isoformat()}",
        f"Fault fraction: {args.fault_fraction}  (HEURISTIC — NOT ground truth)",
        f"Include FFT:    {include_freq}",
        f"Max files:      {args.max_files or 'ALL'}",
        "",
        "LABELING STRATEGY:",
        "  Binary heuristic: last FAULT_FRACTION of snapshot timeline = Faulty",
        "  No explicit fault onset. Standard academic IMS labeling approach.",
        "  Known fault outcomes from IMS documentation used for reference only.",
        "",
    ]

    all_feature_cols = {}
    t_total = time.perf_counter()

    for test_id in args.tests:
        print(f"\n{'─'*65}")
        print(f"  Processing Test {test_id}...")
        print(f"{'─'*65}")

        df = extract_test(
            test_id=test_id,
            fault_fraction=args.fault_fraction,
            include_freq=include_freq,
            max_files=args.max_files,
            report_lines=report_lines,
        )

        # Capture feature columns for report
        feat_cols = [c for c in df.columns
                     if c not in {"test_id","source_file","timestamp","n_channels",
                                   "snapshot_index","label","label_name",
                                   "label_method","fault_start_idx"}]
        all_feature_cols[test_id] = feat_cols

        # Print feature summary
        print_feature_summary(df, test_id, n_sample=3)

        # Save to CSV
        out_path = output_dir / f"test{test_id}_features.csv"
        df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024**2
        print(f"\n  ✓ Saved: {out_path}")
        print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"    Size:  {size_mb:.1f} MB")

        report_lines.extend([
            f"  Output file:     {out_path}",
            f"  Output shape:    {df.shape[0]} rows × {df.shape[1]} cols",
            f"  Output size:     {size_mb:.1f} MB",
        ])

    # ── Final cross-test report ───────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total

    print(f"\n{'='*65}")
    print(f"  PHASE 3 COMPLETE")
    print(f"{'='*65}")
    print(f"  Total elapsed:   {elapsed_total:.1f}s")

    print(f"\n  FEATURE STRUCTURE PER TEST:")
    for tid in args.tests:
        n_channels = len(COLUMNS[tid])
        n_feat     = len(all_feature_cols.get(tid, []))
        print(f"    Test {tid}: {n_channels} channels × {features_per_ch} "
              f"features/channel = {n_feat} feature columns")

    print(f"\n  FEATURE NAMES (Test 1, first 15):")
    if 1 in all_feature_cols:
        for name in all_feature_cols[1][:15]:
            print(f"    {name}")
        if len(all_feature_cols[1]) > 15:
            print(f"    ... ({len(all_feature_cols[1]) - 15} more)")

    print(f"\n  FEATURE NAMES (Test 2, all — 4-channel):")
    if 2 in all_feature_cols:
        for name in all_feature_cols[2]:
            print(f"    {name}")

    print(f"\n  CHRONOLOGICAL SPLIT INFO (for Phase 4):")
    print(f"    Split strategy:   time-based rank split (NOT random)")
    print(f"    Train fraction:   0.70 (first 70% of snapshots)")
    print(f"    Test  fraction:   0.30 (last 30% of snapshots)")
    print(f"    Scaler fitting:   MUST be done on train partition ONLY")
    print(f"    NOTE:             Do not apply scaler until Phase 4")

    # Save report
    report_lines.extend([
        "",
        "PHASE 3 COMPLETE",
        f"Total elapsed: {elapsed_total:.1f}s",
        "",
        "CHRONOLOGICAL SPLIT (Phase 4 guidance):",
        "  Strategy:   time-based rank split",
        "  Train:      first 70% of snapshots per test",
        "  Test:       last 30% of snapshots per test",
        "  Scaler:     fit on train only — NOT applied yet",
    ])

    report_path = output_dir / "extraction_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  ✓ Report saved: {report_path}")

    print(f"\n  OUTPUT FILES:")
    for test_id in args.tests:
        p = output_dir / f"test{test_id}_features.csv"
        if p.exists():
            sz = p.stat().st_size / 1024**2
            print(f"    {p}  ({sz:.1f} MB)")
    print()
    print("  STOP — Phase 3 complete. Do NOT proceed to Phase 4 yet.")
    print("  Review this output, check the report, then approve Phase 4.\n")


if __name__ == "__main__":
    main()
