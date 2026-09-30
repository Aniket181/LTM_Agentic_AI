"""
Phase 3 Validation Script — verify the extracted feature CSVs.
Run: python scripts/validate_features.py
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

FEATURES_DIR = PROJECT_ROOT / "data" / "features"

METADATA_COLS = {
    "test_id", "snapshot_index", "timestamp", "source_file",
    "n_channels", "label", "label_name", "label_method", "fault_start_idx",
}

EXPECTED = {
    1: {"n_rows": 2156, "n_feat_cols": 104, "n_channels": 8, "n_total_cols": 113},
    2: {"n_rows": 984,  "n_feat_cols": 52,  "n_channels": 4, "n_total_cols": 61},
    3: {"n_rows": 6324, "n_feat_cols": 52,  "n_channels": 4, "n_total_cols": 61},
}

print("=" * 65)
print("  PHASE 3 VALIDATION REPORT")
print("=" * 65)

all_ok = True

for test_id in [1, 2, 3]:
    path = FEATURES_DIR / f"test{test_id}_features.csv"
    exp  = EXPECTED[test_id]

    print(f"\n{'─'*65}")
    print(f"  TEST {test_id}: {path.name}")
    print(f"{'─'*65}")

    if not path.exists():
        print("  ERROR: file does not exist!")
        all_ok = False
        continue

    df = pd.read_csv(path, parse_dates=["timestamp"])
    feat_cols = [c for c in df.columns if c not in METADATA_COLS]

    # ── Shape ────────────────────────────────────────────────────────────────
    print(f"  Rows:            {df.shape[0]}  (expected {exp['n_rows']})")
    print(f"  Total cols:      {df.shape[1]}  (expected {exp['n_total_cols']})")
    print(f"  Feature cols:    {len(feat_cols)}  (expected {exp['n_feat_cols']})")

    if df.shape[0] != exp["n_rows"]:
        print(f"  ⚠ ROW COUNT MISMATCH")
        all_ok = False
    if len(feat_cols) != exp["n_feat_cols"]:
        print(f"  ⚠ FEATURE COUNT MISMATCH")
        all_ok = False

    # ── Metadata presence ────────────────────────────────────────────────────
    missing_meta = METADATA_COLS - set(df.columns)
    print(f"  Missing metadata cols: {missing_meta or 'none'}")

    # ── Chronological ordering ───────────────────────────────────────────────
    is_mono = df["timestamp"].is_monotonic_increasing
    print(f"  Timestamp monotonic:   {is_mono}")
    if not is_mono:
        print("  ⚠ TIMESTAMPS NOT SORTED")
        all_ok = False

    # ── snapshot_index continuity ────────────────────────────────────────────
    idx_ok = list(df["snapshot_index"]) == list(range(df.shape[0]))
    print(f"  snapshot_index 0…N-1:  {idx_ok}")

    # ── Label balance ────────────────────────────────────────────────────────
    n_normal = (df["label"] == 0).sum()
    n_faulty = (df["label"] == 1).sum()
    fault_pct = n_faulty / df.shape[0] * 100
    print(f"  Labels — Normal: {n_normal}  Faulty: {n_faulty}  "
          f"({fault_pct:.1f}% faulty)")
    fault_start = df["fault_start_idx"].iloc[0]
    print(f"  fault_start_idx:       {fault_start}")
    print(f"  label_method:          {df['label_method'].iloc[0]}")

    # ── NaN / Inf ────────────────────────────────────────────────────────────
    n_nan = df[feat_cols].isnull().sum().sum()
    n_inf = np.isinf(df[feat_cols].select_dtypes(include=np.number).values).sum()
    print(f"  NaN in features:       {n_nan}")
    print(f"  Inf in features:       {n_inf}")
    if n_nan > 0 or n_inf > 0:
        print("  ⚠ NON-FINITE VALUES PRESENT")
        all_ok = False

    # ── Constant features ────────────────────────────────────────────────────
    const_cols = [c for c in feat_cols if df[c].std() < 1e-10]
    print(f"  Constant features:     {len(const_cols)}")
    if const_cols:
        print(f"    Columns: {const_cols[:5]}")

    # ── dominant_freq check (should NOT be all 0.0 anymore) ──────────────────
    dom_freq_cols = [c for c in feat_cols if c.endswith("__dominant_freq")]
    all_zero_dom = [c for c in dom_freq_cols if df[c].max() < 1e-6]
    print(f"  dominant_freq cols all-zero: {all_zero_dom or 'none'}")
    if all_zero_dom:
        print("  ⚠ dominant_freq still all-zero — DC fix may not have applied")
        all_ok = False

    # ── Feature distribution summary (key features only) ────────────────────
    rms_cols  = [c for c in feat_cols if c.endswith("__rms")]
    kurt_cols = [c for c in feat_cols if c.endswith("__kurtosis")]
    dom_cols  = [c for c in feat_cols if c.endswith("__dominant_freq")]

    print(f"\n  RMS range per channel:")
    for c in rms_cols:
        v = df[c]
        print(f"    {c:<35} min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}  std={v.std():.4f}")

    print(f"\n  Kurtosis range per channel:")
    for c in kurt_cols:
        v = df[c]
        print(f"    {c:<35} min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}")

    print(f"\n  Dominant Freq range per channel (Hz):")
    for c in dom_cols:
        v = df[c]
        print(f"    {c:<35} min={v.min():.2f}  max={v.max():.2f}  "
              f"mean={v.mean():.2f}")

    # ── Normal vs Faulty feature contrast ────────────────────────────────────
    print(f"\n  RMS contrast (Normal vs Faulty mean):")
    df_n = df[df["label"] == 0]
    df_f = df[df["label"] == 1]
    for c in rms_cols[:4]:  # first 4 channels only
        mean_n = df_n[c].mean() if len(df_n) else 0
        mean_f = df_f[c].mean() if len(df_f) else 0
        ratio  = mean_f / mean_n if mean_n > 1e-10 else float("inf")
        flag   = "  ← degradation detectable" if ratio > 1.2 else ""
        print(f"    {c:<35}  Normal={mean_n:.4f}  Faulty={mean_f:.4f}  "
              f"ratio={ratio:.2f}×{flag}")

    # ── Check source_file uniqueness ─────────────────────────────────────────
    n_unique_files = df["source_file"].nunique()
    print(f"\n  Unique source files:   {n_unique_files} / {df.shape[0]}")
    if n_unique_files != df.shape[0]:
        print("  ⚠ DUPLICATE SOURCE FILES DETECTED")
        all_ok = False

    # ── First 3 rows ─────────────────────────────────────────────────────────
    print(f"\n  First 3 rows (metadata only):")
    meta_display = ["snapshot_index", "timestamp", "source_file",
                    "label_name", "n_channels"]
    print(df[meta_display].head(3).to_string(index=False))

print(f"\n{'='*65}")
if all_ok:
    print("  VALIDATION PASSED — Phase 3 feature datasets are clean.")
    print("  Recommend: review feature contrasts above, then approve Phase 4.")
else:
    print("  VALIDATION FAILED — issues listed above must be resolved.")
print("=" * 65)
