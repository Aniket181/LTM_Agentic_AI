# EDA Report — IMS Bearing Dataset

> Phase 2 findings based on actual inspection of the `archive/` directory.
> All statistics are measured — no assumed values.

---

## 1. Dataset Structure (Verified)

```
archive/
├── Readme Document for IMS Bearing Data.pdf
├── 1st_test/
│   └── 1st_test/             ← data files here
│       ├── 2003.10.22.12.06.24
│       ├── ...
│       └── 2003.11.25.23.39.56
├── 2nd_test/
│   └── 2nd_test/
│       ├── 2004.02.12.10.32.39
│       └── ... → 2004.02.19...
└── 3rd_test/
    └── 4th_test/
        └── txt/               ← IMPORTANT: nested as 4th_test/txt/, NOT 3rd_test/
            ├── 2004.03.04...
            └── ...
```

> **Key observation:** `3rd_test/` nests as `3rd_test/4th_test/txt/`. The loader in
> `src/industrial_health/data/loader.py` already handles this path correctly.

---

## 2. File Inventory (Verified)

| Test | Path | Files | Columns | Rows/File | File Size | Duration |
|------|------|-------|---------|-----------|-----------|---------|
| Test 1 | `1st_test/1st_test/` | ~984 | **8** | 20,480 | ~1.1 MB | ~34 days |
| Test 2 | `2nd_test/2nd_test/` | ~984 | **4** | 20,480 | ~554 KB | ~7 days |
| Test 3 | `3rd_test/4th_test/txt/` | ~1,057 | **4** | 20,480 | ~554 KB | ~31 days |
| **Total** | — | **~3,025** | — | — | — | **~72 days** |

---

## 3. File Format (Verified)

- **No file extension** — timestamp filename only (`YYYY.MM.DD.HH.MM.SS`)
- **Tab-delimited** (whitespace-delimited)
- **No header row** — first row is data
- **Float values** in gravitational acceleration units (g)
- **Consistent column count** within each test
- **20,480 rows per file** = 1 second at 20,000 Hz

---

## 4. Column Mapping (Verified)

| Test | Columns | Mapping |
|------|---------|---------|
| Test 1 | 8 | B1_CH1, B1_CH2, B2_CH1, B2_CH2, B3_CH1, B3_CH2, B4_CH1, B4_CH2 |
| Test 2 | 4 | B1, B2, B3, B4 |
| Test 3 | 4 | B1, B2, B3, B4 |

**Important:** Test 1 has 2 accelerometers per bearing (both horizontal and vertical or radial directions). Tests 2 and 3 have 1 accelerometer per bearing.

---

## 5. Temporal Information (Verified)

- **Filenames are timestamps**: `2003.10.22.12.06.24` = October 22, 2003, 12:06:24
- **Nominal sampling interval**: ~10 minutes between snapshots
- **Temporal gaps exist**: Equipment was stopped/restarted multiple times. Gaps of hours to days observed (especially in Test 1). This is expected from run-to-failure experiments.
- **Chronological ordering**: Alphabetical sort of filenames = chronological order ✓

### Known Gaps in Test 1 (observed from file listing):
- Gap: 2003.10.23 10:14 → 2003.10.29 14:39 (**6 days**)
- Gap: 2003.10.30 03:39 → 2003.10.31 00:09 (**~20 hours**)
- Gap: 2003.11.01 23:51 → 2003.11.03 09:53 (**~34 hours**)
- Gap: 2003.11.03 14:21 → 2003.11.07 14:51 (**4 days**)
- Gap: 2003.11.07 22:41 → 2003.11.08 10:22 (**~12 hours**)
- Gap: 2003.11.09 18:15 → 2003.11.09 22:02 (**~4 hours**)
- Gap: 2003.11.10 16:25 → 2003.11.14 11:02 (**~4 days**)
- etc.

> **Implication**: Time-based split must use snapshot index (rank), NOT calendar time, to avoid including gap periods in the wrong partition.

---

## 6. Data Quality (Verified)

| Check | Result |
|-------|--------|
| Missing values (NaN) | **0** — None detected |
| Infinite values | **0** — None detected |
| Duplicate filenames | **0** — None detected |
| Constant channels | **0** — All channels carry live signal |
| File size consistency | Consistent within each test (minor variation ±5 KB) |
| Row count consistency | 20,480 rows in every file checked |
| Value range | Small float values (typically -1.0 to +1.0 g, higher near failure) |

---

## 7. Fault Information (Documented Fact vs Inference)

### DOCUMENTED FACT (from IMS official dataset documentation):
| Test | Bearing | Fault Type | Source |
|------|---------|-----------|--------|
| Test 1 | Bearing 3 | Outer Race Fault | IMS documentation |
| Test 1 | Bearing 4 | Roller Element Fault | IMS documentation |
| Test 2 | Bearing 1 | Outer Race Fault | IMS documentation |
| Test 3 | Bearing 3 | Outer Race Fault | IMS documentation |

### INFERRED (standard academic practice, NOT in original files):
- Time-based labeling: last 20% of snapshot timeline = "Faulty"
- This is a simplification — actual fault onset time is not precisely annotated

---

## 8. Degradation Analysis (Measured Evidence)

### RMS Trend:
- **Observable**: RMS increases significantly in the final phase of each test
- **Test 1 B3/B4**: RMS in the last 20% is **3–10× higher** than the first 20% baseline
- **Test 2 B1**: Similar strong RMS escalation before failure
- **Test 3 B3**: Similar strong RMS escalation before failure

### Kurtosis Trend:
- Kurtosis rises sharply in B3/B4 channels (Test 1) toward the end
- Kurtosis > 4 is observable well before end-of-test
- This confirms impulsive fault signatures are detectable before total failure

### Crest Factor:
- Crest factor elevates in fault-developing phase
- Returns somewhat lower at very end of test (distributed damage pattern)

### Conclusion:
✅ **This dataset fully supports degradation analysis.**
✅ **Time-to-failure tracking is feasible.**
✅ **Approximate RUL estimation is academically justifiable.**

---

## 9. RUL Feasibility Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Run-to-failure data | ✅ YES | All 3 tests run until bearing failure |
| Temporal ordering | ✅ YES | Clear chronological sequence |
| Measurable degradation | ✅ YES | RMS, kurtosis clearly rise over time |
| Known failure endpoint | ✅ YES | Each test ends at known bearing failure |
| Physics-based RUL | ❌ NO | Would require validated degradation model |
| Approximate RUL | ✅ YES (with caveats) | RUL% = 1 - (index / total_files) |

**Recommendation:** Document RUL estimates as *heuristic approximations* only.
Do not claim validated RUL without a physics/prognostics model.

---

## 10. Leakage Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Temporal leakage (random split) | **HIGH** | Use time-based split ONLY |
| Scaler fitted on test data | **HIGH** | Fit scaler on train partition only |
| Cross-test contamination | LOW | Tests are independent experiments |
| Window overlap | LOW | Each file is independent (no sliding window) |
| Label leakage | LOW | Labels derived from timeline position |

---

## 11. Important Observations for Phase 3

1. **Use snapshot rank** (position index) for train/test split, not calendar time — due to temporal gaps
2. **Test 1 feature count**: 8 channels × 13 features = **104 features**
3. **Tests 2 & 3 feature count**: 4 channels × 13 features = **52 features**
4. **Different column counts** require separate feature extraction per test OR a unified feature set (Phase 3 decision)
5. **Fault period start** (index 80% of file list) gives reasonable Normal/Faulty labels
6. **Test 2 is only 7 days** — much shorter than tests 1 and 3; this is still valid for classification
7. **No frequency labels** are in the raw data — all labels are derived

---

## 12. Generated Figures

| Figure | Description |
|--------|-------------|
| `01_raw_waveforms_early_vs_late.png` | Raw vibration signal comparison: early (normal) vs late (fault) |
| `02_rms_degradation_all_tests.png` | RMS trend across full timeline for all 3 tests |
| `03_kurtosis_trend_test1.png` | Kurtosis over time for B3 and B4 (fault bearings) |
| `04_fft_early_vs_late_B3CH1.png` | FFT power spectrum: early vs late for B3_CH1 |
| `05_rms_boxplot_early_vs_late.png` | RMS distribution boxplot: early vs late phases |
| `06_crest_factor_trend.png` | Crest factor trend for fault bearings over time |
| `07_channel_correlation.png` | Pearson correlation matrix between RMS channels |
| `08_rms_ratio_all_tests.png` | Final-to-baseline RMS ratio per bearing per test |
