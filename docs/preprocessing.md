# Preprocessing — Phase 4

## 1. Purpose

Phase 4 produces leakage-safe, model-ready feature arrays from the Phase 3
feature CSVs. It applies a chronological train/test split and fits a
`StandardScaler` on the training partition only.

No model is trained in this phase.

---

## 2. Input Datasets

| File | Rows | Feature cols | Metadata cols | Total cols |
|------|------|-------------|---------------|------------|
| `data/features/test1_features.csv` | 2,156 | 104 | 9 | 113 |
| `data/features/test2_features.csv` | 984 | 52 | 9 | 61 |
| `data/features/test3_features.csv` | 6,324 | 52 | 9 | 61 |

---

## 3. Schema Differences

Test 1 and Tests 2/3 have **different channel schemas** and must be processed
independently. They must **never** be concatenated.

| | Test 1 | Tests 2 & 3 |
|-|--------|-------------|
| Channels | 8 | 4 |
| Channel names | `bearing1_ch1`, `bearing1_ch2`, … `bearing4_ch2` | `bearing1`, `bearing2`, `bearing3`, `bearing4` |
| Features/channel | 13 | 13 |
| Feature columns | 104 | 52 |

---

## 4. Feature and Metadata Separation

### Metadata columns (9) — excluded from X arrays

```
test_id, source_file, timestamp, n_channels,
snapshot_index, label, label_name, label_method, fault_start_idx
```

### Feature columns — included in X arrays

Named `{channel}__{feature}`, e.g. `bearing3_ch1__kurtosis`.

The 13 features per channel:

| # | Name | Domain |
|---|------|--------|
| 1 | mean | Time |
| 2 | rms | Time |
| 3 | std | Time |
| 4 | variance | Time |
| 5 | kurtosis | Time |
| 6 | skewness | Time |
| 7 | peak | Time |
| 8 | peak_to_peak | Time |
| 9 | crest_factor | Time |
| 10 | shape_factor | Time |
| 11 | dominant_freq | Frequency |
| 12 | spectral_energy | Frequency |
| 13 | spectral_centroid | Frequency |

---

## 5. Chronological Split

**Split method:** Rank-based (ascending `snapshot_index`), never random.

```
Sorted by snapshot_index (ascending)
       ┌──────────────────────────────┬──────────────┐
       │  TRAIN (first 70%)           │  TEST (30%)  │
       └──────────────────────────────┴──────────────┘
       0                           split_idx       n-1
```

**Why rank-based instead of timestamp-based:**
The IMS dataset contains temporal gaps (machine stops). Using calendar-time
percentiles could assign gap periods to the wrong partition. Snapshot index
(chronological rank) is gap-robust.

**Guarantees:**
- No row appears in both train and test.
- All train `snapshot_index` values are strictly less than all test values.
- All train `timestamp` values precede all test `timestamp` values.
- No random shuffling is ever applied.

| Test | n_total | n_train (~70%) | n_test (~30%) |
|------|---------|----------------|---------------|
| 1 | 2,156 | ~1,509 | ~647 |
| 2 | 984 | ~688 | ~296 |
| 3 | 6,324 | ~4,426 | ~1,898 |

---

## 6. Scaling

**Scaler:** `sklearn.preprocessing.StandardScaler`

**Fitting rule — strictly train-only:**

```python
scaler.fit(X_train)          # fit on training data ONLY
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # transform only — never fit
```

The test set statistics are never used to compute scaling parameters.
One scaler is saved per test:

```
models/scaler/test1_scaler.pkl
models/scaler/test2_scaler.pkl
models/scaler/test3_scaler.pkl
```

**Expected result:** X_train columns have approximately mean≈0, std≈1.
X_test may deviate slightly (its distribution was not used in fitting).

---

## 7. Leakage Prevention

| Risk | Prevention |
|------|-----------|
| Temporal leakage from random split | Chronological split only; never `train_test_split(shuffle=True)` |
| Scaler fitted on full dataset | `scaler.fit()` called on `X_train` only |
| Test statistics influencing scaler | `scaler.transform()` only on X_test; no `.fit_transform()` |
| Cross-test contamination | Tests processed independently; never concatenated |
| Duplicate rows across partitions | Validated: no `snapshot_index` or `source_file` overlap |

---

## 8. Label Methodology

> **IMPORTANT:** The binary Normal/Faulty labels used at this stage are
> **heuristic temporal labels** based on the final 20% of each run because
> explicit fault-onset annotations are not available in the current dataset.
> These labels must not be described as verified ground truth.

| Label | Integer | Meaning |
|-------|---------|---------|
| Normal | 0 | Snapshot in first 80% of run timeline |
| Faulty | 1 | Snapshot in final 20% of run timeline |

The `label_method` field in all metadata CSVs and `split_config.json` is set
to `"heuristic_time_fraction"` as a permanent, queryable record of this choice.

The `fault_start_idx` field records the exact snapshot index at which the
Faulty period begins for each test.

---

## 9. Output Structure

```
data/processed/
├── test1/
│   ├── X_train.npy         — float64, shape (n_train, 104)
│   ├── X_test.npy          — float64, shape (n_test, 104)
│   ├── y_train.npy         — int,     shape (n_train,)
│   ├── y_test.npy          — int,     shape (n_test,)
│   ├── meta_train.csv      — metadata rows aligned with X_train
│   ├── meta_test.csv       — metadata rows aligned with X_test
│   ├── feature_names.txt   — ordered feature names, one per line
│   └── split_config.json   — split parameters and label disclaimer
├── test2/
│   └── (same structure, 52 features)
└── test3/
    └── (same structure, 52 features)

models/scaler/
├── test1_scaler.pkl
├── test2_scaler.pkl
└── test3_scaler.pkl
```

### split_config.json fields

| Field | Description |
|-------|-------------|
| `test_id` | 1, 2, or 3 |
| `n_total` | Total snapshots in this test |
| `n_train` | Training partition size |
| `n_test` | Test partition size |
| `split_idx` | First row index of test partition |
| `train_fraction` | 0.70 |
| `n_features` | Number of feature columns |
| `split_method` | `"chronological_snapshot_index"` |
| `scaler` | `"sklearn.preprocessing.StandardScaler"` |
| `scaler_fit` | `"train_partition_only"` |
| `label_col` | `"label"` |
| `label_encoding` | `{"0": "Normal", "1": "Faulty"}` |
| `label_disclaimer` | Full heuristic label warning text |
| `metadata_cols` | List of all metadata column names |

---

## 10. Validation

Run:
```bash
python scripts/validate_preprocessing.py --tests 1 2 3
```

Checks performed (14 total):

| # | Check |
|---|-------|
| 1 | Train/test row counts ≈ 70/30 (±3% tolerance) |
| 2 | No `source_file` overlap between train and test |
| 3 | No `snapshot_index` overlap |
| 4 | Training timestamps all precede test timestamps |
| 5 | No NaN in X_train or X_test |
| 6 | No Inf in X_train or X_test |
| 7 | Feature dimensions consistent (X_train, X_test, feature_names) |
| 8 | Metadata row count matches X array row count |
| 9 | `label`, `label_method`, `fault_start_idx` preserved in metadata |
| 10 | `label_method == "heuristic_time_fraction"` |
| 11 | Scaler fitted with correct feature count |
| 12 | X_train column means ≈ 0 (|mean| < 0.10) |
| 13 | X_train column stds ≈ 1 (|std-1| < 0.15) |
| 14 | X_test matches re-transformation with saved scaler |
| 15 | Label disclaimer present in split_config.json |

---

## 11. Limitations

1. **Heuristic labels:** The 80/20 Normal/Faulty boundary is an approximation.
   True fault onset timing is not annotated in the IMS dataset.
   Model performance metrics derived from these labels reflect the heuristic,
   not a clinically or mechanically validated threshold.

2. **Test 1 vs Tests 2/3 incompatibility:** Different channel counts and naming
   mean these tests cannot be combined without a feature alignment step.
   A bearing-agnostic feature set (e.g. selecting only shared feature names)
   could be used in future work.

3. **3-class labeling not implemented:** A Normal / Degrading / Faulty
   classification (using e.g. 0–60% Normal, 60–80% Degrading, 80–100% Faulty)
   is a valid extension but is not implemented here to keep Phase 4 minimal.

4. **Scalers are test-specific:** A scaler trained on Test 1 must not be used
   to scale Test 2 or Test 3 data — the channel schemas differ.

5. **Processed outputs are not committed to Git** — they are reproducible from
   Phase 3 feature CSVs by re-running `run_preprocessing.py`.
