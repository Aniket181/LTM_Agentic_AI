# Fault Diagnosis — Phase 5

## 1. Objective

Phase 5 develops a supervised Random Forest baseline for bearing fault
classification using Phase 3 engineered features and a chronologically-safe
experimental split protocol.

No ground-truth fault-onset labels are available in the IMS dataset. All
classification results must be interpreted under the heuristic temporal
labeling protocol described below.

---

## 2. Input Features

Source: `data/features/test1_features.csv` (Phase 3 output).

| Property | Test 1 |
|----------|--------|
| Snapshots | 2,156 |
| Feature columns | 104 |
| Channels | 8 (`bearing1_ch1` … `bearing4_ch2`) |
| Features/channel | 13 |

### Features used (per channel × 13)

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

### Features excluded (metadata — never used as model inputs)

```
test_id, source_file, timestamp, n_channels,
snapshot_index, label, label_name, label_method, fault_start_idx
```

---

## 3. Why Phase 4 Split Cannot Directly Train a Binary Classifier

Phase 4 applies a strict chronological 70/30 split across the entire run
timeline (sorted by `snapshot_index`):

```
0 ──────────────── 70% ──────────────── 100%
│  Train (1,509)    │    Test (647)       │
│  100% Normal      │  Normal + Faulty    │
```

Because the heuristic Faulty label begins at the 80% mark (snapshot 1,724
for Test 1), the entire training partition (indices 0–1,508) contains only
Normal samples. A supervised binary classifier cannot be trained on a
single-class dataset; it requires both Normal and Faulty examples in the
training phase to learn a decision boundary. `class_weight='balanced'` cannot
create a missing class.

---

## 4. Phase 5 Experimental Split

> **Phase 4 provides the leakage-safe preprocessing pipeline. Because the
> heuristic Faulty label begins at 80% of each run, its 70/30 split produces
> a single-class training partition. Therefore, Phase 5 uses a separate
> chronological development/evaluation protocol for supervised classification.
> This protocol is experimental and does not convert the heuristic labels
> into ground-truth fault annotations.**

### Stratified Chronological Split Algorithm

```
Full timeline (sorted by snapshot_index):
[Normal period — snapshots 0–1723] [Faulty period — snapshots 1724–2155]

Within Normal period (1,724 snapshots):
  First 70% → Train-Normal
  Last  30% → Test-Normal

Within Faulty period (432 snapshots):
  First 70% → Train-Faulty
  Last  30% → Test-Faulty

Combined train = Train-Normal + Train-Faulty  (sorted by snapshot_index)
Combined test  = Test-Normal  + Test-Faulty   (sorted by snapshot_index)
```

> **Within each heuristic state period, the training portion chronologically
> precedes the corresponding testing portion. The Normal and Faulty periods
> are partitioned independently to ensure both classes are represented in
> training and evaluation.**

### Split sizes (Test 1)

| Partition | Total | Normal | Faulty |
|-----------|-------|--------|--------|
| Train | ~1,508 | ~1,207 | ~301 |
| Test | ~648 | ~517 | ~131 |

---

## 5. Label Methodology

> **IMPORTANT:** The binary Normal/Faulty labels are **heuristic temporal
> labels** based on the final 20% of each run's snapshot sequence. Explicit
> fault-onset annotations are NOT available in the IMS dataset.
> These labels must not be described as verified ground truth.

| Label | Integer | Meaning |
|-------|---------|---------|
| Normal | 0 | Snapshot in first 80% of run timeline |
| Faulty | 1 | Snapshot in final 20% of run timeline |

`label_method` = `"heuristic_time_fraction"` throughout.

---

## 6. Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
```

**Why `class_weight="balanced"`:** Training data has roughly 4:1 Normal:Faulty
ratio. Balanced weighting prevents the classifier from optimising for Normal
only.

**No hyperparameter tuning:** The goal is a transparent academic baseline, not
optimised benchmark performance.

---

## 7. Leakage Prevention

| Risk | Prevention Applied |
|------|-------------------|
| Random shuffling of time series | Never used; chronological ordering preserved throughout |
| Scaler fitted on full dataset | New `StandardScaler` fitted **only on X_train** of Phase 5 split |
| Phase 4 scaler reuse | Not reused — its fitting partition differs (0–70% of full run) |
| Test statistics in scaler | X_test is only `.transform()`-ed, never `.fit()`-ed |
| Metadata used as features | All metadata columns explicitly excluded |
| Feature selection using test data | Not performed |
| Hyperparameter tuning using test data | Not performed |

---

## 8. Evaluation Metrics

Evaluated on the Phase 5 test partition (unseen during training and scaling).

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision (Normal) | TP_Normal / (TP_Normal + FP_Normal) |
| Recall (Normal) | TP_Normal / (TP_Normal + FN_Normal) |
| F1 (Normal) | Harmonic mean of Normal precision and recall |
| Precision (Faulty) | TP_Faulty / (TP_Faulty + FP_Faulty) |
| Recall (Faulty) | TP_Faulty / (TP_Faulty + FN_Faulty) |
| F1 (Faulty) | Harmonic mean of Faulty precision and recall |
| Macro F1 | Unweighted mean of per-class F1 |
| Weighted F1 | Support-weighted mean of per-class F1 |
| Confusion matrix | 2×2 (Normal/Faulty × True/Predicted) |

For fault-detection problems, **Faulty Recall** (sensitivity) is the most
safety-critical metric — it measures the fraction of actual fault periods that
the model correctly identifies.

---

## 9. Feature Importance

`RandomForestClassifier.feature_importances_` (mean decrease in impurity, MDI)
is used to rank all 104 features. The top 20 are reported and plotted.

**Caution:** MDI-based importance can be biased toward high-cardinality features
on small datasets. For a rigorous academic analysis, permutation importance
(out-of-bag) would be preferable but is outside the Phase 5 scope.

---

## 10. Results

See: `reports/phase5_experiment_test1.json` for full numeric results.

Figures:
- `reports/figures/phase5_confusion_matrix_test1.png`
- `reports/figures/phase5_rf_feature_importance_test1.png`

---

## 11. Limitations

1. **Heuristic labels:** All evaluation metrics are measured against
   artificially defined Normal/Faulty temporal boundaries. A model that
   perfectly memorises the temporal boundary (last 20% = Faulty) would score
   well under this protocol without learning any genuine fault signature.
   Results must not be reported as verified fault-detection accuracy.

2. **Temporal autocorrelation:** Adjacent snapshots (10-minute intervals) are
   highly correlated. The stratified chronological split reduces temporal
   leakage but neighbouring snapshots may still share similar feature values.

3. **No degradation modelling:** The heuristic binary labels lose all
   information about gradual degradation leading up to failure. A degradation
   trend, health score, or multi-stage label would be more physically
   meaningful (planned for later phases).

4. **Test 1 only:** Tests 2 and 3 have different known fault bearings and
   different degradation profiles. Results from Test 1 should not be
   generalised to other tests.

5. **Scaler is test-specific:** The Phase 5 scaler was fitted on the Test 1
   Stratified Chronological training split only. It cannot be applied to
   Test 2 or Test 3 data.

---

## 12. Reproducibility

```bash
# Run Phase 5 experiment (Test 1 only)
python scripts/run_fault_diagnosis.py --test_id 1

# Run minimal model tests
python -m pytest tests/test_model.py -v
```

All randomness is seeded with `random_state=42`.

Outputs:
```
models/random_forest_test1.joblib
models/scaler/phase5_test1_scaler.pkl
reports/phase5_experiment_test1.json
reports/figures/phase5_confusion_matrix_test1.png
reports/figures/phase5_rf_feature_importance_test1.png
```

All generated binaries are git-ignored. Source code is version-controlled.
