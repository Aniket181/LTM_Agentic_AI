# Anomaly Detection — Phase 6

## 1. Purpose

Phase 6 implements an unsupervised anomaly detection baseline using Isolation
Forest. The detector is trained exclusively on Normal-period vibration feature
snapshots and then evaluated against all remaining snapshots — including both
the held-out Normal period and the entire heuristic Faulty period.

This is a complementary component to the Phase 5 supervised Random Forest
classifier. Where Phase 5 requires labelled data (from the heuristic protocol),
Phase 6 requires only unlabelled Normal-period data to learn a normal operating
signature.

---

## 2. Isolation Forest

Isolation Forest (Liu et al., 2008) detects anomalies by randomly partitioning
the feature space using isolation trees. Anomalous samples require fewer
partitions to isolate (shorter average path length), producing a lower score.

**Key properties:**
- Unsupervised — no labels required during training.
- Scales well to high-dimensional tabular features.
- Produces a continuous anomaly score (not just a binary prediction).
- Not suited for sequence modelling; treats each snapshot independently.

---

## 3. Dataset

Source: `data/features/test1_features.csv` (Phase 3 output).

| Property | Value |
|----------|-------|
| Total snapshots | 2,156 |
| Normal snapshots (label=0) | 1,724 |
| Faulty snapshots (label=1) | 432 |
| Feature columns | 104 |
| Channels | 8 |

---

## 4. Test 1 Protocol

This phase processes **Test 1 only**. Tests 2 and 3 are not evaluated here.

---

## 5. Normal-Only Training Strategy

> **The anomaly detector is trained exclusively on the Normal training period.
> Faulty-period samples are not used during fitting.**

The Isolation Forest learns a representation of normal bearing behaviour from
the earliest chronological Normal snapshots. Any future snapshot that deviates
significantly from this learned profile receives a high anomaly score.

---

## 6. Chronological Evaluation

```
Full timeline, sorted by snapshot_index:
│  NORMAL PERIOD (snapshots 0–1723)  │  FAULTY PERIOD (snapshots 1724–2155)  │

Within Normal period (1,724 snapshots):
    First 70%  → IsolationForest training  (~1,207 snapshots)
    Remaining 30% → evaluation only        (~517 snapshots)

Faulty period (432 snapshots):
    100%        → evaluation only (NEVER used for fitting)
```

The Isolation Forest, the StandardScaler, and all fitting statistics are
derived **only from the Normal training partition**.

All other data (held-out Normal snapshots + entire Faulty period) is only
transformed and evaluated — never used to compute any fitting parameters.

---

## 7. Feature Preprocessing

**Scaler:** `sklearn.preprocessing.StandardScaler`

Fitted **exclusively on the Normal training partition** (~1,207 rows).
All other partitions are only transformed using training statistics.

The Phase 4 and Phase 5 scalers are **not reused** — their fitting partitions
differ from the Phase 6 Normal training partition.

Phase 6 scaler saved at:
```
models/scaler/phase6_test1_scaler.pkl
```

**Note:** IsolationForest does not technically require scaling (it uses random
splits, not distance-based methods). Scaling is applied here for consistency
and to prevent features with large numerical ranges from dominating random splits.

**Metadata columns excluded (never used as features):**
```
test_id, source_file, timestamp, n_channels,
snapshot_index, label, label_name, label_method, fault_start_idx
```

---

## 8. Parameters

```python
IsolationForest(
    n_estimators=200,
    contamination="auto",   # ≈ 10% assumed anomaly rate
    max_samples="auto",     # subsampling for tree building
    max_features=1.0,       # all features used per tree
    random_state=42,
    n_jobs=-1,
)
```

`contamination="auto"` uses the standard IsolationForest threshold of
`score < -0.5` for anomaly classification. This is a baseline; the threshold
is not optimised.

---

## 9. Score Semantics

| Attribute | Source | Semantics |
|-----------|--------|-----------|
| `decision_scores()` | `−score_samples()` | **HIGHER = MORE ANOMALOUS** |
| `score_samples()` | raw IF output | More negative = more anomalous |
| `predict()` | converted IF predict | `1` = anomaly, `0` = normal |

> **Explicit note:** IsolationForest's raw `predict()` returns `−1` for
> outliers and `+1` for inliers. This module converts these to `1` (anomaly)
> and `0` (normal) for clarity and consistency with the Phase 5 label encoding.
> The `decision_scores()` method negates `score_samples()` so that **higher
> values always indicate greater anomaly likelihood**.

---

## 10. Evaluation Metrics

Evaluated on the combined evaluation set (held-out Normal + all Faulty
snapshots) against the heuristic labels.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall agreement with heuristic labels |
| Precision (Normal) | TP_Normal / (TP_Normal + FP_Normal) |
| Recall (Normal) | TP_Normal / (TP_Normal + FN_Normal) |
| F1 (Normal) | Harmonic mean |
| Precision (Faulty) | TP_Faulty / (TP_Faulty + FP_Faulty) |
| Recall (Faulty) | TP_Faulty / (TP_Faulty + FN_Faulty) |
| F1 (Faulty) | Harmonic mean |
| Macro F1 | Unweighted mean of per-class F1 |
| Weighted F1 | Support-weighted mean of per-class F1 |
| Confusion matrix | 2×2 (heuristic label × IF prediction) |

> **Evaluation uses heuristic temporal labels derived from the dataset's
> run-to-failure structure and should not be interpreted as verified
> fault-onset ground truth.**

---

## 11. Anomaly Timeline

`reports/figures/phase6_anomaly_score_test1.png`

X-axis: snapshot index (chronological).  
Y-axis: anomaly score (`−score_samples()`, higher = more anomalous).  
Blue: Normal-period snapshots; Red: Faulty-period snapshots.  
The heuristic fault boundary is marked with a vertical line.

`reports/figures/phase6_confusion_matrix_test1.png`

2×2 confusion matrix: heuristic label (rows) vs IF prediction (cols).

---

## 12. Limitations

1. **Heuristic labels:** The evaluation labels (final 20% = Faulty) are not
   verified fault-onset annotations. Metrics reflect agreement between the
   model and an artificial temporal boundary.

2. **Independent snapshot assumption:** Isolation Forest treats each 10-minute
   snapshot as independent. It does not model temporal autocorrelation or
   gradual degradation trends.

3. **contamination="auto" threshold:** The default threshold is not optimised
   for IMS fault detection. A higher contamination value would flag more
   snapshots as anomalies, potentially improving Faulty recall at the cost of
   Normal precision.

4. **Test 1 only:** The detector was trained on Test 1 Normal data. It cannot
   be applied to Tests 2 or 3 without retraining (different channel schemas).

5. **No sequence modelling:** A sliding window or LSTM-based approach would
   better capture the temporal degradation signature but is outside Phase 6 scope.

---

## 13. Academic Interpretation

> Results should be described as: "Isolation Forest anomaly detection
> performance under the heuristic temporal labeling protocol."

> Results must NOT be described as: "Verified bearing fault detection accuracy."

The model learns what the earliest Normal-period snapshots look like and
flags deviations. Whether those deviations correspond to genuine physical
fault onset — or simply to natural operating variability in the later Normal
period — cannot be determined from heuristic labels alone.

---

## 14. Reproducibility

```bash
# Run Phase 6 anomaly detection (Test 1 only)
python scripts/run_anomaly_detection.py --test_id 1

# Run minimal anomaly tests
python -m pytest tests/test_anomaly.py -v
```

All randomness seeded with `random_state=42`.

Outputs:
```
models/isolation_forest_test1.joblib
models/scaler/phase6_test1_scaler.pkl
reports/phase6_anomaly_test1.csv
reports/phase6_experiment_test1.json
reports/figures/phase6_anomaly_score_test1.png
reports/figures/phase6_confusion_matrix_test1.png
```

All generated binaries and data outputs are git-ignored.
