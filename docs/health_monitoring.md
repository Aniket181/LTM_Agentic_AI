# Health Monitoring — Phase 7

## 1. Objective

Phase 7 builds an explainable Health Monitoring layer that integrates outputs
from Phase 5 (Random Forest fault classifier) and Phase 6 (Isolation Forest
anomaly detector) with Phase 3 RMS features to produce a per-snapshot
**Health Score** (0–100) for Test 1.

> **The Health Score is an explainable engineering integration layer and is
> not presented as a scientifically validated clinical/industrial health index.**

---

## 2. Inputs

| Input | Source | Role |
|-------|--------|------|
| Phase 3 feature CSV | `data/features/test1_features.csv` | RMS degradation signal |
| Phase 5 RF model | `models/random_forest_test1.joblib` | Fault probability |
| Phase 5 RF scaler | `models/scaler/phase5_test1_scaler.pkl` | Feature scaling for RF |
| Phase 6 IF model | `models/isolation_forest_test1.joblib` | Anomaly score |
| Phase 6 IF scaler | `models/scaler/phase6_test1_scaler.pkl` | Feature scaling for IF |

No new ML model is trained in Phase 7.

---

## 3. Architecture

```
Phase 3 features (raw)
    │
    ├──[Phase 6 IF scaler]──► IF anomaly score  → anomaly_health_component (0–1)
    ├──[Phase 5 RF scaler]──► RF fault proba    → fault_health_component   (0–1)
    └──[RMS reference]──────► composite RMS     → rms_health_component     (0–1)
                                                          │
                                          ┌───────────────┘
                                          ▼
                         health_score = clip(100 × Σ w_i × component_i, 0, 100)
                                          │
                                          ▼
                                 Health Status (Stable / Degrading / Critical)
                                          │
                                          ▼
                                 Rule-based Explanation
```

---

## 4. Health Score Formula

```
health_score = clip(
    100 × (
        w_rms     × rms_component     +
        w_anomaly × anomaly_component +
        w_fault   × fault_component
    ),
    0.0, 100.0
)
```

All three components are normalised to **[0, 1]** where **1 = healthy, 0 = worst**.

---

## 5. Component Normalization

### 5.1 RMS Component

Derived from the mean of all `__rms` feature columns across all 8 channels.

```
composite_rms[i] = mean(bearing1_ch1__rms[i], ..., bearing4_ch2__rms[i])

rms_range = ref_max_rms - baseline_rms
rms_component[i] = clip(1 - (composite_rms[i] - baseline_rms) / rms_range, 0, 1)
```

**Reference statistics** are derived exclusively from the **Normal training period**
(first 70% of Normal snapshots — the same partition used by Phase 6):

- `baseline_rms` = median of the Normal training period composite RMS
- `ref_max_rms`  = 95th percentile of the Normal training period composite RMS

This prevents future-data leakage into earlier health values.

### 5.2 Anomaly Component

Uses Phase 6 `AnomalyDetector.decision_scores()`:
- `decision_scores() = −score_samples()` → **HIGHER = MORE ANOMALOUS**

```
clipped = clip(decision_scores, 0, ∞)       # negative = more normal than training
score_max = max(clipped)
anomaly_component[i] = clip(1 - clipped[i] / score_max, 0, 1)
```

### 5.3 Fault Component

Uses Phase 5 `RandomForestClassifier.predict_proba()[:, 1]` (P(Faulty)):

```
fault_component[i] = clip(1 - prob_faulty[i], 0, 1)
```

> **Note:** Phase 5 RF probabilities are not calibrated. They should not be
> interpreted as precise Bayesian fault probabilities.

---

## 6. Weights

| Component | Default Weight | Role |
|-----------|---------------|------|
| RMS (`w_rms`) | 0.20 | Physical degradation signal |
| Anomaly (`w_anomaly`) | 0.40 | Unsupervised deviation from normal |
| Fault (`w_fault`) | 0.40 | Supervised fault classifier confidence |

Weights are configurable via `HealthConfig`. They must sum to 1.0.

> **These weights are engineering defaults and have not been optimised against
> ground-truth fault data.**

---

## 7. Health Status Thresholds

| Status | Condition |
|--------|-----------|
| **Stable** | health_score ≥ 70 |
| **Degrading** | 40 ≤ health_score < 70 |
| **Critical** | health_score < 40 |

Thresholds are configurable via `HealthConfig.threshold_stable` and
`HealthConfig.threshold_degrading`.

> **These are engineering thresholds for this project prototype.
> They are not industry-certified or scientifically validated.**

---

## 8. Anomaly Score Semantics

Preserving Phase 6 score semantics throughout Phase 7:

| Signal | Semantics |
|--------|-----------|
| `decision_scores()` | Higher = more anomalous (HIGHER IS WORSE) |
| `anomaly_pred` | 1 = anomaly, 0 = normal |
| `anomaly_component` | Higher = healthier (HIGHER IS BETTER) |

The negation `anomaly_component = 1 - normalized(decision_scores)` performs the
semantic inversion explicitly and transparently.

---

## 9. RF Fault Confidence

Phase 5 RF output used in Phase 7:
- `prob_faulty = predict_proba()[:, 1]` — probability of Faulty class
- `fault_component = 1 - prob_faulty`

The RF was trained using:
- Heuristic temporal labels (final 20% = Faulty)
- Stratified Chronological Split protocol
- `class_weight="balanced"` to handle class imbalance

---

## 10. Trend Logic

Rolling linear regression slope over a configurable window (default 50 snapshots).

| Slope | Label |
|-------|-------|
| slope < −0.5 | Degrading |
| slope > +0.5 | Improving |
| otherwise | Stable |

**First degradation detection:** first position where at least 5 consecutive
snapshots are classified as Degrading or Critical.

**Run-level trend:** compares Q1 mean health (first 25% of run) vs Q4 mean
(last 25% of run). Delta > 5 → Improving; delta < −5 → Degrading; else Stable.

---

## 11. Outputs

| File | Type | Description |
|------|------|-------------|
| `reports/phase7_health_test1.csv` | CSV | Per-snapshot health, components, status, explanation |
| `reports/phase7_experiment_test1.json` | JSON | Experiment summary (tracked by Git) |
| `reports/figures/phase7_health_score_test1.png` | PNG | Health score timeline coloured by status |
| `reports/figures/phase7_component_comparison_test1.png` | PNG | Four stacked component panels |
| `reports/figures/phase7_trend_overlay_test1.png` | PNG | Health score with rolling trend labels |

Generated CSV and PNG files are git-ignored. The JSON is tracked.

---

## 12. Reproducibility

```bash
# Run Phase 7 health monitoring (Test 1 only)
python scripts/run_health_monitoring.py --test_id 1

# Run Phase 7 tests
python -m pytest tests/test_health.py -v

# Override weights and thresholds (must still sum to 1.0):
python scripts/run_health_monitoring.py --test_id 1 \
    --weight_rms 0.3 --weight_anomaly 0.35 --weight_fault 0.35
```

Outputs are deterministic: given the same input CSV and model artifacts,
the health scores are identical across runs.

---

## 13. Limitations

1. **Engineering prototype:** The Health Score is a prototype integration
   layer. Component weights, thresholds, and normalization choices are
   engineering defaults, not scientifically validated parameters.

2. **Heuristic labels propagate:** Phase 5 RF was trained on heuristic
   temporal labels. Its fault probabilities reflect the model's ability to
   separate the first 80% from the final 20% of the run, not genuine physical
   fault states.

3. **Phase 5/6 are experimental:** Both constituent models were trained under
   exploratory protocols. Health scores inherit all limitations of those models.

4. **No calibration:** RF probabilities are not calibrated against independent
   test data. `prob_faulty = 0.8` does not imply an 80% Bayesian probability
   of fault.

5. **Test 1 only:** Phase 7 processes Test 1 only. Tests 2 and 3 have different
   channel schemas and require retraining of all constituent models.

6. **Static reference period:** The RMS normalization reference is derived once
   from the Normal training period. Concept drift or sensor recalibration would
   require re-establishing the reference baseline.
