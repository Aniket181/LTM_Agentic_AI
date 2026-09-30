# Methodology Documentation

## Feature Engineering

### Time-Domain Features (10 per channel)
Computed from 20,480 raw vibration samples per snapshot:

| Feature | Formula | Fault Relevance |
|---------|---------|----------------|
| Mean | E[x] | Baseline shift detection |
| RMS | √(E[x²]) | Overall vibration energy — primary health indicator |
| Std | √(E[(x-μ)²]) | Signal variability |
| Variance | E[(x-μ)²] | Signal spread |
| Kurtosis | E[(x-μ)⁴]/σ⁴ | Impulsiveness — key early fault indicator |
| Skewness | E[(x-μ)³]/σ³ | Signal asymmetry |
| Peak | max(|x|) | Maximum shock amplitude |
| Peak-to-Peak | max(x) - min(x) | Total amplitude range |
| Crest Factor | Peak/RMS | Impulsive fault detection |
| Shape Factor | RMS/mean(|x|) | Waveform shape indicator |

### Frequency-Domain Features (3 per channel)
Computed via FFT of 20,480 samples at 20 kHz:

| Feature | Description |
|---------|-------------|
| Dominant Frequency | Frequency with highest spectral magnitude |
| Spectral Energy | Total energy in frequency domain (Σ|X(f)|²) |
| Spectral Centroid | Weighted mean frequency (Σf·|X(f)|/Σ|X(f)|) |

### Total Feature Count
- Test 1: 8 channels × 13 features = **104 features per snapshot**
- Tests 2 & 3: 4 channels × 13 features = **52 features per snapshot**

## Labeling Methodology

**Approach:** Time-based binary labeling using known fault outcomes.

```
Timeline:   |-------- Normal (80%) ---------|-- Faulty (20%) --|
             t=0                           t=0.8T             t=T (failure)
Label:        0                              0         1  1  1  1
```

**Rationale:**
- IMS is a run-to-failure dataset; faults develop progressively
- Last 20% captures the actively degrading + failed period
- This is standard practice in academic IMS research
- Limitation: actual fault onset time is not precisely known

**Alternative approaches** (for future work):
- RMS threshold crossing (label fault when RMS exceeds 2× baseline)
- Manual annotation from subject matter experts
- Change-point detection algorithms

## Train/Test Split Strategy

**Method:** Time-based split (NO random splitting)
- Train: First 70% of chronological snapshots
- Test: Last 30% of chronological snapshots

**Why not random split?**
- IMS data has temporal autocorrelation
- Random split would allow test data to contain snapshots from the same time window as training data
- This creates data leakage — the model would "see the future"
- Time-based split is the only valid approach for time-series data

## Model Training Details

### Random Forest
- 200 estimators, balanced class weights (handles Normal >> Faulty imbalance)
- Features: StandardScaler normalized (fitted on train only)
- No hyperparameter search initially (baseline first)

### Isolation Forest
- Trained on Normal period only (unsupervised)
- Contamination: 0.05 (5% expected outliers)
- 200 estimators

### Normalization
- StandardScaler: zero mean, unit variance
- Fitted on training data only
- Applied (transformed) to test data using training statistics
- Scaler saved to models/feature_scaler.pkl

## Health Score Computation

```
health_score = 0.40 × anomaly_component
             + 0.40 × fault_component
             + 0.20 × degradation_component

Where:
  anomaly_component    = normalized_anomaly_score × 100
  fault_component      = confidence_in_normal_class × 100
  degradation_component = (1 - (RMS_current - RMS_baseline) / RMS_range) × 100
```

**Weight rationale:**
- Anomaly and fault have equal weight (40% each) — both are ML predictions
- Degradation has lower weight (20%) — RMS trend is a simpler heuristic
- Weights are configurable — treat as initial engineering judgment

## Evaluation Metrics

### Fault Classification
- Accuracy, Precision, Recall, F1-score
- Confusion matrix (True Positive, False Negative, False Positive, True Negative)
- All metrics computed on held-out test set (last 30% of timeline)

### Anomaly Detection
- Precision: fraction of flagged anomalies that are truly faulty
- Recall: fraction of truly faulty samples caught
- Threshold-based evaluation using fault labels as ground truth

> All metrics are **MEASURED results** from actual experiments.
> No results are assumed or fabricated.
