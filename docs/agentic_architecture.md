# Agentic Architecture — Phase 8

## 1. Motivation

Phase 8 introduces an orchestration layer over the existing deterministic ML
and health-monitoring components developed in Phases 5–7. It does not add any
new machine-learning training or LLM reasoning.

The agent architecture improves the system in three ways:

1. **Separation of concerns** — each component's logic (fault diagnosis,
   anomaly detection, health monitoring) lives in a dedicated agent.
2. **Transparent execution** — a shared state object carries every result,
   and an execution log records which agent ran, when, and with what outcome.
3. **Extensibility** — the sequential state-based pattern can be upgraded
   to LangGraph or another workflow framework in a later phase without
   changing the individual agent APIs.

> **Phase 8 introduces orchestration around existing deterministic ML and
> health-monitoring components. It does not introduce autonomous LLM
> reasoning or new machine-learning training.**

---

## 2. Agent Responsibilities

| Agent | Phase reused | Responsibility |
|-------|-------------|----------------|
| `DataHealthAgent` | Phase 7 | Load feature CSV, run HealthMonitor, run trend analysis |
| `FaultDiagnosisAgent` | Phase 5 | Load RF model + scaler, score all snapshots, report fault summary |
| `AnomalyAgent` | Phase 6 | Load IF model + scaler, score all snapshots, report anomaly summary |
| `OrchestratorAgent` | (new) | Initialize state, run agents in sequence, assemble final result |

---

## 3. Shared State (`AgentState`)

A typed `dataclass` passed between agents. Never stores model objects.

Key fields:

```
test_id                  — IMS test ID
feature_csv_path         — resolved path to feature CSV
n_snapshots / n_features — dataset dimensions
fault_start_idx          — heuristic fault boundary

# Populated by DataHealthAgent:
health_results           — dict of health summary statistics
mean_health_score        — float
final_health_score       — float
final_health_status      — "Stable" | "Degrading" | "Critical"
final_health_trend       — "Stable" | "Degrading" | "Improving"
status_counts            — {Stable: N, Degrading: N, Critical: N}

# Populated by FaultDiagnosisAgent:
fault_results            — dict of RF summary statistics
final_rf_pred            — 0 (Normal) | 1 (Faulty)
final_prob_normal        — float
final_prob_faulty        — float
final_fault_label        — "Normal" | "Faulty"

# Populated by AnomalyAgent:
anomaly_results          — dict of IF summary statistics
final_anomaly_score      — float (higher = more anomalous)
final_anomaly_pred       — 0 (normal) | 1 (anomaly)
n_anomaly_predicted      — int

# Execution metadata:
execution_log            — list of {agent, status, duration_s, error?}
errors                   — list of "AgentName: message" strings
status                   — "INITIALIZED" | "RUNNING" | "COMPLETED" | "FAILED"
```

---

## 4. Execution Sequence

```
OrchestratorAgent.run()
    │
    ├─ 1. DataHealthAgent.run(state)
    │       Load feature CSV
    │       Lazy-load HealthMonitor (Phase 7)
    │       score_dataset() → per-snapshot health scores
    │       rolling_trend() → per-snapshot trend labels
    │       compute_run_trend_summary() → run-level trend
    │       → state.health_results, state.final_health_*
    │
    ├─ 2. FaultDiagnosisAgent.run(state)
    │       Lazy-load FaultClassifier + scaler (Phase 5)
    │       predict_proba() on all snapshots
    │       → state.fault_results, state.final_fault_*
    │
    └─ 3. AnomalyAgent.run(state)
            Lazy-load AnomalyDetector + scaler (Phase 6)
            decision_scores() on all snapshots
            predict() on all snapshots
            → state.anomaly_results, state.final_anomaly_*
```

If `stop_on_failure=True` (default) and `DataHealthAgent` fails, downstream
agents are not called.

---

## 5. Error Handling

- Each agent catches its own errors and writes them to `state.errors`.
- Errors are never swallowed silently.
- No agent fabricates health/fault/anomaly values if its model is missing.
- The orchestrator checks each agent's log entry for `"failed"` status.
- Missing model artifacts raise `FileNotFoundError` with an explicit message.

Error entry format in `state.execution_log`:
```json
{
  "agent": "FaultDiagnosisAgent",
  "status": "failed",
  "duration_s": 0.003,
  "error": "Phase 5 artifact not found: models/random_forest_test1.joblib. Run Phase 5 first."
}
```

---

## 6. Existing ML Components Reused

| Component | Where loaded | What it does |
|-----------|-------------|--------------|
| `FaultClassifier` (Phase 5) | `FaultDiagnosisAgent` | RF prediction, predict_proba |
| `AnomalyDetector` (Phase 6) | `AnomalyAgent` | decision_scores(), predict() |
| `HealthMonitor` (Phase 7) | `DataHealthAgent` | score_dataset() |
| `HealthConfig` (Phase 7) | `DataHealthAgent` / CLI | Weights, thresholds |
| `rolling_trend` (Phase 7) | `DataHealthAgent` | Per-snapshot trend labels |
| `compute_run_trend_summary` (Phase 7) | `DataHealthAgent` | Run-level trend |

---

## 7. Score Semantics (preserved from Phase 6)

| Signal | Source | Semantics |
|--------|--------|-----------|
| `anomaly_score` | `AnomalyDetector.decision_scores()` | **HIGHER = MORE ANOMALOUS** |
| `anomaly_pred` | `AnomalyDetector.predict()` | `1` = anomaly, `0` = normal |
| `prob_faulty` | `RF.predict_proba()[:, 1]` | P(Faulty class) |
| `fault_confidence` | `max(prob_normal, prob_faulty)` | confidence in predicted class |

These semantics are not redefined or reversed in Phase 8.

---

## 8. Deterministic Orchestration

All Phase 8 behavior is deterministic:
- No random seeds.
- No LLM calls.
- No external API requests.
- Given the same input CSV and model artifacts, every run produces identical results.

The pipeline is fully testable with mocked agents and synthetic data (no
real model artifacts required for unit tests).

---

## 9. Limitations

1. **Sequential only:** Phase 8 runs agents one after another. Parallel
   execution (e.g., running FaultDiagnosisAgent and AnomalyAgent concurrently)
   is not implemented.

2. **Test 1 only:** Phase 8 processes Test 1 only (same scope as Phases 5–7).

3. **No LLM:** Explanations are deterministic rule-based strings from Phase 7.
   LLM-based natural language summaries are a Phase 9+ concern.

4. **No RAG:** Knowledge retrieval, maintenance recommendations, and root-cause
   analysis are out of scope for Phase 8.

5. **Heuristic labels propagate:** All constituent models (Phase 5 RF, Phase 6 IF)
   were trained or evaluated using heuristic temporal labels. Their outputs
   carry the same limitations documented in Phases 5 and 6.

---

## 10. Reproducibility

```bash
# Run Phase 8 agentic pipeline (Test 1 only)
python scripts/run_agentic_pipeline.py --test_id 1

# Run Phase 8 unit tests
python -m pytest tests/test_agents.py -v

# Override health weights (must sum to 1.0):
python scripts/run_agentic_pipeline.py --test_id 1 \
    --weight_rms 0.3 --weight_anomaly 0.35 --weight_fault 0.35
```

Required artifacts (must exist before running):
```
models/random_forest_test1.joblib      (Phase 5)
models/scaler/phase5_test1_scaler.pkl  (Phase 5)
models/isolation_forest_test1.joblib   (Phase 6)
models/scaler/phase6_test1_scaler.pkl  (Phase 6)
data/features/test1_features.csv       (Phase 3)
```
