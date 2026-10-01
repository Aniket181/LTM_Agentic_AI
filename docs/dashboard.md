# Dashboard — Phase 10

## 1. Purpose

Phase 10 adds a **Streamlit presentation layer** over the existing
Phase 5–9 pipeline. It provides a real-time interactive dashboard for:

- Equipment health score and trend
- Fault diagnosis (Random Forest)
- Anomaly detection (Isolation Forest)
- RAG knowledge retrieval from the curated bearing knowledge base

> **No ML logic lives in the dashboard.** All computation is delegated to
> existing Phase 5–9 components via `dashboard/services.py`.

---

## 2. Architecture

```
dashboard/app.py            ← Streamlit presentation (tabs, layout)
dashboard/config.py         ← Path constants
dashboard/services.py       ← Bridge to Phase 5–9 pipeline (cached)
dashboard/components/
    health_card.py          ← Health gauge + metrics
    trend_chart.py          ← Health score timeline (Plotly)
    fault_card.py           ← Fault prediction display
    anomaly_card.py         ← Anomaly score display
    rag_panel.py            ← Knowledge retrieval UI
    exec_log.py             ← Agent execution log
```

**Execution flow:**

```
User clicks "Run Pipeline"
        ↓
services.run_pipeline(test_id)
        ↓
OrchestratorAgent.run()   ← Phase 8
        ↓
  DataHealthAgent          ← Phase 7 HealthMonitor
  FaultDiagnosisAgent      ← Phase 5 RandomForest
  AnomalyAgent             ← Phase 6 IsolationForest
        ↓
OrchestratorAgent.build_final_result()
        ↓
dashboard tabs render result

User types knowledge query
        ↓
services.retrieve_knowledge(query)
        ↓
KnowledgeRetriever.retrieve()  ← Phase 9 ChromaDB + Ollama
        ↓
Results displayed as expandable sections
```

---

## 3. How to Run

### Prerequisites

1. Phase 5–9 must be complete (models trained, ChromaDB built)
2. Ollama must be running for the RAG panel (optional — rest of dashboard works without it)

```bash
# Verify model artifacts exist
ls models/random_forest_test1.joblib
ls models/isolation_forest_test1.joblib
ls models/scaler/phase5_test1_scaler.pkl
ls models/scaler/phase6_test1_scaler.pkl

# Build knowledge base if not already done
python scripts/build_knowledge_base.py

# Start Ollama (for RAG panel)
ollama serve
```

### Launch Dashboard

```powershell
# From project root
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default.

---

## 4. Dashboard Panels

### Tab 1 — Health Overview

| Element | Source |
|---|---|
| Health Score gauge | `OrchestratorAgent` → `HealthMonitor.score_dataset()` |
| Final Score metric | `AgentState.final_health_score` |
| Mean Score metric | `AgentState.mean_health_score` |
| Status (Stable / Degrading / Critical) | `AgentState.final_health_status` |
| Trend (Improving / Stable / Degrading) | `AgentState.final_health_trend` |
| Health timeline chart | `HealthMonitor.score_dataset()` per-snapshot DataFrame |
| Component breakdown (toggle) | `health_rms_component`, `health_anomaly_component`, `health_fault_component` |

**Threshold lines:**
- Green dashed at 70 → Stable boundary
- Red dashed at 40 → Critical boundary
- Purple dotted → Heuristic label boundary (NOT verified fault onset)

### Tab 2 — Fault & Anomaly

| Element | Source |
|---|---|
| Fault label | `FaultDiagnosisAgent` → `AgentState.final_fault_label` |
| Fault confidence | `AgentState.final_prob_faulty` or `final_prob_normal` |
| Probability bars | `prob_normal` / `prob_faulty` |
| Anomaly label | `AnomalyAgent` → `AgentState.final_anomaly_pred` |
| Anomaly score | `AgentState.final_anomaly_score` (higher = more anomalous) |
| Anomaly count | `AgentState.n_anomaly_predicted` |

### Tab 3 — Knowledge Base

| Element | Source |
|---|---|
| Query input | User text |
| Suggestions | Predefined query list |
| Results | `KnowledgeRetriever.retrieve()` → ChromaDB |
| Source document | `RetrievalResult.source` |
| Section | `RetrievalResult.section` |
| Distance | `RetrievalResult.distance` (0=most similar, 2=dissimilar) |

---

## 5. Caching Strategy

| Object | Cache Type | TTL |
|---|---|---|
| `OrchestratorAgent` instance | `@st.cache_resource` | Persistent (session) |
| `HealthMonitor` instance | `@st.cache_resource` | Persistent (session) |
| `KnowledgeRetriever` instance | `@st.cache_resource` | Persistent (session) |
| Pipeline result dict | `@st.cache_data` | 3600s (1 hour) |
| Health timeseries DataFrame | `@st.cache_data` | 3600s (1 hour) |
| RAG query results | Not cached (user input) | None |

Use **"Re-run (Clear Cache)"** in the sidebar to reset cached results.

---

## 6. Error Handling

| Scenario | Dashboard Behaviour |
|---|---|
| Model artifact missing | `st.error()` with path details; `st.stop()` |
| Feature CSV missing | `st.error()` within health timeseries section |
| Ollama not running | `st.error("Ollama not available")` in RAG panel only; rest of dashboard unaffected |
| ChromaDB empty | `st.error()` with `build_knowledge_base.py` instruction |
| Pipeline status = FAILED | `st.error()` with agent error details; `st.stop()` |

---

## 7. Limitations and Disclaimers

1. **Heuristic labels:** The RF model was trained on temporal labels (final 20% = Faulty). These are not verified fault-onset timestamps. The purple marker in the trend chart marks this boundary.

2. **IF anomaly semantics:** Anomaly score = `−score_samples()`. Higher = more anomalous. IF was trained on Normal-only data.

3. **Health Score is an integration layer**, not a scientifically validated health index. It combines RF probability, IF anomaly score, and RMS degradation with fixed weights.

4. **RAG retrieval only:** No LLM generation occurs. Displayed knowledge comes from curated Markdown documents in `knowledge_base/`. Retrieved passages are reference material, not fault confirmation.

5. **Test 1 only (Phase 10):** Test 2 and Test 3 feature CSVs exist but no models have been trained for them. They will appear in the Test selector automatically when models are trained in a future phase.

6. **Not for safety-critical decisions.** This dashboard is a research prototype.

---

## 8. Running Tests

```powershell
# Dashboard unit tests only (no models, no Ollama required)
pytest -q tests/test_dashboard.py

# Full suite regression
pytest -q

# Import smoke test
python -c "import sys; sys.path.insert(0,'src'); import dashboard.app; print('imports OK')"
```

---

## 9. File Index

| File | Role |
|---|---|
| `dashboard/app.py` | Streamlit entrypoint |
| `dashboard/config.py` | Project path constants |
| `dashboard/services.py` | Cached pipeline bridge |
| `dashboard/components/health_card.py` | Health gauge widget |
| `dashboard/components/trend_chart.py` | Plotly timeline |
| `dashboard/components/fault_card.py` | Fault prediction widget |
| `dashboard/components/anomaly_card.py` | Anomaly widget |
| `dashboard/components/rag_panel.py` | RAG query UI |
| `dashboard/components/exec_log.py` | Execution log viewer |
| `tests/test_dashboard.py` | Phase 10 test suite |
