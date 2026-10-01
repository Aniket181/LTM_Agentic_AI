# API Documentation — Phase 11

## 1. Overview

Phase 11 exposes the Agentic AI Industrial Equipment Health Monitoring pipeline
as a **versioned REST API** via FastAPI.

The API delegates all computation to existing Phase 5–9 components:

| Route area | Underlying component |
|---|---|
| `/api/v1/pipeline/run` | `OrchestratorAgent` (Phase 8) |
| `/api/v1/pipeline/{id}/timeseries` | `HealthMonitor.score_dataset()` (Phase 7) |
| `/api/v1/pipeline/{id}/fault` | `FaultDiagnosisAgent` → RF (Phase 5) |
| `/api/v1/pipeline/{id}/anomaly` | `AnomalyAgent` → IF (Phase 6) |
| `/api/v1/rag/query` | `KnowledgeRetriever` → ChromaDB (Phase 9) |

**No ML logic is implemented in the API layer.**

---

## 2. How to Start the API Server

```powershell
# From the project root, with venv activated
venvagentic\Scripts\activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Or without `--reload` for stable operation:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Interactive documentation (auto-generated):**

| URL | Format |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

## 3. Base URL and Versioning

```
http://127.0.0.1:8000
```

All application endpoints are versioned under `/api/v1/`.
The `/health` endpoint is unversioned (service-level liveness check).

---

## 4. Endpoint Reference

### 4.1 `GET /health`

Service liveness and artifact availability check.
**Does not load models.** Safe to call frequently.

**Response 200:**
```json
{
  "status": "ok",
  "service": "Industrial Equipment Health API",
  "version": "1.0.0",
  "artifacts": {
    "test1_models":   true,
    "knowledge_base": true,
    "chroma_db":      true
  }
}
```

`status` is `"degraded"` if any artifact is missing.

---

### 4.2 `GET /api/v1/pipeline/available-tests`

List test IDs for which all model artifacts exist.

**Response 200:**
```json
{
  "available_test_ids": [1],
  "note": "Test IDs for which all required model artifacts exist."
}
```

---

### 4.3 `POST /api/v1/pipeline/run`

Execute the full Phase 8 agentic pipeline and cache the result.

**⚠ This operation may take 10–60 seconds** (RF + IF inference on 2,156 snapshots).
The result is cached in memory — subsequent GET requests use the cache.

**Request body:**
```json
{"test_id": 1}
```

**Response 200:**
```json
{
  "test_id": 1,
  "health": {
    "mean_health_score":   72.5,
    "final_health_score":  68.1,
    "final_health_status": "Degrading",
    "health_trend":        "Degrading",
    "status_counts": {"Stable": 1500, "Degrading": 500, "Critical": 156}
  },
  "fault": {
    "final_rf_pred":      1,
    "predicted_fault":    "Faulty",
    "normal_probability": 0.18,
    "fault_probability":  0.82,
    "fault_confidence":   0.82,
    "label_disclaimer":   "RF trained on heuristic temporal labels..."
  },
  "anomaly": {
    "anomaly_score":       0.45,
    "anomaly_prediction":  1,
    "anomaly_label":       "Anomaly",
    "n_anomaly_predicted": 312,
    "score_semantics":     "decision_scores() = -score_samples(). HIGHER = MORE ANOMALOUS.",
    "label_disclaimer":    "IF trained on Normal-only data..."
  },
  "execution": {
    "agents_executed":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
    "execution_order":  ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
    "execution_status": "COMPLETED",
    "errors": [],
    "execution_log": [
      {"agent": "DataHealthAgent",    "status": "completed", "duration_s": 5.2},
      {"agent": "FaultDiagnosisAgent","status": "completed", "duration_s": 0.4},
      {"agent": "AnomalyAgent",       "status": "completed", "duration_s": 0.5}
    ]
  }
}
```

**Error codes:**
- `404` — no models for `test_id`
- `422` — invalid `test_id` (out of range 1–3)
- `500` — pipeline internal error

---

### 4.4 `GET /api/v1/pipeline/{test_id}/summary`

Return the **cached** pipeline result. Returns `409` if `/run` has not been called yet.

**Example:** `GET /api/v1/pipeline/1/summary`

Response schema identical to `POST /run`.

**Error codes:**
- `404` — no models for `test_id`
- `409` — pipeline not yet run for `test_id`

---

### 4.5 `GET /api/v1/pipeline/{test_id}/timeseries`

Return the per-snapshot health score time series from `HealthMonitor.score_dataset()`.
Cached on first access. Includes all component scores per snapshot.

**Example:** `GET /api/v1/pipeline/1/timeseries`

**Response 200 (truncated):**
```json
{
  "test_id": 1,
  "n_snapshots": 2156,
  "fault_start": 1724,
  "disclaimer": "fault_start marks the heuristic temporal label boundary...",
  "snapshots": [
    {
      "snapshot_index":            0,
      "timestamp":                 "2003-11-01 12:10:00",
      "label":                     0,
      "label_name":                "Normal",
      "composite_rms":             0.082341,
      "anomaly_score":             0.051200,
      "anomaly_pred":              0,
      "rf_pred":                   0,
      "prob_normal":               0.9200,
      "prob_faulty":               0.0800,
      "health_rms_component":      93.20,
      "health_anomaly_component":  88.40,
      "health_fault_component":    92.00,
      "health_score":              90.64,
      "health_status":             "Stable",
      "explanation":               "Health is Stable. RMS: normal. Anomaly: low. Fault: low."
    }
  ]
}
```

---

### 4.6 `GET /api/v1/pipeline/{test_id}/fault`

Return the Phase 5 RF fault diagnosis result (final snapshot only).
Requires `/run` to have been called first.

**Response 200:**
```json
{
  "final_rf_pred":      1,
  "predicted_fault":    "Faulty",
  "normal_probability": 0.18,
  "fault_probability":  0.82,
  "fault_confidence":   0.82,
  "label_disclaimer":   "RF trained on heuristic temporal labels..."
}
```

---

### 4.7 `GET /api/v1/pipeline/{test_id}/anomaly`

Return the Phase 6 IF anomaly detection result (final snapshot only).

**Response 200:**
```json
{
  "anomaly_score":       0.45,
  "anomaly_prediction":  1,
  "anomaly_label":       "Anomaly",
  "n_anomaly_predicted": 312,
  "score_semantics":     "decision_scores() = -score_samples(). HIGHER = MORE ANOMALOUS.",
  "label_disclaimer":    "IF trained on Normal-only data..."
}
```

---

### 4.8 `POST /api/v1/rag/query`

Retrieve relevant passages from the bearing knowledge base using vector similarity.

**Requires Ollama running** (`ollama serve`, model: `nomic-embed-text`).
Returns `503` if Ollama is unavailable.

**Request body:**
```json
{
  "query": "What are common bearing outer race fault symptoms?",
  "top_k": 5
}
```

**Response 200:**
```json
{
  "query":  "What are common bearing outer race fault symptoms?",
  "count":  5,
  "results": [
    {
      "content":     "Outer race faults produce periodic impulses at BPFO frequency...",
      "source":      "bearing_faults.md",
      "title":       "Bearing Fault Types",
      "section":     "Outer Race Fault",
      "chunk_index": 0,
      "distance":    0.152,
      "relevance_pct": 92.4
    }
  ],
  "disclaimer":      "Retrieval only. No LLM generation.",
  "embedding_model": "nomic-embed-text (Ollama)"
}
```

**Error codes:**
- `422` — empty query or top_k out of range (1–10)
- `500` — ChromaDB or internal retrieval error
- `503` — Ollama not running

---

## 5. Error Responses

All errors return a consistent JSON body:

```json
{
  "error":  "Human-readable summary.",
  "detail": "Optional additional detail."
}
```

**Errors never expose:**
- Python tracebacks
- Windows filesystem paths
- Internal variable names or module paths

---

## 6. Relationship to Streamlit Dashboard

The Streamlit dashboard (`dashboard/app.py`) and the FastAPI API are **independent consumers** of the same Phase 5–9 ML pipeline.

```
Streamlit dashboard → dashboard/services.py → Phase 5–9 components
FastAPI API        → api/services/          → Phase 5–9 components
```

They run as separate processes on different ports:

| Service | Port | Command |
|---|---|---|
| Streamlit | 8501 | `streamlit run dashboard/app.py` |
| FastAPI   | 8000 | `python -m uvicorn api.main:app --reload` |

**Streamlit does NOT call FastAPI.** Both independently use the same underlying models and classes. This avoids unnecessary HTTP round-trips and keeps the dashboard self-contained.

---

## 7. Model Artifacts and Caching

Model objects (`OrchestratorAgent`, `HealthMonitor`, `KnowledgeRetriever`) are loaded **once on first use** and held in memory for the API server process lifetime.

Pipeline results are cached in memory per `test_id`. Restart the server to reset all caches.

**No models are loaded on API startup.** Model loading is triggered by the first actual request.

---

## 8. Heuristic Label Disclaimers

All fault and anomaly results carry explicit disclaimers in the API response:

- **RF fault labels** are heuristic temporal labels (final 20% of each IMS test run = "Faulty"). They are NOT verified fault-onset timestamps.
- **IF anomaly score semantics:** `decision_scores() = -score_samples()`. HIGHER = MORE ANOMALOUS.
- **Health score** integrates RF probability, IF score, and RMS degradation with fixed weights. It is a research prototype, not a validated safety-critical index.

**These results are NOT suitable for safety-critical decisions.**

---

## 9. CORS Configuration

Default allowed origins (Streamlit local ports):
```
http://localhost:8501
http://127.0.0.1:8501
```

Override via environment variable:
```ini
CORS_ORIGINS=http://localhost:8501,http://myserver.example.com
```

Wildcard `*` is never used by default.

---

## 10. RAG / Ollama Dependency

The RAG endpoint requires Ollama to be running for query embedding:

```powershell
ollama serve
ollama pull nomic-embed-text   # if not already downloaded
```

All other API endpoints (health, pipeline, timeseries, fault, anomaly) work **without Ollama**.

---

## 11. Running Tests

```powershell
# API tests only (no models, no Ollama needed)
python -m pytest tests/test_api.py -v

# Full regression
python -m pytest -q
```
