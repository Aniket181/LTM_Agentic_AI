# Architecture Documentation

## System Overview

This project implements an **Agentic AI-based industrial bearing health monitoring system**.
The architecture is designed to be simple, modular, and academically understandable.

## Core Design Principle

> ML models perform the actual numerical analysis.
> Agentic AI coordinates the analysis and produces an explainable report.
> The LLM only enhances text — it never fabricates results.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│  Health Score | Status | Fault | RCA | Recommendation       │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls
┌──────────────────────────▼──────────────────────────────────┐
│                   ORCHESTRATOR AGENT                         │
│  Manages state dict, calls agents in sequence               │
└──┬───────────────┬───────────────┬───────────────┬──────────┘
   │               │               │               │
   ▼               ▼               ▼               ▼
Health Agent  Diagnosis Agent   RCA Agent   Maintenance Agent
   │               │               │               │
   ▼               ▼               │               ▼
Isolation      Random Forest      │           ChromaDB RAG
Forest         Classifier        Synthesizes  Knowledge Base
               │                evidence
               ▼
         feature_importance
```

## Agent State Dictionary

All agents share a single Python dictionary (the **state**).
Each agent reads from it and writes its results back.

```python
state = {
    # Input
    "test_id": 1,
    "snapshot_index": -1,

    # Health Agent writes
    "features_df": pd.DataFrame(...),
    "current_snapshot": {...},
    "snapshot_timestamp": "2003-11-25T15:01:00",
    "anomaly_result": {
        "anomaly_score": 0.12,
        "anomaly_status": "ANOMALOUS",
        "is_anomaly": True,
    },
    "rms_current": 0.284,
    "rms_baseline": 0.051,
    "health_result": {
        "health_score": 32.4,
        "status": "CRITICAL",
        "risk_level": "CRITICAL",
        "trend": "DEGRADING",
    },

    # Diagnosis Agent writes
    "fault_result": {
        "fault_class": 1,
        "fault_label": "Faulty",
        "confidence": 0.94,
    },
    "feature_importance": [...],

    # RCA Agent writes
    "rca_result": {
        "observed_evidence": [...],
        "probable_cause": "...",
        "probable_interpretation": "...",
    },

    # Maintenance Agent writes
    "maintenance_recommendation": {...},
    "final_report": "...",

    # Metadata
    "status": "COMPLETED",
    "errors": [],
}
```

## ML Models

### Fault Diagnosis: Random Forest
- **Input:** Feature vector (per-snapshot statistical features)
- **Output:** fault_class (0=Normal, 1=Faulty), confidence, probabilities
- **Training:** Time-based split — first 70% of chronological data
- **Evaluation:** Accuracy, Precision, Recall, F1, Confusion Matrix

### Anomaly Detection: Isolation Forest
- **Input:** Feature vector (same as above)
- **Training:** Normal-period data only (unsupervised)
- **Output:** Normalized anomaly score (0=anomalous, 1=normal)
- **No labels needed** for anomaly model training

### Health Score
- **Formula:** 0.40 × anomaly + 0.40 × fault + 0.20 × degradation
- **Range:** 0–100 (higher = healthier)
- **Thresholds:** Configurable in .env

## Data Flow

```
archive/ (raw IMS files)
    │
    ▼ loader.py
Raw DataFrame (20480 rows × 8 cols per file)
    │
    ▼ extractor.py
Feature vector (13 features × channels per file = ~104 features)
    │
    ▼ preprocess.py
Labeled + normalized feature dataset (one row per 10-min snapshot)
    │
    ├──▶ fault_model.py (Random Forest)
    └──▶ anomaly_model.py (Isolation Forest)
              │
              ▼
         health_score.py
              │
              ▼
         Agents → Report
```

## RAG Architecture

```
knowledge_base/*.md (4 markdown files)
    │
    ▼ ChromaDB ingestion (sentence-transformers, local)
Persistent Vector Store (chroma_db/)
    │
    ▼ Query: "bearing outer race fault high risk maintenance"
Top-3 relevant chunks
    │
    ▼ Maintenance Agent
Enhanced recommendation
```

## Upgrade Path to LangGraph

The current state-dict architecture is directly convertible to LangGraph:

```python
# Current: sequential Python calls
state = orchestrator.run(test_id=1)

# Future LangGraph equivalent:
from langgraph.graph import StateGraph
graph = StateGraph(AgentState)
graph.add_node("health_agent", health_agent.run)
graph.add_node("diagnosis_agent", diagnosis_agent.run)
graph.add_edge("health_agent", "diagnosis_agent")
# ... etc
```

The upgrade requires minimal code changes since state management is already clean.
