# Agentic AI-Based Industrial Equipment Health Monitoring and Fault Diagnosis

> M.Tech Academic Prototype | IMS Bearing Dataset | Python 3.9+

---

## Problem Statement

Industrial equipment failures cause costly unplanned downtime and safety risks.
Traditional condition monitoring is reactive — maintenance happens after failure occurs.
This project builds an intelligent, agent-driven bearing health monitoring system
that analyzes vibration sensor data continuously and provides:

- Early fault detection before catastrophic failure
- Equipment health scoring with trend analysis
- Probable root cause identification from sensor evidence
- Risk-stratified maintenance recommendations

---

## Objectives

1. Analyze IMS bearing vibration data end-to-end
2. Extract time-domain and frequency-domain signal features
3. Train a Random Forest fault classifier (Normal vs Faulty)
4. Detect anomalies using Isolation Forest
5. Compute an explainable equipment health score (0–100)
6. Perform degradation analysis (dataset is run-to-failure)
7. Orchestrate 5 AI agents for analysis → diagnosis → recommendation
8. Retrieve relevant maintenance knowledge via RAG (ChromaDB)
9. Present results on a Streamlit dashboard
10. Expose a FastAPI REST endpoint

---

## Dataset

**IMS Bearing Dataset** (NASA/University of Cincinnati)
- Kaggle: [vinayak123tyagi/bearing-dataset](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
- 3 run-to-failure tests on 4 bearings each
- 8-channel vibration (accelerometer) data per snapshot
- Sampled at 20 kHz, 20,480 samples per file (~1 second window)
- Snapshots every ~10 minutes → full degradation timeline
- Known fault outcomes: Outer Race Fault, Roller Element Fault
- **~3,000 files total across 3 tests**

Place downloaded dataset in `archive/` (already done — not committed to Git).

---

## Architecture

```
USER
 │
 ▼
STREAMLIT DASHBOARD
 │
 ▼
ORCHESTRATOR AGENT
 │
 ├──► DATA / HEALTH AGENT ──► Health Score + Degradation Trend
 │
 ├──► FAULT DIAGNOSIS AGENT ──► Random Forest Prediction + Confidence
 │
 ├──► RCA AGENT ──► Root Cause + Evidence + Anomaly Score
 │
 └──► MAINTENANCE AGENT ──► RAG Retrieval + Risk + Recommendation
          │
          ▼
    FINAL HEALTH REPORT
```

Each agent calls actual Python/ML model functions.
LLM (optional) enhances report text — not the analysis itself.

---

## Project Structure

```
Agentic_AI/
│
├── archive/                    # Raw IMS dataset — NOT committed
│
├── data/
│   ├── processed/              # Cleaned, normalized data
│   └── features/               # Extracted feature datasets (CSV)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Dataset exploration
│   ├── 02_preprocessing.ipynb  # Data cleaning & segmentation
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   └── industrial_health/
│       ├── data/
│       │   ├── loader.py       # Read raw IMS files
│       │   └── preprocess.py   # Normalization, labeling, splitting
│       │
│       ├── features/
│       │   └── extractor.py    # Time & frequency domain features
│       │
│       ├── models/
│       │   ├── fault_model.py  # Random Forest classifier
│       │   └── anomaly_model.py # Isolation Forest
│       │
│       ├── health/
│       │   └── health_score.py # Explainable 0-100 health score
│       │
│       ├── agents/
│       │   ├── orchestrator.py
│       │   ├── health_agent.py
│       │   ├── diagnosis_agent.py
│       │   ├── rca_agent.py
│       │   └── maintenance_agent.py
│       │
│       ├── rag/
│       │   └── retriever.py    # ChromaDB vector retrieval
│       │
│       └── pipeline/
│           └── analysis_pipeline.py  # End-to-end pipeline
│
├── knowledge_base/             # Maintenance knowledge docs (committed)
│   ├── bearing_faults.md
│   ├── maintenance_guidelines.md
│   ├── troubleshooting.md
│   └── equipment_conditions.md
│
├── models/                     # Saved model files (not committed)
│
├── reports/
│   └── figures/                # Generated plots
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── tests/                      # Pytest test suite
│
├── scripts/
│   └── setup_venv.ps1          # PowerShell venv setup script
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── methodology.md
│
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aniket181/Agentic_AI.git
cd Agentic_AI
```

### 2. Create and activate the virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venvagentic
venvagentic\Scripts\Activate.ps1
```

**Verify:**
```powershell
python --version
pip --version
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

### 4. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env — add your LLM API key if available
# System works without LLM key (uses deterministic fallback)
```

### 5. Place the dataset

The IMS Bearing Dataset must be in:
```
archive/
├── 1st_test/1st_test/     ← ~983 timestamp files
├── 2nd_test/2nd_test/     ← ~984 timestamp files
├── 3rd_test/4th_test/txt/ ← ~1000+ timestamp files
└── Readme Document for IMS Bearing Data.pdf
```

---

## Usage

### Run EDA Notebook

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

### Train Models

```powershell
python scripts/train_models.py
```

### Run Streamlit Dashboard

```powershell
streamlit run dashboard/app.py
```

### Run FastAPI Server

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Model Evaluation

| Model | Metric | Value |
|-------|--------|-------|
| Random Forest (Fault) | Accuracy | *TBD — run Phase 4* |
| Random Forest (Fault) | F1-Score | *TBD* |
| Isolation Forest (Anomaly) | Precision | *TBD* |

> Results will be filled in after model training (Phase 4–5).
> All values come from actual experiments — never fabricated.

---

## Limitations

1. **No explicit fault labels** — labels are derived from known IMS test outcomes (time-based labeling)
2. **Single operating condition** — IMS dataset uses one speed/load; no multi-condition support yet
3. **RUL is approximated** — expressed as fraction of remaining life, not validated against physics models
4. **Vibration only** — no temperature or current data in this dataset
5. **LLM dependency** — enhanced text explanations require an API key; falls back to template-based reports

---

## Future Work

- Add Paderborn bearing dataset for multi-fault-class classification
- Implement 1D CNN for raw waveform feature learning
- Add LSTM Autoencoder for temporal anomaly detection
- Integrate real-time sensor streaming via MQTT
- Multi-equipment monitoring dashboard

---

## Academic Integrity

All experimental results shown in this project come from actual model training and evaluation.
No results are fabricated or assumed.
Every displayed metric is clearly marked as **MEASURED** or **TBD (pending experiment)**.

---

## License

MIT License — Academic use only.
