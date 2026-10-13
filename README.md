# Agentic AI-Based Industrial Equipment Health Monitoring and Fault Diagnosis

> **An integrated, explainable decision-support system combining high-frequency vibration signal processing, supervised fault classification, unsupervised anomaly detection, composite health scoring, deterministic multi-agent orchestration, and local retrieval-augmented maintenance intelligence.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11%20%7C%203.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: FastAPI](https://img.shields.io/badge/API-FastAPI%20v0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%20v1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Vector DB: ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20v0.4+-brightgreen.svg)](https://www.trychroma.com/)
[![Embeddings: Ollama](https://img.shields.io/badge/Embeddings-Ollama%20(nomic--embed--text)-orange.svg)](https://ollama.ai/)
[![Dataset: NASA IMS](https://img.shields.io/badge/Dataset-NASA%20IMS%20Bearing-lightgrey.svg)](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
  - [Problem Statement](#problem-statement)
  - [Proposed Solution](#proposed-solution)
  - [Target Audience](#target-audience)
- [2. Key Features](#2-key-features)
- [3. System Architecture](#3-system-architecture)
  - [Data and Execution Flow](#data-and-execution-flow)
  - [Core Subsystems](#core-subsystems)
- [4. Technology Stack](#4-technology-stack)
- [5. Repository Structure](#5-repository-structure)
- [6. Dataset and Data Preparation](#6-dataset-and-data-preparation)
  - [The IMS Bearing Dataset](#the-ims-bearing-dataset)
  - [Vibration Feature Extraction Pipeline](#vibration-feature-extraction-pipeline)
  - [Chronological Split and Heuristic Labeling Protocol](#chronological-split-and-heuristic-labeling-protocol)
- [7. Machine Learning Models and Health Analytics](#7-machine-learning-models-and-health-analytics)
  - [Fault Diagnosis (Random Forest)](#fault-diagnosis-random-forest)
  - [Anomaly Detection (Isolation Forest)](#anomaly-detection-isolation-forest)
  - [Composite Equipment Health Scoring and Trend Analysis](#composite-equipment-health-scoring-and-trend-analysis)
- [8. Deterministic Agentic AI Workflow](#8-deterministic-agentic-ai-workflow)
- [9. Retrieval-Augmented Knowledge Base (RAG)](#9-retrieval-augmented-knowledge-base-rag)
- [10. Interactive Streamlit Dashboard](#10-interactive-streamlit-dashboard)
- [11. FastAPI REST API Service](#11-fastapi-rest-api-service)
- [12. Prerequisites and Installation](#12-prerequisites-and-installation)
- [13. Running the Project](#13-running-the-project)
- [14. API Usage Examples](#14-api-usage-examples)
- [15. Verification and Testing](#15-verification-and-testing)
- [16. Experimental Results](#16-experimental-results)
- [17. Limitations and Future Work](#17-limitations-and-future-work)
- [18. Security and Data Governance](#18-security-and-data-governance)
- [19. Troubleshooting](#19-troubleshooting)
- [20. References and Acknowledgments](#20-references-and-acknowledgments)

---

## 1. Project Overview

### Problem Statement

In heavy industries, rotary machinery—specifically rolling-element bearings—operates under demanding mechanical stresses. Unplanned bearing failures lead to catastrophic equipment breakdown, costly emergency outages, secondary structural damage, and occupational safety hazards. Traditional condition-monitoring systems generally rely on manual vibration inspections or rigid, static alarm thresholds (such as raw RMS acceleration triggers). These legacy approaches suffer from critical operational limitations:

1. **Reactive or Late Warnings:** Static threshold alarms trigger only after significant surface spalling or mechanical damage has already occurred.
2. **High False Alarm Rates:** Sensor spikes, mechanical transient surges, and noise frequently trip threshold alarms without underlying mechanical faults.
3. **Siloed Diagnostics:** Sensor readings, anomaly scores, and domain maintenance knowledge are kept in disconnected silos, requiring extensive manual correlation by senior maintenance personnel.
4. **Lack of Actionable Insights:** Alarm systems indicate *that* a metric breached a threshold, but do not provide contextualized probable root causes or equipment-specific troubleshooting protocols.

### Proposed Solution

This project presents an integrated, end-to-end **Agentic AI Industrial Equipment Health Monitoring and Fault Diagnosis** decision-support system. Developed using NASA/University of Cincinnati run-to-failure bearing vibration data, the system unites statistical signal processing, supervised machine learning, unsupervised anomaly detection, composite health scoring, deterministic multi-agent orchestration, and localized retrieval-augmented generation (RAG).

Rather than relying on unconstrained generative language models to perform numerical diagnoses, the system utilizes a **deterministic multi-agent pipeline** where specialized agents evaluate feature-engineered vibration signals using validated scikit-learn models. A local RAG system powered by ChromaDB and Ollama then pairs detected anomalies with peer-reviewed maintenance protocols and failure-mode catalogs, presenting synthesized findings through both an interactive Streamlit engineering dashboard and a versioned FastAPI REST service.

```
Raw Vibration Waveforms (20 kHz)
       │
       ▼
Time- & Frequency-Domain Feature Extraction (104 features)
       │
       ▼
Deterministic Agentic Orchestration
 ├── DataHealthAgent     ──► Health Score (0–100) & Degradation Trend
 ├── FaultDiagnosisAgent ──► Random Forest Prediction (Normal vs Faulty)
 └── AnomalyAgent        ──► Isolation Forest Baseline Deviation Score
       │
       ▼
Retrieval-Augmented Maintenance Intelligence (ChromaDB + Ollama)
       │
       ▼
Presentation & Integration (Streamlit Dashboard & FastAPI REST API)
```

### Target Audience

- **Condition-Monitoring Engineers:** To evaluate real-time and historical vibration feature trajectories, spectral energy distribution, and statistical metrics across bearing channels.
- **Industrial Maintenance & Reliability Engineers:** To triage health degradation warnings, review automated fault probabilities, and access context-grounded maintenance guidelines before dispatching maintenance crews.
- **Academic Researchers & ML Practitioners:** To benchmark chronological machine learning workflows, evaluate leak-free splitting methodologies on run-to-failure machinery datasets, and inspect deterministic agentic architectures.

---

## 2. Key Features

- **Continuous Vibration Signal Analytics:** Ingests raw accelerometer sensor snapshots (20,480 samples/file sampled at 20 kHz) from 8 channels across 4 dual-row bearings.
- **Comprehensive Dual-Domain Feature Extraction:** Computes 10 time-domain statistical metrics (RMS, peak-to-peak, kurtosis, crest factor, skewness, shape factor, etc.) and 3 frequency-domain spectral metrics (dominant frequency, spectral energy, spectral centroid) per channel, generating a 104-feature vector per snapshot for Test 1.
- **Supervised Fault Classification:** Implements an optimized Random Forest classifier to identify bearing fault states, incorporating class-balanced weighting to handle severe operational class imbalances.
- **Baseline Unsupervised Anomaly Detection:** Utilizes an Isolation Forest trained exclusively on the healthy baseline regime (first 70% of the normal operational period) to detect behavioral degradation without prior exposure to damaged states.
- **Explainable Equipment Health Score:** Calculates a unified, transparent health metric (0–100 scale) synthesized from RMS degradation, anomaly severity, and fault classification confidence with chronological trend analysis (Stable, Degrading, Critical).
- **Deterministic Multi-Agent Orchestration:** Coordinates specialized execution agents (`DataHealthAgent`, `FaultDiagnosisAgent`, and `AnomalyAgent`) through a state-preserving `OrchestratorAgent` implemented in pure Python.
- **Local Retrieval-Augmented Generation (RAG):** Integrates ChromaDB with local Ollama embeddings (`nomic-embed-text`) across 66 indexed Markdown chunks containing industrial maintenance guides, bearing fault physics, and troubleshooting procedures.
- **Interactive Multi-View Streamlit Dashboard:** Features 7 operational views (Executive Dashboard, Equipment Health, Fault Diagnosis, Anomaly Detection, Maintenance Insights, Agent Activity, and Alerts/Events) plus custom CSV data upload and validation.
- **Production-Ready FastAPI REST Service:** Exposes versioned (`/api/v1/`) endpoints for pipeline execution, time-series telemetry retrieval, model inferences, and knowledge queries, fully documented via OpenAPI Swagger UI (`/docs`).

---

## 3. System Architecture

### Data and Execution Flow

The system implements a strictly decoupled pipeline architecture, ensuring that data processing, machine learning inference, agent orchestration, and presentation layers remain independent.

```mermaid
flowchart TD
    subgraph DataIngestion["1. Data Ingestion & Signal Processing"]
        RAW["Raw Vibration Files\n(20 kHz, 20,480 samples/snapshot)"]
        EXTRACT["Feature Extractor\n(src/industrial_health/features/extractor.py)\n10 Time-Domain + 3 Spectral Features"]
        CSV["Feature Store\n(data/features/test1_features.csv)\n2,156 Snapshots x 104 Features"]
        RAW --> EXTRACT --> CSV
    end

    subgraph Preprocessing["2. Leakage-Safe Preprocessing"]
        SPLIT["Stratified Chronological Partitioning\n(Train: 1,508 | Test: 648)"]
        SCALER["StandardScaler\n(Fitted strictly on Train partition)"]
        CSV --> SPLIT --> SCALER
    end

    subgraph Analytics["3. Machine Learning & Health Scoring"]
        RF["Random Forest Classifier\n(models/random_forest_test1.joblib)"]
        IF["Isolation Forest Detector\n(models/isolation_forest_test1.joblib)"]
        HM["Health Monitor & Trend Engine\n(src/industrial_health/health/health_monitor.py)"]
        SCALER --> RF
        SCALER --> IF
        RF & IF --> HM
    end

    subgraph Agents["4. Deterministic Multi-Agent Orchestrator"]
        ORCH["OrchestratorAgent\n(src/industrial_health/agents/orchestrator.py)"]
        STATE[("AgentState (Shared Data)") ]
        A1["DataHealthAgent"]
        A2["FaultDiagnosisAgent"]
        A3["AnomalyAgent"]

        ORCH --> A1 --> STATE
        STATE --> A2 --> STATE
        STATE --> A3 --> STATE
        HM -.-> A1
        RF -.-> A2
        IF -.-> A3
    end

    subgraph Knowledge["5. Retrieval-Augmented Knowledge Base (RAG)"]
        DOCS["Domain Markdown Docs\n(knowledge_base/*.md)"]
        CHUNKER["Heading-Aware Chunker\n(66 chunks, max 800 chars)"]
        EMBED["Local Ollama Embeddings\n(nomic-embed-text)"]
        CHROMA[("ChromaDB Vector Store\n(chroma_db/)")]
        RETRIEVER["KnowledgeRetriever\n(Cosine Distance Ranking)"]

        DOCS --> CHUNKER --> EMBED --> CHROMA --> RETRIEVER
    end

    subgraph Presentation["6. Presentation & Delivery Layers"]
        FASTAPI["FastAPI REST API\n(api/main.py @ Port 8000)\n/api/v1/pipeline, /api/v1/rag"]
        STREAMLIT["Streamlit Industrial Dashboard\n(dashboard/app.py @ Port 8501)\n7 Dedicated Pages + CSV Upload"]

        STATE --> FASTAPI
        STATE --> STREAMLIT
        RETRIEVER --> FASTAPI
        RETRIEVER --> STREAMLIT
    end
```

### Core Subsystems

1. **Feature Engineering Engine (`src/industrial_health/features/`):** Processes raw vibration arrays into high-density statistical indicators. Eliminates raw waveform storage overhead while retaining critical indicators such as kurtosis spikes and high-frequency spectral shifts.
2. **Classification & Anomaly Engine (`src/industrial_health/models/`):** Hosts the scikit-learn model definitions, hyperparameter specifications, and chronological splitters. Prevents test data leakage by isolating scaler fitting to the training partition.
3. **Health Integration Layer (`src/industrial_health/health/`):** Merges multi-model inferences with physical vibration metrics (RMS) into a bounded, explainable 0–100 equipment health score and tracks rolling degradation slopes.
4. **Agent Orchestration Framework (`src/industrial_health/agents/`):** Manages a deterministic state-machine workflow where agents run in strict sequence, validating upstream results before executing downstream steps.
5. **RAG Knowledge Retriever (`src/industrial_health/rag/`):** Implements a chunked vector retrieval pipeline backed by ChromaDB and local Ollama embeddings, enabling semantic search over maintenance manuals without generative hallucinations.
6. **Delivery & Interface Layer (`api/` and `dashboard/`):** Exposes independent presentation services. The Streamlit dashboard and FastAPI backend consume the same underlying Python services independently without cross-service dependencies.

---

## 4. Technology Stack

| Layer / Component | Technology | Version | Purpose in Implementation |
|---|---|---|---|
| **Programming Language** | Python | `3.11` (compatible `3.9+`) | Core execution runtime for ML pipelines, APIs, and dashboard |
| **Numerical Processing** | NumPy | `^1.24.0` | High-performance array computing, signal processing, and FFT |
| **Data Manipulation** | Pandas | `^2.0.0` | Chronological time-series management and tabular feature extraction |
| **Scientific Computing** | SciPy | `^1.10.0` | Statistical feature calculations (kurtosis, skewness, signal integrals) |
| **Machine Learning** | Scikit-learn | `^1.3.0` | Random Forest classifier, Isolation Forest anomaly detector, StandardScaler |
| **Model Serialization** | Joblib | `^1.3.0` | Serialization and persistence of trained ML models and scalers |
| **Vector Database** | ChromaDB | `^0.4.0` | Local persistent vector storage for maintenance documentation chunks |
| **Embedding Generation** | Ollama (`nomic-embed-text`) | `^0.1.0` | 768-dimensional local embeddings via HTTP (`http://localhost:11434`) |
| **Web API Backend** | FastAPI & Uvicorn | `^0.104.0` / `^0.24.0` | REST API service with OpenAPI documentation and Pydantic validation |
| **Schema Validation** | Pydantic | `^2.4.0` | Strict data validation and response sanitization across API layers |
| **Engineering Dashboard** | Streamlit | `^1.28.0` | Multi-page industrial monitoring dashboard with dark theme UI |
| **Interactive Plotting** | Plotly | `^5.17.0` | Dynamic time-series plots, health degradation curves, and KPI charts |
| **Static Visualizations** | Matplotlib & Seaborn | `^3.7.0` / `^0.12.0` | Headless generation of experiment reports and confusion matrix plots |
| **Logging & Tracing** | Loguru | `^0.7.0` | Structured, diagnostic logging across pipeline execution steps |
| **Testing Framework** | Pytest & Pytest-Cov | `^7.4.0` / `^4.1.0` | Automated test runner for unit, integration, and regression suites |

---

## 5. Repository Structure

```
Agentic_AI/
├── .env.example                     # Environment configuration template (tracked)
├── .gitignore                        # Git exclusion rules (tracked)
├── pyproject.toml                   # Project metadata and packaging specification
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Master project documentation
│
├── api/                             # Phase 11: FastAPI REST API implementation
│   ├── config.py                    # API path resolution and configuration
│   ├── dependencies.py              # Service dependency injection providers
│   ├── main.py                      # FastAPI application factory and route registration
│   ├── schemas.py                   # Pydantic v2 request/response schemas
│   ├── routes/                      # Route handlers
│   │   ├── health.py                # Liveness & artifact verification (/health)
│   │   ├── pipeline.py              # Pipeline execution & summaries (/api/v1/pipeline/*)
│   │   └── rag.py                   # Knowledge retrieval endpoint (/api/v1/rag/*)
│   └── services/                    # API business logic wrapping core modules
│
├── archive/                         # Raw IMS Bearing Dataset (gitignored; local only)
│   ├── 1st_test/1st_test/           # 2,156 raw ASCII vibration snapshot files
│   ├── 2nd_test/2nd_test/           # 984 raw ASCII vibration snapshot files
│   └── 3rd_test/4th_test/txt/       # ~1,000 raw ASCII vibration snapshot files
│
├── chroma_db/                       # ChromaDB persistent vector database (gitignored; local only)
│
├── dashboard/                       # Phase 10: Streamlit Engineering Dashboard
│   ├── app.py                       # Application shell and sidebar navigation
│   ├── config.py                    # UI configuration and artifact discovery
│   ├── services.py                  # Direct bindings to Phase 5–9 pipeline layers
│   ├── upload_service.py            # User CSV upload validation and processing
│   ├── components/                  # Reusable Streamlit UI widgets
│   └── views/                       # 7 operational dashboard views
│       ├── v1_dashboard.py          # Executive overview & KPIs
│       ├── v2_equipment_health.py   # Health trajectory & degradation bands
│       ├── v3_fault_diagnosis.py    # Random Forest fault predictions & feature importances
│       ├── v4_anomaly_detection.py  # Isolation Forest anomaly score progression
│       ├── v5_maintenance_insights.py# Local RAG knowledge retrieval interface
│       ├── v6_agent_activity.py     # Multi-agent execution logs & latency telemetry
│       └── v7_alerts.py             # Chronological alert stream & degradation milestones
│
├── data/                            # Processed data and engineered feature storage
│   ├── features/                    # Extracted feature CSVs (gitignored; generated)
│   │   ├── test1_features.csv       # Test 1 features (2,156 snapshots x 104 features)
│   │   ├── test2_features.csv       # Test 2 features (984 snapshots x 52 features)
│   │   └── test3_features.csv       # Test 3 features (1,000 snapshots x 52 features)
│   └── processed/                   # Preprocessed train/test chronological partitions
│
├── docs/                            # Architectural specifications & phase documentation
│   ├── agentic_architecture.md      # Multi-agent design and shared state specification
│   ├── anomaly_detection.md         # Phase 6 Isolation Forest methodology and results
│   ├── api.md                       # Phase 11 REST API reference manual
│   ├── architecture.md              # System design and component interactions
│   ├── dashboard.md                 # Dashboard architecture and view designs
│   ├── dataset.md                   # IMS dataset characteristics and sensor mappings
│   ├── eda.md                       # Exploratory data analysis insights
│   ├── fault_diagnosis.md           # Phase 5 Random Forest classification report
│   ├── health_monitoring.md         # Phase 7 composite scoring formulas and thresholds
│   ├── methodology.md               # Scientific methodology and experimental protocols
│   ├── preprocessing.md             # Chronological partitioning and leakage controls
│   └── rag_knowledge_base.md        # Phase 9 vector store chunking and retrieval docs
│
├── knowledge_base/                  # Industrial maintenance knowledge documents (tracked)
│   ├── bearing_faults.md            # Physics of BPFO, BPFI, BSF, FTF fault signatures
│   ├── equipment_conditions.md      # Equipment operating regimes and baseline limits
│   ├── maintenance_guidelines.md    # ISO 10816 vibration severity and lubrication protocols
│   └── troubleshooting.md          # Step-by-step diagnostic workflows for maintenance teams
│
├── models/                          # Trained model artifacts and scalers (gitignored; local only)
│   ├── isolation_forest_test1.joblib# Phase 6 Isolation Forest model
│   ├── random_forest_test1.joblib   # Phase 5 Random Forest classifier
│   └── scaler/                      # Fitted StandardScaler instances
│       ├── phase5_test1_scaler.pkl  # Scaler fitted strictly on RF train partition
│       ├── phase6_test1_scaler.pkl  # Scaler fitted strictly on IF normal partition
│       └── test1_scaler.pkl         # Phase 4 general preprocessed scaler
│
├── reports/                         # Experiment reports and evaluation figures
│   ├── phase5_experiment_test1.json # Tracked RF experiment metrics and feature importances
│   ├── phase6_experiment_test1.json # Tracked IF anomaly detection evaluation results
│   ├── phase7_experiment_test1.json # Tracked composite health score evaluation report
│   ├── phase8_experiment_test1.json # Tracked agentic pipeline execution log and results
│   └── figures/                     # Headless experiment charts (.png files gitignored)
│
├── scripts/                         # Command-line pipelines and reproduction utilities
│   ├── build_knowledge_base.py      # Chunks and upserts knowledge base into ChromaDB
│   ├── run_agentic_pipeline.py      # Executes the Phase 8 multi-agent pipeline
│   ├── run_anomaly_detection.py     # Fits and evaluates Phase 6 Isolation Forest
│   ├── run_eda.py                   # Runs Phase 2 exploratory data analysis
│   ├── run_fault_diagnosis.py       # Trains and evaluates Phase 5 Random Forest
│   ├── run_feature_extraction.py    # Extracts 104 signal features across raw IMS tests
│   ├── run_health_monitoring.py     # Computes Phase 7 health score time series
│   ├── run_preprocessing.py         # Performs Phase 4 chronological data splitting
│   ├── setup_venv.ps1               # Automated PowerShell environment configuration
│   └── train_pipeline.py            # End-to-end training pipeline wrapper
│
├── src/industrial_health/           # Core library source code
│   ├── agents/                      # Deterministic agent implementations
│   │   ├── anomaly_agent.py         # Wraps Isolation Forest anomaly inference
│   │   ├── data_health_agent.py     # Wraps feature loading and health scoring
│   │   ├── fault_diagnosis_agent.py # Wraps Random Forest fault inference
│   │   ├── orchestrator.py          # State-preserving pipeline coordinator
│   │   └── state.py                 # Dataclass defining the shared agent execution state
│   ├── data/                        # Raw data loaders and preprocessing routines
│   ├── features/                    # Feature extraction functions (time and spectral)
│   ├── health/                      # Health scoring and rolling trend analysis
│   ├── models/                      # Machine learning model class wrappers
│   ├── pipeline/                    # High-level training and evaluation pipelines
│   └── rag/                         # Document chunkers, vector store, and retriever
│
└── tests/                           # Comprehensive Pytest automated test suite
    ├── test_agents.py               # Tests for multi-agent coordination and state transitions
    ├── test_anomaly.py              # Tests for Isolation Forest model and scoring
    ├── test_api.py                  # Full test suite for FastAPI REST endpoints
    ├── test_dashboard.py            # Integration tests for Streamlit dashboard services
    ├── test_features.py             # Tests for time- and frequency-domain feature extractors
    ├── test_health.py               # Tests for health scoring logic and edge cases
    ├── test_health_score.py         # Tests for component weights and bounds validation
    ├── test_loader.py               # Tests for raw ASCII file reading and shape validation
    ├── test_model.py                # Tests for Random Forest model fitting and metrics
    ├── test_preprocessing.py        # Tests for chronological splitting and leakage prevention
    └── test_rag.py                  # Tests for document chunking, ChromaDB, and retrieval
```

---

## 6. Dataset and Data Preparation

### The IMS Bearing Dataset

The project utilizes the benchmark **IMS Bearing Dataset**, generated by the University of Cincinnati Intelligent Maintenance Systems (IMS) Center under NASA research sponsorship (available on [Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)).

The experimental setup comprised a specialized test rig hosting four Rexnord ZA-2115 double-row ball bearings mounted on a single shaft driven by an AC motor at a constant 2,000 RPM. A radial load of 6,000 lbs was continuously applied via a spring mechanism. Lubrication was provided via a circulating oil system. High-sensitivity PCB 353B33 quartz ICP accelerometers were mounted on the bearing housings.

Data was captured in periodic 1-second snapshots every 10 minutes at a 20,480 Hz sampling rate, yielding 20,480 discrete acceleration readings (in $g$) per channel per file.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ IMS Bearing Dataset Test Matrix                                                 │
├──────┬────────────┬─────────────┬──────────┬──────────┬─────────────────────────┤
│ Test │ Duration   │ Snapshots   │ Channels │ Sensors  │ Confirmed Failure Mode  │
├──────┼────────────┼─────────────┼──────────┼──────────┼─────────────────────────┤
│ 1    │ ~35 Days   │ 2,156 files │ 8 ch     │ 2 / brg  │ Brg 3: Outer Race Fault │
│      │            │             │          │          │ Brg 4: Roller Element   │
│ 2    │ ~7 Days    │ 984 files   │ 4 ch     │ 1 / brg  │ Brg 1: Outer Race Fault │
│ 3    │ ~31 Days   │ ~1,000 files│ 4 ch     │ 1 / brg  │ Brg 3: Outer Race Fault │
└──────┴────────────┴─────────────┴──────────┴──────────┴─────────────────────────┘
```

> **Scope Verification Note:** While feature extraction scripts support all three tests, the complete trained model suite (`models/random_forest_test1.joblib`, `models/isolation_forest_test1.joblib`), health monitor, and agentic orchestration pipeline are fully implemented and verified for **IMS Test 1**. Tests 2 and 3 feature files exist for exploratory analysis, but do not host trained models in the active inference pipeline.

### Vibration Feature Extraction Pipeline

Raw vibration waveforms (20,480 floating-point numbers per channel) are converted into compact statistical feature vectors by [`src/industrial_health/features/extractor.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/features/extractor.py). For each 1-second snapshot, 13 features are calculated per channel:

#### Time-Domain Features (10 per channel)
- **Mean:** DC bias of the signal ($\mu = \frac{1}{N}\sum x_i$)
- **Root Mean Square (RMS):** Total vibrational energy ($\text{RMS} = \sqrt{\frac{1}{N}\sum x_i^2}$)
- **Standard Deviation ($\sigma$):** AC signal variance
- **Variance ($\sigma^2$):** Second central moment
- **Kurtosis:** Impulsiveness and tail heaviness ($\text{Kurtosis} = \frac{\frac{1}{N}\sum (x_i - \mu)^4}{\sigma^4} - 3$)
- **Skewness:** Signal asymmetry ($\text{Skewness} = \frac{\frac{1}{N}\sum (x_i - \mu)^3}{\sigma^3}$)
- **Peak:** Maximum absolute vibration amplitude ($\max |x_i|$)
- **Peak-to-Peak:** Dynamic amplitude range ($\max(x_i) - \min(x_i)$)
- **Crest Factor:** Ratio of peak amplitude to RMS ($\frac{\text{Peak}}{\text{RMS}}$), highly sensitive to early impulsive spalling
- **Shape Factor:** Ratio of RMS to mean absolute value ($\frac{\text{RMS}}{\frac{1}{N}\sum |x_i|}$)

#### Frequency-Domain Features via FFT (3 per channel)
- **Dominant Frequency:** Spectral peak index with highest magnitude
- **Spectral Energy:** Cumulative sum of squared FFT magnitude bins ($\sum |X(f)|^2$)
- **Spectral Centroid:** Amplitude-weighted center of gravity of the spectrum ($\frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$)

For Test 1, across 8 channels (Bearings 1–4, Channels 1–2), this produces:
$$\text{Total Features} = 8 \text{ channels} \times 13 \text{ features/channel} = 104 \text{ features per snapshot}$$

### Chronological Split and Heuristic Labeling Protocol

The IMS Bearing Dataset is an unannotated run-to-failure endurance test. The physical apparatus was operated until severe vibration tripped the machine, without micro-level timestamps recording the exact initial moment of microscopic fatigue pitting.

To support supervised classification and anomaly evaluation, the system adopts the standard academic heuristic protocol:
- **Normal Period:** The first 80% of the timeline (snapshots 0 to 1,723) is assigned the class label `0 (Normal)`.
- **Faulty Period:** The final 20% of the timeline (snapshots 1,724 to 2,155) is assigned the class label `1 (Faulty)`.

> [!IMPORTANT]
> **Heuristic Labeling Disclaimer:** The binary Normal/Faulty labels are heuristic temporal labels based on run progression, not metallurgical ground-truth fault onset annotations. Model evaluation metrics reflect agreement with this heuristic operational boundary and must be interpreted as experimental performance rather than certified industrial diagnostic accuracy.

To evaluate models without future-data leakage:
- **Random shuffling is strictly prohibited.**
- For supervised classification (Phase 5), a **stratified chronological split** partitions the within-class data chronologically: the first 70% of the Normal period and the first 70% of the Faulty period constitute the training set (1,508 snapshots); the remaining 30% of each period constitutes the unseen evaluation set (648 snapshots).
- The `StandardScaler` is fitted **strictly on the training partition** and applied via `transform()` to the test partition.

---

## 7. Machine Learning Models and Health Analytics

### Fault Diagnosis (Random Forest)

Implemented in [`src/industrial_health/models/fault_classifier.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/models/fault_classifier.py) and trained via [`scripts/run_fault_diagnosis.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/scripts/run_fault_diagnosis.py).

- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`, `random_state=42`).
- **Feature Vector:** 104 standardized features.
- **Protocol:** Stratified chronological split on Test 1 (1,508 train, 648 test).
- **Verified Evaluation Results (on 648 unseen test snapshots):**
  - **Accuracy:** `83.18%` (0.83179)
  - **Weighted F1-Score:** `84.67%` (0.84674)
  - **Macro F1-Score:** `79.35%` (0.79351)
  - **Faulty Class Recall:** `100.0%` (Recall = 1.0, 130 of 130 faulty snapshots correctly identified)
  - **Faulty Class Precision:** `54.39%` (109 early normal snapshots flagged as faulty)
  - **Normal Class Precision:** `100.0%` (Precision = 1.0, zero faulty snapshots misclassified as normal)
  - **Confusion Matrix:**
    $$\begin{pmatrix} \text{True Normal} = 409 & \text{False Faulty} = 109 \\ \text{False Normal} = 0 & \text{True Faulty} = 130 \end{pmatrix}$$
- **Top Predictive Features:** The top 6 features by Gini importance all originate from Bearing 4 (which suffered roller element failure): `bearing4_ch2__std` (11.69%), `bearing4_ch1__std` (9.09%), `bearing4_ch1__variance` (8.71%), `bearing4_ch2__variance` (8.44%), `bearing4_ch2__rms` (7.49%), and `bearing4_ch1__rms` (5.69%).

> [!NOTE]
> Probabilities output by `predict_proba()` represent tree-vote frequencies. They have not undergone isotonic calibration or Platt scaling and should not be interpreted as calibrated Bayesian posterior probabilities of failure.

### Anomaly Detection (Isolation Forest)

Implemented in [`src/industrial_health/models/anomaly_detector.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/models/anomaly_detector.py) and evaluated via [`scripts/run_anomaly_detection.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/scripts/run_anomaly_detection.py).

- **Algorithm:** Isolation Forest (`n_estimators=200`, `contamination='auto'`, `random_state=42`).
- **Training Paradigm:** **Healthy Baseline Training.** The model is fitted exclusively on the first 70% of the Normal operational period (1,206 snapshots), exposing it only to healthy bearing behavior.
- **Score Semantics:** Isolation Forest decision scores are inverted:
  $$\text{anomaly\_score} = -\text{score\_samples}(X)$$
  *Higher scores represent greater deviation from the healthy baseline.* Predictions are mapped to `1 (Anomaly)` or `0 (Normal)`.
- **Verified Evaluation Results (on 950 unseen snapshots: 518 normal, 432 faulty):**
  - **Accuracy:** `81.68%` (0.81684)
  - **Weighted F1-Score:** `81.37%` (0.81371)
  - **Macro F1-Score:** `81.53%` (0.81526)
  - **Faulty Class Recall:** `100.0%` (432 of 432 faulty snapshots correctly flagged as anomalies)
  - **Normal Class Precision:** `100.0%` (Zero faulty snapshots misidentified as normal baseline)
  - **Confusion Matrix:**
    $$\begin{pmatrix} \text{True Normal} = 344 & \text{False Anomaly} = 174 \\ \text{False Normal} = 0 & \text{True Anomaly} = 432 \end{pmatrix}$$

> [!NOTE]
> The anomaly score reflects topological isolation distance in feature space, not a probability of imminent mechanical collapse.

### Composite Equipment Health Scoring and Trend Analysis

Implemented in [`src/industrial_health/health/health_monitor.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/health/health_monitor.py).

The Health Monitoring layer is a **deterministic integration engine** (no ML model is fitted). It synthesizes physical signal trends, anomaly severity, and classification output into an intuitive index from `0` (failed) to `100` (optimal):

$$\text{Health Score} = \text{clip}\Big(100 \times \big(w_{\text{rms}} \cdot C_{\text{rms}} + w_{\text{anomaly}} \cdot C_{\text{anomaly}} + w_{\text{fault}} \cdot C_{\text{fault}}\big),\, 0,\, 100\Big)$$

Where components are normalized on $[0, 1]$ such that $1 = \text{healthy}$ and $0 = \text{severely degraded}$:
- **$C_{\text{rms}}$ (RMS Component):** Computed against the baseline distribution of the first 70% of normal snapshots.
- **$C_{\text{anomaly}}$ (Anomaly Component):** Normalized inverse of the Isolation Forest anomaly score ($1 - \frac{\text{score}}{\text{ref\_max}}$).
- **$C_{\text{fault}}$ (Fault Component):** Probability of the machine belonging to the Normal class ($1 - P(\text{Faulty})$).

#### Configured Weights and Status Boundaries

- **Weights:** $w_{\text{rms}} = 0.20$, $w_{\text{anomaly}} = 0.40$, $w_{\text{fault}} = 0.40$ ($\sum w = 1.0$)
- **Status Thresholds:**
  - **Stable:** $\text{Health Score} \ge 70.0$
  - **Degrading:** $40.0 \le \text{Health Score} < 70.0$
  - **Critical:** $\text{Health Score} < 40.0$

#### Rolling Trend Analysis

[`src/industrial_health/health/trend.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/health/trend.py) computes a per-snapshot trend label using a rolling linear regression slope over a 50-snapshot window ($\sim 8.3$ hours):
- $\text{Slope} < -0.5 \implies \text{Degrading}$
- $\text{Slope} > 0.5 \implies \text{Improving}$
- Otherwise $\implies \text{Stable}$

#### Verified Test 1 Run-Level Trajectory
- **Mean Health Score:** `43.43`
- **Median Health Score:** `55.32`
- **Maximum Health Score:** `70.73`
- **Final Health Score (Snapshot 2,155):** `0.53` (Status: **Critical**)
- **First Sustained Degradation Trigger:** Snapshot `1624` (matches early physical degradation prior to heuristic cutoff 1724)
- **Status Distribution:** Stable: 7 snapshots ($0.3\%$), Degrading: 1,443 snapshots ($66.9\%$), Critical: 706 snapshots ($32.7\%$)
- **Overall Trajectory:** **Degrading** (Q1 Mean: `59.02` $\to$ Q4 Mean: `1.18`, $\Delta = -57.84$)

---

## 8. Deterministic Agentic AI Workflow

The agentic framework is implemented in [`src/industrial_health/agents/`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/agents/).

### Architectural Clarification

> [!IMPORTANT]
> The active multi-agent pipeline is **deterministic Python-based state-machine orchestration**. It does **not** rely on unconstrained autonomous LLM reasoning loops, stochastic tool selection, or LangGraph. Each agent is a specialized Python class with explicit inputs, deterministic logic, and strongly-typed outputs.

### Orchestration Sequence

The [`OrchestratorAgent`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/agents/orchestrator.py) coordinates execution through a shared [`AgentState`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/agents/state.py) object in a fixed sequence:

```
                  ┌──────────────────────┐
                  │ OrchestratorAgent    │
                  └──────────┬───────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │ Step 1                │ Step 2                │ Step 3
     ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ DataHealthAgent  │───►│FaultDiagnosisAg. │───►│ AnomalyAgent     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
  • Ingests features      • Loads RF model        • Loads IF model
  • Computes Phase 7      • Calculates fault      • Computes anomaly
    Health Score            probabilities           score
  • Assesses rolling      • Evaluates class       • Flags baseline
    trend                   prediction              deviations
     │                       │                       │
     └───────────────────────┼───────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ AgentState      │
                    │ (Unified State) │
                    └─────────────────┘
```

1. **`DataHealthAgent`:** Validates artifact availability, loads feature sets, evaluates the Phase 7 composite Health Score across the timeline, calculates rolling slopes, and updates `AgentState` with overall health metrics.
2. **`FaultDiagnosisAgent`:** Ingests standardized features, evaluates the Phase 5 Random Forest model, and appends the fault classification (`Normal` vs `Faulty`), class probabilities, and top contributing vibration features to `AgentState`.
3. **`AnomalyAgent`:** Evaluates the Phase 6 Isolation Forest model, records the numerical anomaly score, flags statistical anomalies against the healthy baseline, and populates the execution log.
4. **Final Assembly:** The orchestrator packages `AgentState` into a serializable JSON summary containing health metrics, fault confidence, anomaly levels, and execution timestamps.

---

## 9. Retrieval-Augmented Knowledge Base (RAG)

The RAG subsystem ([`src/industrial_health/rag/`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/rag/)) bridges numerical sensor telemetry and human maintenance operations.

### Knowledge Base Documents

Located in [`knowledge_base/`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/knowledge_base/):
- [`bearing_faults.md`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/knowledge_base/bearing_faults.md): Defect mechanics, characteristic frequencies (BPFO, BPFI, BSF, FTF), and acoustic signatures.
- [`equipment_conditions.md`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/knowledge_base/equipment_conditions.md): Operating regimes, load/speed classifications, and baseline vibration ceilings.
- [`maintenance_guidelines.md`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/knowledge_base/maintenance_guidelines.md): ISO 10816 vibration severity zones, lubrication intervals, and replacement criteria.
- [`troubleshooting.md`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/knowledge_base/troubleshooting.md): Systematic root-cause decision trees, inspection checklists, and corrective action workflows.

### Document Ingestion and Vector Store

- **Chunking Strategy:** Heading-aware section splitting via [`documents.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/src/industrial_health/rag/documents.py). Documents are segmented on `##` Markdown headings (max 800 characters, min 20 characters) with deterministic SHA-256 chunk IDs. Produces exactly **66 indexed chunks**.
- **Embedding Model:** Local Ollama running `nomic-embed-text` (768-dimensional embeddings) over HTTP (`http://localhost:11434`).
- **Vector Database:** Local ChromaDB persistent instance stored in `chroma_db/`, collection name `industrial_health_knowledge` with cosine distance space.
- **Retrieval Engine:** Pure vector similarity search. Computes cosine distance ($0.0 = \text{identical}$, $2.0 = \text{orthogonal}$) and returns ranked chunks with document sources, section titles, and relevance percentages.

> [!NOTE]
> The active implementation performs **retrieval only** (no generative LLM hallucination). Engineers receive authoritative passages directly from the indexed engineering literature.

---

## 10. Interactive Streamlit Dashboard

The web dashboard ([`dashboard/app.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/dashboard/app.py)) provides a desktop-grade engineering interface with a customized dark palette (`#0b1120`, `#1e293b`, `#38bdf8`):

### Operational Views

1. **Executive Dashboard (`v1_dashboard.py`):** Primary status center displaying overall Equipment Health Score, Health Status badge (Stable, Degrading, Critical), Rolling Trend indicator, Total Anomalies Detected, and Fault State.
2. **Equipment Health (`v2_equipment_health.py`):** Interactive Plotly timeline of the complete run-to-failure degradation trajectory with colored threshold bands (Stable $\ge 70$, Degrading $40\text{--}70$, Critical $< 40$) and component breakdowns (RMS, Anomaly, Fault).
3. **Fault Diagnosis (`v3_fault_diagnosis.py`):** Random Forest prediction readouts, class probabilities (Normal vs Faulty), and horizontal bar charts ranking top vibration features by Gini importance.
4. **Anomaly Detection (`v4_anomaly_detection.py`):** Isolation Forest anomaly score progression over time, score distribution histogram, and a filterable table of anomalous snapshots.
5. **Maintenance Insights (`v5_maintenance_insights.py`):** Semantic RAG search interface allowing engineers to query bearing failure modes, ISO limits, and maintenance guidelines with real-time vector retrieval from ChromaDB.
6. **Agent Activity (`v6_agent_activity.py`):** Diagnostic telemetry tracking multi-agent execution order, per-agent latency (in milliseconds), execution status, and raw JSON inspect of the shared `AgentState`.
7. **Alerts / Events (`v7_alerts.py`):** Chronological log of operational events, recording when the equipment entered degradation and when anomaly counts exceeded warning limits.
8. **User Data Upload Mode:** Built-in sidebar toggle allowing engineers to upload arbitrary CSV sensor files. Includes automated validation ([`dashboard/upload_service.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/dashboard/upload_service.py)) for data dimensions, channel counts, and real-time ad-hoc feature extraction and scoring.

### Dashboard Screenshots

*(Screenshots can be added to `docs/images/` and linked below:)*
- `docs/images/dashboard_overview.png` — Main KPI status dashboard
- `docs/images/health_trajectory.png` — Equipment health degradation curve
- `docs/images/rag_insights.png` — Maintenance knowledge retrieval view

---

## 11. FastAPI REST API Service

The backend REST API ([`api/main.py`](file:///c:/Users/T10007/Desktop/Ltm_projects/Agentic_AI/api/main.py)) exposes the complete analytical workflow to external systems, SCADA platforms, or microservices.

### Endpoint Matrix

| Method | Endpoint | Description | Query / Body Params | Response Status |
|---|---|---|---|---|
| `GET` | `/health` | Liveness and artifact availability check | None | `200 OK` |
| `GET` | `/api/v1/pipeline/available-tests` | Returns list of test IDs with valid models | None | `200 OK` (returns `[1]`) |
| `POST` | `/api/v1/pipeline/run` | Executes Phase 8 multi-agent pipeline | `{"test_id": 1}` | `200 OK` (cached) |
| `GET` | `/api/v1/pipeline/{test_id}/summary` | Retrieves cached pipeline execution summary | Path: `test_id` | `200 OK` / `409 Conflict` |
| `GET` | `/api/v1/pipeline/{test_id}/timeseries`| Returns per-snapshot health scores and features | Path: `test_id` | `200 OK` |
| `GET` | `/api/v1/pipeline/{test_id}/fault` | Retrieves final Random Forest fault diagnosis | Path: `test_id` | `200 OK` |
| `GET` | `/api/v1/pipeline/{test_id}/anomaly`| Retrieves final Isolation Forest anomaly score | Path: `test_id` | `200 OK` |
| `POST` | `/api/v1/rag/query` | Executes semantic search against ChromaDB | `{"query": "...", "top_k": 5}` | `200 OK` / `503 Unavailable` |

### Key API Characteristics

- **Interactive Documentation:** Automatically available at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc` (ReDoc).
- **Sanitized Error Handling:** Errors never leak filesystem paths, Python tracebacks, or environment variables to callers.
- **In-Memory Caching:** Inferences on the 2,156 snapshots are cached after the initial `/run` execution, providing sub-millisecond responses on subsequent summary and timeseries requests.
- **Decoupled Architecture:** The Streamlit dashboard and FastAPI service operate independently; Streamlit imports core services directly and does not make HTTP calls to FastAPI.

---

## 12. Prerequisites and Installation

### System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Python `3.11` recommended (supports Python `3.9` through `3.11`)
- **RAM:** Minimum 8 GB (16 GB recommended for raw IMS data feature extraction)
- **Disk Space:** ~5 GB free space (raw IMS dataset is ~3.5 GB uncompressed)
- **Ollama:** Required for local RAG embeddings ([Download Ollama](https://ollama.ai/))

### Step-by-Step Installation (Windows PowerShell)

#### 1. Clone the Repository
```powershell
git clone https://github.com/Aniket181/Agentic_AI.git
cd Agentic_AI
```

#### 2. Create and Activate Virtual Environment
```powershell
python -m venv venvagentic
# If PowerShell script execution is restricted, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venvagentic\Scripts\Activate.ps1
```

#### 3. Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 4. Configure Environment Variables
```powershell
Copy-Item .env.example .env
```
*(The default configuration works out-of-the-box for local deterministic execution. An external LLM API key is optional.)*

#### 5. Set Up Ollama for Local Embeddings
```powershell
# In a separate terminal, ensure Ollama is running:
ollama serve

# Pull the required embedding model:
ollama pull nomic-embed-text
```

#### 6. Build the Local RAG Knowledge Base
```powershell
python scripts/build_knowledge_base.py
```

#### 7. (Optional) Place Raw Dataset if Extracting from Scratch
If you wish to re-extract features from raw vibration data, download the dataset from [Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) and extract into `archive/`:
```
archive/
├── 1st_test/1st_test/
├── 2nd_test/2nd_test/
└── 3rd_test/4th_test/txt/
```
*(Pre-extracted feature files in `data/features/` and trained models in `models/` allow immediate execution of the pipeline, dashboard, and API without downloading raw data.)*

---

## 13. Running the Project

All commands should be executed from the project root with `venvagentic` activated:

### 1. Launch the Streamlit Industrial Dashboard
```powershell
python -m streamlit run dashboard/app.py
```
*Access the dashboard at:* `http://localhost:8501`

### 2. Launch the FastAPI REST Server
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Access the Swagger UI at:* `http://127.0.0.1:8000/docs`

### 3. Execute the Deterministic Agentic Pipeline CLI
```powershell
python scripts/run_agentic_pipeline.py --test_id 1
```

### 4. Run Individual Analytics Pipelines

- **Exploratory Data Analysis (EDA):**
  ```powershell
  python scripts/run_eda.py
  ```
- **Feature Extraction (Raw Waveforms $\to$ 104 Features):**
  ```powershell
  python scripts/run_feature_extraction.py --tests 1
  ```
- **Chronological Preprocessing & Scaling:**
  ```powershell
  python scripts/run_preprocessing.py --tests 1
  ```
- **Fault Classification Experiment (Random Forest):**
  ```powershell
  python scripts/run_fault_diagnosis.py --test_id 1
  ```
- **Anomaly Detection Experiment (Isolation Forest):**
  ```powershell
  python scripts/run_anomaly_detection.py --test_id 1
  ```
- **Health Monitoring & Trend Trajectory:**
  ```powershell
  python scripts/run_health_monitoring.py --test_id 1
  ```
- **Rebuild ChromaDB Knowledge Base:**
  ```powershell
  python scripts/build_knowledge_base.py --rebuild
  ```

---

## 14. API Usage Examples

### 1. Check Service Health
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get | ConvertTo-Json
```
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response:**
```json
{
  "status": "ok",
  "service": "Industrial Equipment Health API",
  "version": "1.0.0",
  "artifacts": {
    "test1_models": true,
    "knowledge_base": true,
    "chroma_db": true
  }
}
```

### 2. Execute and Cache Pipeline
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/pipeline/run" -Method Post `
  -ContentType "application/json" -Body '{"test_id": 1}' | ConvertTo-Json -Depth 4
```
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/pipeline/run" \
     -H "Content-Type: application/json" \
     -d '{"test_id": 1}'
```
**Response (Summary):**
```json
{
  "test_id": 1,
  "health": {
    "mean_health_score": 43.43,
    "final_health_score": 0.53,
    "final_health_status": "Critical",
    "health_trend": "Degrading",
    "status_counts": {
      "Stable": 7,
      "Degrading": 1443,
      "Critical": 706
    }
  },
  "fault": {
    "final_rf_pred": 1,
    "predicted_fault": "Faulty",
    "normal_probability": 0.0133,
    "fault_probability": 0.9867,
    "fault_confidence": 0.9867
  },
  "anomaly": {
    "anomaly_score": 0.6901,
    "anomaly_prediction": 1,
    "anomaly_label": "Anomaly",
    "n_anomaly_predicted": 771
  },
  "execution": {
    "agents_executed": ["DataHealthAgent", "FaultDiagnosisAgent", "AnomalyAgent"],
    "execution_status": "COMPLETED",
    "errors": []
  }
}
```

### 3. Semantic Query Against Maintenance Knowledge Base
```powershell
$body = @{ query = "What are the common symptoms of bearing outer race defects?"; top_k = 2 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/rag/query" -Method Post `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 4
```
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the common symptoms of bearing outer race defects?", "top_k": 2}'
```
**Response:**
```json
{
  "query": "What are the common symptoms of bearing outer race defects?",
  "count": 2,
  "results": [
    {
      "source": "bearing_faults.md",
      "title": "Bearing Fault Types and Failure Modes",
      "section": "Outer Race Fault (BPFO)",
      "distance": 0.3124,
      "relevance_pct": 84.4,
      "content": "Outer race defects generate repetitive impacts at the Ball Pass Frequency Outer (BPFO)..."
    }
  ],
  "disclaimer": "Retrieval only. No LLM generation.",
  "embedding_model": "nomic-embed-text (Ollama)"
}
```

---

## 15. Verification and Testing

The project maintains a comprehensive, automated test suite built on `pytest`. At the completion of Phase 11 development, the test suite verified **338 passing tests** across 11 test modules with zero failures:

```
tests/
├── test_loader.py            # Raw ASCII file loading, edge cases, shape checks
├── test_preprocessing.py     # Leakage-safe chronological splitting & scalers
├── test_features.py          # Time-domain and FFT spectral feature calculation
├── test_model.py             # Random Forest training, hyperparameter validation
├── test_anomaly.py           # Isolation Forest training, inversion score semantics
├── test_health.py            # Health monitoring integration, weight validations
├── test_health_score.py      # Health score edge cases, clipping, score boundaries
├── test_agents.py            # Deterministic orchestrator, agent state transitions
├── test_rag.py               # Markdown chunker, ChromaDB integration, retrieval
├── test_dashboard.py         # Streamlit services, cache handling, upload validation
└── test_api.py               # FastAPI endpoints, Pydantic schemas, error handling
```

To run the complete test suite:
```powershell
python -m pytest -q
```
Or run individual test modules:
```powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/test_agents.py -v
```

> *Historical Checkpoint Note:* 338 passing tests were verified at the Phase 11 completion milestone. In restricted deployment environments where dynamic subprocess execution is disabled, live test re-execution can be initiated using the command above.

---

## 16. Experimental Results

The following metrics are extracted directly from the tracked experiment reports in `reports/`:

| Subsystem / Model | Metric Name | Measured Value | Evaluation Protocol & Dataset |
|---|---|---|---|
| **Random Forest (Fault Classifier)** | **Accuracy** | **83.18%** | 648 unseen test snapshots (Test 1, Phase 5) |
| | **Weighted F1-Score** | **84.67%** | Stratified chronological 70/30 split |
| | **Macro F1-Score** | **79.35%** | Class-balanced weights, 104 features |
| | **Faulty Recall** | **100.0%** | Zero false negatives on faulty period |
| | **Faulty Precision** | **54.39%** | Early degradation flagged prior to 80% cutoff |
| | **Normal Precision** | **100.0%** | Zero faulty snapshots classified as normal |
| **Isolation Forest (Anomaly Detector)** | **Accuracy** | **81.68%** | 950 unseen test snapshots (Test 1, Phase 6) |
| | **Weighted F1-Score** | **81.37%** | Evaluated against heuristic timeline |
| | **Macro F1-Score** | **81.53%** | Trained strictly on normal baseline (1,206 rows) |
| | **Faulty Recall** | **100.0%** | 432 of 432 faulty snapshots detected |
| | **Normal Precision** | **100.0%** | Zero faulty snapshots marked normal |
| **Composite Health Monitoring** | **Mean Health Score** | **43.43** | Evaluated across all 2,156 Test 1 snapshots |
| | **Median Health Score** | **55.32** | $w_{\text{rms}}=0.2, w_{\text{if}}=0.4, w_{\text{rf}}=0.4$ |
| | **Final Snapshot Score** | **0.53** | Snapshot 2,155 (Final health status: **Critical**) |
| | **Overall Run Trend** | **Degrading** | Q1 Mean: 59.02 $\to$ Q4 Mean: 1.18 ($\Delta = -57.84$) |
| | **Degradation Trigger** | **Snapshot 1624**| First sustained 5-snapshot degradation |
| **RAG Knowledge Base** | **Indexed Chunks** | **66 chunks** | 4 Markdown engineering guides in ChromaDB |

---

## 17. Limitations and Future Work

### Verified Limitations

1. **Single Operating Regime:** The IMS dataset was collected under a single steady state (2,000 RPM, 6,000 lbs radial load). Model behavior under variable speeds, fluctuating dynamic loads, or stop-start industrial duty cycles has not been evaluated.
2. **Heuristic Temporal Labels:** Due to the absence of micro-level fault annotations in the original IMS data, binary labels are based on run-tail positioning (final 20%). Early spalling that manifests before the 80% mark is statistically flagged as a false positive, accounting for the 54.39% precision on the faulty class.
3. **Test 1 Specialization:** Full model artifacts and orchestration pipelines are trained on IMS Test 1. Extending inference to Tests 2 and 3 requires retraining models on their 4-channel accelerometer configuration.
4. **Research Decision-Support Prototype:** The composite Health Score and threshold boundaries ($\ge 70, 40\text{--}70, < 40$) are engineering heuristics, not ISO-certified or safety-critical industrial standards. This software is not certified for closed-loop autonomous plant control.
5. **No Direct RUL Prognostics:** The system computes health score trajectories and degradation slopes, but does not provide validated Remaining Useful Life (RUL) Weibull hazard modeling or physics-of-failure prognostic estimates.

### Planned Future Enhancements

- [ ] **Multi-Condition Generalization:** Benchmarking feature extractors and classifiers against the Paderborn University bearing dataset to evaluate variable operating conditions.
- [ ] **Physics-Informed RUL Prognostics:** Implementing particle filtering and degradation modeling for calibrated Remaining Useful Life predictions with confidence bounds.
- [ ] **Real-Time Streaming Ingestion:** Integrating an Apache Kafka / MQTT consumer service to process live vibration streams directly into the feature extraction engine.
- [ ] **Multi-Equipment Monitoring:** Extending the dashboard and database schema to manage fleets of motors, gearboxes, and pumps simultaneously.
- [ ] **Agentic LLM Synthesis:** Introducing optional local LLM synthesis (via Ollama) to draft natural-language shift handover reports from the deterministic `AgentState`.

---

## 18. Security and Data Governance

- **Credential & Secret Protection:** The application relies on local, open-source models (scikit-learn, ChromaDB, Ollama). It does not require cloud API keys to operate. Environment variables are managed via `.env` (gitignored), and `.env.example` contains zero secrets.
- **Data Exclusion Policies:** As configured in `.gitignore`:
  - Raw vibration archives (`archive/`), intermediate numpy splits (`data/processed/`), and extracted CSVs (`data/features/`) are excluded from Git.
  - Serialized model files (`models/*.joblib`, `models/*.pkl`) and the local ChromaDB directory (`chroma_db/`) are excluded from source control.
  - Virtual environments (`venvagentic/`), Python caches (`__pycache__`), and local logs are strictly ignored.
- **API Error Sanitization:** The FastAPI layer sanitizes exception payloads to prevent internal filesystem structures, module paths, or operating system details from being exposed to clients.

---

## 19. Troubleshooting

### 1. Ollama Connection Error (503 Service Unavailable on RAG Queries)
- **Symptom:** RAG search returns `503 Ollama embedding service unavailable` or `Connection refused`.
- **Solution:** Ensure the Ollama daemon is running and the embedding model is installed:
  ```powershell
  ollama serve
  ollama pull nomic-embed-text
  ```

### 2. PowerShell Script Execution Policy Restriction
- **Symptom:** `.\venvagentic\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system`.
- **Solution:** Set execution policy for the current PowerShell process:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  .\venvagentic\Scripts\Activate.ps1
  ```

### 3. Missing Model Artifacts
- **Symptom:** Dashboard or API reports `No trained models found` or `random_forest_test1.joblib missing`.
- **Solution:** Run the pipeline training scripts to generate model artifacts:
  ```powershell
  python scripts/run_fault_diagnosis.py --test_id 1
  python scripts/run_anomaly_detection.py --test_id 1
  ```

### 4. ChromaDB Collection Not Found
- **Symptom:** Knowledge retriever reports `Collection industrial_health_knowledge does not exist`.
- **Solution:** Build the vector database:
  ```powershell
  python scripts/build_knowledge_base.py --rebuild
  ```

### 5. Port In Use Errors
- **Symptom:** `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`.
- **Solution:** Kill existing background processes using the target port or specify a custom port:
  ```powershell
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
  python -m streamlit run dashboard/app.py --server.port 8502
  ```

---

## 20. References and Acknowledgments

1. **IMS Bearing Dataset:** J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). *Bearing Data Set*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
2. **Isolation Forest:** Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. In Eighth IEEE International Conference on Data Mining (pp. 413-422).
3. **Random Forests:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. **Vibration Analysis Standards:** ISO 10816-3:2009. *Mechanical vibration — Evaluation of machine vibration by measurements on non-rotating parts*.
5. **Open Source Ecosystem:**
   - [Scikit-learn](https://scikit-learn.org/) — Machine learning in Python
   - [FastAPI](https://fastapi.tiangolo.com/) — Modern, high-performance web APIs
   - [Streamlit](https://streamlit.io/) — Interactive application framework for data science
   - [ChromaDB](https://www.trychroma.com/) — The AI-native open-source embedding database
   - [Ollama](https://ollama.ai/) — Local large language and embedding model execution

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Academic and experimental prototype.
