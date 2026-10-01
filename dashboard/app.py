"""
dashboard/app.py — Phase 10: Industrial Equipment Health Monitor Dashboard

ARCHITECTURE:
  This file is a PRESENTATION LAYER ONLY.
  No ML logic lives here. All computation is in:
    - dashboard/services.py      (bridge to Phase 5–9 pipeline)
    - dashboard/components/*.py  (reusable UI widgets)

FLOW:
  User → Streamlit Dashboard
       → services.run_pipeline()        → OrchestratorAgent (Phase 8)
       → services.get_health_timeseries() → HealthMonitor (Phase 7)
       → services.retrieve_knowledge()  → KnowledgeRetriever (Phase 9)

IMPORTANT:
  - No models are loaded or trained in this file.
  - No ChromaDB is accessed directly in this file.
  - All expensive computations are cached in services.py.

RUN:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path so `dashboard` is importable as a package,
# and src/ is on sys.path so `industrial_health` is importable.
# This is required when Streamlit launches the file directly via:
#   streamlit run dashboard/app.py
# because Streamlit adds dashboard/ to sys.path but NOT the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.config import get_available_test_ids
from dashboard import services
from dashboard.components.health_card  import render_health_card
from dashboard.components.trend_chart  import render_trend_chart
from dashboard.components.fault_card   import render_fault_card
from dashboard.components.anomaly_card import render_anomaly_card
from dashboard.components.rag_panel    import render_rag_panel
from dashboard.components.exec_log     import render_exec_log


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial Equipment Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Industrial Equipment Health Monitor**\n\n"
            "Agentic AI-Based Bearing Health Monitoring and Fault Diagnosis.\n"
            "Phase 10 Dashboard — IMS Bearing Dataset, Test 1.\n\n"
            "⚠️ Results are based on heuristic temporal labels. "
            "Not for safety-critical use."
        ),
    },
)


# ── Minimal CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .stMetric label { font-size: 0.78rem !important; }
    .stMetric value { font-size: 1.3rem !important; }
    div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Gears_animation.gif/120px-Gears_animation.gif",
             width=80, caption="")
    st.markdown("## ⚙️ Health Monitor")
    st.markdown("---")

    available_tests = get_available_test_ids()
    if not available_tests:
        st.error("No trained models found. Run Phase 5 and 6 pipelines first.")
        st.stop()

    test_id = st.selectbox(
        "Select Test Dataset",
        options=available_tests,
        format_func=lambda x: f"IMS Test {x}",
        index=0,
        key="test_id_select",
    )

    st.markdown("---")

    run_clicked = st.button(
        "▶ Run Pipeline",
        use_container_width=True,
        type="primary",
        help="Execute the agentic pipeline (cached for 1 hour).",
    )

    if st.button(
        "🔄 Re-run (Clear Cache)",
        use_container_width=True,
        help="Clear cached results and force a full pipeline re-run.",
    ):
        services.clear_pipeline_cache()
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "**Phase 10** Streamlit Dashboard\n\n"
        "Integrates Phase 5 (RF Fault), Phase 6 (IF Anomaly), "
        "Phase 7 (Health Score), Phase 8 (Orchestration), "
        "Phase 9 (RAG Knowledge).\n\n"
        "IMS Bearing Dataset, Test 1 only.\n\n"
        "⚠️ Heuristic temporal labels. Not for safety-critical decisions."
    )


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown(
    "<h1>⚙️ Industrial Equipment Health Monitor"
    "<span style='font-size:0.7rem;font-weight:400;color:gray;margin-left:12px'>"
    "Phase 10 · IMS Bearing Dataset"
    "</span></h1>",
    unsafe_allow_html=True,
)

# ── Auto-load pipeline on first visit (Run Pipeline button or fresh load) ──────

if run_clicked or "pipeline_result" not in st.session_state:
    with st.spinner("Running agentic pipeline…"):
        result = services.run_pipeline(test_id)
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_test_id"] = test_id
else:
    result = st.session_state.get("pipeline_result", {})
    cached_test_id = st.session_state.get("pipeline_test_id", test_id)
    if cached_test_id != test_id:
        # Test selection changed — rerun for new test
        with st.spinner("Running agentic pipeline for new test…"):
            result = services.run_pipeline(test_id)
        st.session_state["pipeline_result"] = result
        st.session_state["pipeline_test_id"] = test_id


# ── Handle pipeline failure ───────────────────────────────────────────────────

if "error" in result:
    st.error(f"**Pipeline Error:** {result['error']}")
    if result.get("details"):
        with st.expander("Error Details"):
            for d in result["details"]:
                st.code(d, language="text")
    st.stop()


# ── Extract result sections ───────────────────────────────────────────────────

health    = result.get("health",    {})
fault     = result.get("fault",     {})
anomaly   = result.get("anomaly",   {})
execution = result.get("execution", {})


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_diagnosis, tab_rag = st.tabs([
    "📊 Health Overview",
    "🔬 Fault & Anomaly",
    "📚 Knowledge Base",
])


# ════════════════════════════════════════════════════════════════════════════════
# Tab 1: Health Overview
# ════════════════════════════════════════════════════════════════════════════════

with tab_overview:
    col_gauge, col_trend = st.columns([1, 2], gap="medium")

    with col_gauge:
        st.markdown("### 🏥 Health Score")
        render_health_card(health)

    with col_trend:
        st.markdown("### 📈 Health Trend")

        show_components = st.toggle(
            "Show component breakdown",
            value=False,
            key="show_components_toggle",
        )

        timeseries_data = services.get_health_timeseries(test_id)
        if "error" in timeseries_data:
            st.warning(f"Could not load health timeseries: {timeseries_data['error']}")
        else:
            import pandas as pd
            ts_df = pd.DataFrame(timeseries_data["records"])
            fault_start = timeseries_data["fault_start"]
            render_trend_chart(ts_df, fault_start, show_components=show_components)

    st.markdown("---")
    render_exec_log(execution)


# ════════════════════════════════════════════════════════════════════════════════
# Tab 2: Fault & Anomaly
# ════════════════════════════════════════════════════════════════════════════════

with tab_diagnosis:
    col_fault, col_anomaly = st.columns(2, gap="medium")

    with col_fault:
        st.markdown("### 🔴 Fault Diagnosis")
        st.caption("Phase 5 — Random Forest Classifier")
        render_fault_card(fault)

    with col_anomaly:
        st.markdown("### 🚨 Anomaly Detection")
        st.caption("Phase 6 — Isolation Forest")
        render_anomaly_card(anomaly)

    # ── Summary metrics row ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Pipeline Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Snapshots", f"{result.get('health', {}).get('status_counts', {}).get('Stable', 0) + result.get('health', {}).get('status_counts', {}).get('Degrading', 0) + result.get('health', {}).get('status_counts', {}).get('Critical', 0):,}")
    with m2:
        st.metric("Final Health", f"{health.get('final_health_score', 0):.1f}")
    with m3:
        st.metric("Fault", fault.get("predicted_fault", "—"))
    with m4:
        st.metric("Anomaly", anomaly.get("anomaly_label", "—"))
    with m5:
        n_anomaly = anomaly.get("n_anomaly_predicted", 0) or 0
        n_total_counts = sum((health.get("status_counts") or {}).values()) or 1
        st.metric("% Anomalous", f"{100*n_anomaly/n_total_counts:.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
# Tab 3: Knowledge Base (RAG)
# ════════════════════════════════════════════════════════════════════════════════

with tab_rag:
    render_rag_panel()


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "⚙️ **Industrial Equipment Health Monitor** · Phase 10 · "
    "IMS Bearing Dataset (Test 1) · "
    "Phase 5 RF + Phase 6 IF + Phase 7 Health Score + Phase 8 Orchestration + Phase 9 RAG · "
    "⚠️ Heuristic temporal labels. Not for safety-critical decisions."
)
