"""
dashboard/app.py — Phase 10: Industrial Equipment Health Monitor Dashboard

ARCHITECTURE:
  Presentation layer acting as a multipage application shell.
  Routes to dedicated views in dashboard/views/ based on sidebar selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.config import get_available_test_ids
from dashboard import services

from dashboard.views import (
    v1_dashboard,
    v2_equipment_health,
    v3_fault_diagnosis,
    v4_anomaly_detection,
    v5_maintenance_insights,
    v6_agent_activity,
    v7_alerts,
)
from dashboard.upload_service import validate_uploaded_csv, process_upload_and_run

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial Equipment Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS for Dark Professional Theme ────────────────────────────────────

st.markdown("""
<style>
    /* Dark Theme Professional Adjustments */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #e2e8f0 !important; }
    h1 { font-size: 1.8rem !important; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
    h2 { font-size: 1.4rem !important; margin-top: 1rem; }
    h3 { font-size: 1.1rem !important; color: #94a3b8 !important; }
    
    /* Metrics / KPI Cards */
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; color: #f8fafc; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b !important;
    }
    .stRadio > label { display: none; }
    div[role="radiogroup"] > label {
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 5px;
        transition: all 0.2s ease;
    }
    
    hr { border-color: #334155 !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #38bdf8 !important; margin-top:0;'>⚙️ HEALTH MONITOR</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem; margin-bottom: 1.5rem;'>Decision Support System</div>", unsafe_allow_html=True)
    
    available_tests = get_available_test_ids()
    if not available_tests:
        st.error("No trained models found. Run Phase 5 and 6 pipelines first.")
        st.stop()

    st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.5rem;'>Data Source</div>", unsafe_allow_html=True)
    
    data_source = st.radio(
        "Source",
        options=["Built-in Dataset", "User Data (Upload CSV)"],
        label_visibility="collapsed"
    )
    
    run_clicked = False
    uploaded_file = None
    
    if data_source == "Built-in Dataset":
        test_id = st.selectbox(
            "Dataset",
            options=available_tests,
            format_func=lambda x: f"IMS Test {x}",
            index=0,
            key="test_id_select"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("▶ Run Health Analysis", type="primary")
    else:
        test_id = "UPLOAD"
        uploaded_file = st.file_uploader("Drag & drop your CSV sensor data here", type=["csv"])
        if uploaded_file:
            validation = validate_uploaded_csv(uploaded_file)
            if validation["valid"]:
                st.success(f"Validated: {validation['rows']} rows, {validation['channels']} channels")
                st.markdown("<br>", unsafe_allow_html=True)
                run_clicked = st.button("▶ Run Health Analysis", type="primary")
            else:
                st.error("Validation Failed")
                st.info(validation["error"])
    
    st.markdown("<br><div style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.5rem;'>Navigation</div>", unsafe_allow_html=True)
    
    pages = {
        "Dashboard": v1_dashboard.render,
        "Equipment Health": v2_equipment_health.render,
        "Fault Diagnosis": v3_fault_diagnosis.render,
        "Anomaly Detection": v4_anomaly_detection.render,
        "Maintenance Insights": v5_maintenance_insights.render,
        "Agent Activity": v6_agent_activity.render,
        "Alerts / Events": v7_alerts.render
    }
    
    selected_page = st.radio(
        "Go to",
        options=list(pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("v1.0.0 · Phase 10")


# ── Pipeline Execution & Caching ──────────────────────────────────────────────

if run_clicked or "pipeline_result" not in st.session_state:
    if test_id == "UPLOAD" and not run_clicked and "pipeline_result" not in st.session_state:
        # Don't auto-run upload until they click the button
        st.info("Please upload a CSV file and click 'Run Health Analysis'.")
        st.stop()
        
    with st.spinner("Analyzing equipment..."):
        if test_id == "UPLOAD":
            # Reposition stream to start
            uploaded_file.seek(0)
            result = process_upload_and_run(validation["df"], uploaded_file.name)
            # Add data source info
            result["data_source"] = f"User Uploaded Data ({uploaded_file.name})"
        else:
            result = services.run_pipeline(test_id)
            if "error" not in result:
                result["timeseries"] = services.get_health_timeseries(test_id)
            result["data_source"] = f"IMS Test {test_id}"
            
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_test_id"] = test_id
else:
    result = st.session_state.get("pipeline_result", {})
    cached_test_id = st.session_state.get("pipeline_test_id", test_id)
    if cached_test_id != test_id and test_id != "UPLOAD":
        with st.spinner("Analyzing equipment..."):
            result = services.run_pipeline(test_id)
            if "error" not in result:
                result["timeseries"] = services.get_health_timeseries(test_id)
            result["data_source"] = f"IMS Test {test_id}"
        st.session_state["pipeline_result"] = result
        st.session_state["pipeline_test_id"] = test_id

if "error" in result:
    st.error("### Analysis Pipeline Failed")
    st.error(result["error"])
    st.stop()


# ── Render Selected Page ──────────────────────────────────────────────────────

page_renderer = pages[selected_page]
page_renderer(result, test_id)
