"""
dashboard/views/v1_dashboard.py
Renders the high-level dashboard with KPIs and trend.
"""
from typing import Any
import streamlit as st
import pandas as pd


from dashboard.components.trend_chart import render_trend_chart
from dashboard import services

def render(result: dict[str, Any], test_id: str) -> None:
    data_source = result.get("data_source", "Unknown")
    st.markdown("## 📊 Equipment Health Overview")
    st.caption(f"High-level system status and health trend. **Data Source:** {data_source}")
    
    health = result.get("health", {})
    fault = result.get("fault", {})
    anomaly = result.get("anomaly", {})

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = health.get("final_health_score", 0.0) or 0.0
        status = health.get("final_health_status", "Unknown")
        status_color = "status-critical" if status == "Critical" else ("status-degrading" if status == "Degrading" else "status-stable")
        
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">🏥 Health Score</div>
            <div style="font-size: 2.2rem; font-weight: bold; color: #f8fafc; margin: 0.5rem 0;">{score:.1f}<span style="font-size: 1rem; color: #64748b;"> / 100</span></div>
            <div class="{status_color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        trend = health.get("health_trend", "Unknown")
        trend_icon = "📉" if trend == "Degrading" else ("📈" if trend == "Improving" else "➡️")
        trend_color = "status-critical" if trend == "Degrading" else "status-stable"
        
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">📈 Health Trend</div>
            <div style="font-size: 2.2rem; font-weight: bold; color: #f8fafc; margin: 0.5rem 0;">{trend_icon}</div>
            <div class="{trend_color}">{trend}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        fault_pred = fault.get("predicted_fault", "Unknown")
        conf = fault.get("fault_confidence", 0.0) or 0.0
        fault_color = "status-critical" if fault_pred == "Faulty" else "status-stable"
        
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">🔴 Fault Diagnosis</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #f8fafc; margin: 0.5rem 0;">{fault_pred}</div>
            <div class="{fault_color}">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        anomaly_label = anomaly.get("anomaly_label", "Unknown")
        score_val = anomaly.get("anomaly_score", 0.0) or 0.0
        anomaly_color = "status-critical" if anomaly_label == "Anomaly" else "status-stable"
        
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">🚨 Anomaly Detection</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #f8fafc; margin: 0.5rem 0;">{anomaly_label}</div>
            <div class="{anomaly_color}">Score: {score_val:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Health Trend Chart
    st.markdown("### 📉 Health Score Trend")
    st.markdown("<div style='border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;'>", unsafe_allow_html=True)
    
    timeseries_data = result.get("timeseries", {})
    if "error" in timeseries_data:
        st.warning(f"Could not load health timeseries: {timeseries_data['error']}")
    elif "records" in timeseries_data:
        ts_df = pd.DataFrame(timeseries_data["records"])
        fault_start = timeseries_data["fault_start"]
        render_trend_chart(ts_df, fault_start, show_components=False)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Current Condition Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Current Condition Summary")
    
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a; border-left: 4px solid {'#ef4444' if status == 'Critical' else '#f59e0b' if status == 'Degrading' else '#10b981'};">
        <p style="font-size: 1.1rem; color: #e2e8f0; margin-bottom: 0;">
            Equipment health is currently <strong>{status}</strong>. 
            The trend indicates it is <strong>{trend}</strong>.
            The latest analysis is classified as <strong>{fault_pred}</strong> and <strong>{anomaly_label}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
