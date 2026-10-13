"""
dashboard/views/v2_equipment_health.py
Renders the detailed equipment health view.
"""
from typing import Any
import streamlit as st
import pandas as pd

from dashboard.components.trend_chart import render_trend_chart
from dashboard import services

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## 🏥 Equipment Health Details")
    st.caption("Detailed view of condition degradation and component health scores.")
    
    health = result.get("health", {})
    score = health.get("final_health_score", 0.0) or 0.0
    status = health.get("final_health_status", "Unknown")
    
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;">Current Health Score</div>
            <div style="font-size: 3rem; font-weight: bold; color: #f8fafc;">{score:.1f}<span style="font-size: 1.5rem; color: #64748b;"> / 100</span></div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;">Condition</div>
            <div style="font-size: 2rem; font-weight: bold;" class="status-{'critical' if status == 'Critical' else 'degrading' if status == 'Degrading' else 'stable'}">{status}</div>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Comprehensive Health & Component Trend")
    st.markdown("""
    <div style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 1rem;">
        The health score combines three key condition indicators: 
        <strong>Vibration Condition (RMS)</strong>, <strong>Anomaly Condition</strong>, and <strong>Fault Condition</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;'>", unsafe_allow_html=True)
    
    timeseries_data = result.get("timeseries", {})
    if "error" in timeseries_data:
        st.warning(f"Could not load health timeseries: {timeseries_data['error']}")
    elif "records" in timeseries_data:
        ts_df = pd.DataFrame(timeseries_data["records"])
        fault_start = timeseries_data["fault_start"]
        render_trend_chart(ts_df, fault_start, show_components=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔧 Action Recommendation")
    
    if status == "Stable":
        rec = "Continue normal monitoring. Equipment is operating within acceptable parameters."
    elif status == "Degrading":
        rec = "Schedule inspection and closely monitor the health trend. Preventative maintenance may be required soon."
    else:
        rec = "Prioritize equipment inspection. High likelihood of failure or significant fault condition."
        
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.2rem; background-color: #1e293b; border-left: 4px solid #38bdf8;">
        <strong>Recommended Next Step:</strong> {rec}
    </div>
    """, unsafe_allow_html=True)
