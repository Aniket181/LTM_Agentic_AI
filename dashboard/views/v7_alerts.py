"""
dashboard/views/v7_alerts.py
Renders the alerts and events timeline view.
"""
from typing import Any
import streamlit as st
import pandas as pd

from dashboard import services

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## 🔔 Alerts & Events Timeline")
    st.caption("Chronological timeline of significant condition changes and heuristic thresholds.")
    
    timeseries_data = result.get("timeseries", {})
    if "error" in timeseries_data:
        st.warning(f"Could not load timeseries data to generate events: {timeseries_data['error']}")
        return
        
    if "records" not in timeseries_data:
        st.info("No data available to generate events.")
        return
        
    ts_df = pd.DataFrame(timeseries_data["records"])
    if ts_df.empty:
        st.info("No data available to generate events.")
        return
        
    # Generate events deterministically from the timeseries
    events = []
    
    # Track states
    current_health_status = "Stable"
    current_fault_status = "Normal"
    current_anomaly_status = "Normal"
    
    for _, row in ts_df.iterrows():
        idx = int(row.get("snapshot_idx", 0))
        h_score = row.get("health_score", 100.0)
        f_pred = row.get("predicted_fault", "Unknown")
        a_label = row.get("anomaly_label", "Unknown")
        
        # Health status logic (using the configured thresholds 70 and 40)
        h_status = "Stable"
        if h_score < 40:
            h_status = "Critical"
        elif h_score < 70:
            h_status = "Degrading"
            
        if h_status != current_health_status:
            if h_status == "Degrading":
                events.append({"idx": idx, "type": "warning", "msg": f"Health degraded below Stable threshold ({h_score:.1f})", "icon": "📉"})
            elif h_status == "Critical":
                events.append({"idx": idx, "type": "critical", "msg": f"Health reached Critical condition ({h_score:.1f})", "icon": "⚠️"})
            current_health_status = h_status
            
        if f_pred == "Faulty" and current_fault_status != "Faulty":
            events.append({"idx": idx, "type": "critical", "msg": "Fault prediction changed to FAULTY", "icon": "🔴"})
            current_fault_status = "Faulty"
            
        if a_label == "Anomaly" and current_anomaly_status != "Anomaly":
            events.append({"idx": idx, "type": "warning", "msg": "Anomaly detected in equipment behavior", "icon": "🚨"})
            current_anomaly_status = "Anomaly"
            
    # Reverse events to show newest first
    events.reverse()
    
    if not events:
        st.info("No significant events detected in the current dataset.")
        return
        
    st.markdown("### Critical Events Log")
    
    html = "<div style='display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; position: relative; padding-left: 20px; border-left: 2px solid #334155;'>"
    
    for ev in events:
        color = "#ef4444" if ev["type"] == "critical" else "#f59e0b"
        
        html += f"""
<div style="position: relative;">
    <div style="position: absolute; left: -29px; top: 15px; width: 16px; height: 16px; border-radius: 8px; background-color: {color}; border: 3px solid #0f172a;"></div>
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.2rem; background-color: #0f172a; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 1rem;">
        <div style="font-size: 1.5rem; margin-top: -0.2rem;">{ev['icon']}</div>
        <div style="flex: 1;">
            <div style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">{ev['msg']}</div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.3rem;">Snapshot Index: {ev['idx']}</div>
        </div>
    </div>
</div>
"""
        
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ **Note:** Event generation is based on heuristic threshold crossings within the current dataset timeline.")
