"""
dashboard/views/v4_anomaly_detection.py
Renders the anomaly detection view.
"""
from typing import Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dashboard import services

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## 🚨 Anomaly Detection")
    st.caption("Unsupervised anomaly detection (Isolation Forest) trained on normal baseline.")
    
    anomaly = result.get("anomaly", {})
    health = result.get("health", {})
    
    label = anomaly.get("anomaly_label", "Unknown")
    score = anomaly.get("anomaly_score", 0.0) or 0.0
    n_anomaly = anomaly.get("n_anomaly_predicted", 0) or 0
    total_snaps = sum((health.get("status_counts") or {}).values()) or 1
    anomaly_pct = (n_anomaly / total_snaps) * 100 if total_snaps > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    status_color = "#ef4444" if label == "Anomaly" else "#10b981"
    
    with col1:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Status</div>
            <div style="font-size: 2rem; font-weight: bold; color: {status_color}; margin-top: 0.5rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Current Score</div>
            <div style="font-size: 2rem; font-weight: bold; color: #f8fafc; margin-top: 0.5rem;">{score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Anomaly Snapshots</div>
            <div style="font-size: 2rem; font-weight: bold; color: #f8fafc; margin-top: 0.5rem;">{n_anomaly}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #0f172a;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Anomaly Percentage</div>
            <div style="font-size: 2rem; font-weight: bold; color: #f8fafc; margin-top: 0.5rem;">{anomaly_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Anomaly Score Trend")
    st.markdown("<div style='border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a;'>", unsafe_allow_html=True)
    
    timeseries_data = result.get("timeseries", {})
    if "error" not in timeseries_data and "records" in timeseries_data:
        ts_df = pd.DataFrame(timeseries_data["records"])
        
        if "anomaly_score_norm" in ts_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_df["snapshot_idx"],
                y=ts_df["anomaly_score_norm"],
                mode='lines',
                name='Anomaly Score',
                line=dict(color="#f97316", width=2)
            ))
            
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Anomaly Threshold",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                xaxis=dict(title="Snapshot Index", showgrid=True, gridcolor="#334155", zerolinecolor="#334155"),
                yaxis=dict(title="Anomaly Score (Normalized)", range=[0, 1.05], showgrid=True, gridcolor="#334155", zerolinecolor="#334155")
            )
            st.plotly_chart(fig)
        else:
            st.info("Anomaly score timeseries not available.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Normal Operating Baseline Concept")
    
    st.markdown("""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; background-color: #1e293b; display: flex; align-items: center; justify-content: space-between; text-align: center;">
        <div style="flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: bold; color: #f8fafc;">Normal Baseline</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">Data from early healthy operating period</div>
        </div>
        <div style="font-size: 1.5rem; color: #64748b;">➡</div>
        <div style="flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-weight: bold; color: #f8fafc;">Isolation Forest</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">Learns the boundaries of normal vibration</div>
        </div>
        <div style="font-size: 1.5rem; color: #64748b;">➡</div>
        <div style="flex: 1;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
            <div style="font-weight: bold; color: #f8fafc;">Anomaly Assessment</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">Current behavior compared to learned baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ **Disclaimer:** Higher anomaly score indicates more anomalous behavior compared to the baseline. It is an unsupervised deviation indicator, not a percentage risk of failure.")
