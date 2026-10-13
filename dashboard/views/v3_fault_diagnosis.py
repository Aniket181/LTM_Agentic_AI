"""
dashboard/views/v3_fault_diagnosis.py
Renders the fault diagnosis view.
"""
from typing import Any
import streamlit as st
import plotly.graph_objects as go

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## 🔴 Fault Diagnosis")
    st.caption("Machine Learning classification of current behavior (Random Forest).")
    
    fault = result.get("fault", {})
    label = fault.get("predicted_fault", "Unknown")
    prob_faulty = fault.get("fault_probability", 0.0) or 0.0
    prob_normal = fault.get("normal_probability", 0.0) or 0.0
    confidence = fault.get("fault_confidence", 0.0) or 0.0
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### Prediction")
        status_color = "#ef4444" if label == "Faulty" else "#10b981"
        icon = "⚠️" if label == "Faulty" else "✅"
        
        st.markdown(f"""
        <div style="border: 1px solid #334155; border-radius: 8px; padding: 2rem; background-color: #0f172a; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: {status_color};">{label}</div>
            <div style="font-size: 1.1rem; color: #94a3b8; margin-top: 1rem;">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("### Prediction Probabilities")
        
        # Horizontal probability bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Normal"],
            x=[prob_normal * 100],
            orientation='h',
            marker=dict(color="#10b981"),
            text=[f"{prob_normal * 100:.1f}%"],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            y=["Faulty"],
            x=[prob_faulty * 100],
            orientation='h',
            marker=dict(color="#ef4444"),
            text=[f"{prob_faulty * 100:.1f}%"],
            textposition="auto",
        ))
        
        fig.update_layout(
            template="plotly_dark",
            barmode='group',
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(range=[0, 100], title="Probability (%)", showgrid=True, gridcolor="#334155"),
            yaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔧 Possible Fault Condition")
    
    if label == "Faulty":
        interpretation = "The system has detected vibration characteristics resembling known fault signatures. If monitoring Test 1, this typically correlates with bearing outer race degradation. An inspection of the bearing housing and outer race is recommended."
    else:
        interpretation = "The system classifies the current vibration behavior as normal operating condition."
        
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1.2rem; background-color: #1e293b;">
        <span style="color: #cbd5e1;">{interpretation}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ **Disclaimer:** Fault prediction is based on the trained Random Forest model and heuristic temporal evaluation labels. Probabilities are not calibrated Bayesian fault probabilities.")
