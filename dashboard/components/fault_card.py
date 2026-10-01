"""
dashboard/components/fault_card.py — Phase 10

Renders the Fault Diagnosis panel:
  - st.metric() for fault prediction label
  - Horizontal probability bars (Normal vs Faulty)
  - Confidence display
  - Label disclaimer (heuristic temporal labels)

Input: result["fault"] dict from OrchestratorAgent.build_final_result()
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_fault_card(fault: dict[str, Any]) -> None:
    """
    Render the fault diagnosis panel.

    Args:
        fault: result["fault"] dict with keys:
            predicted_fault, fault_probability, normal_probability,
            fault_confidence, final_rf_pred
    """
    label       = fault.get("predicted_fault",   "Unknown")
    prob_faulty = fault.get("fault_probability", 0.0) or 0.0
    prob_normal = fault.get("normal_probability", 0.0) or 0.0
    confidence  = fault.get("fault_confidence",  0.0) or 0.0

    # ── Status indicator ─────────────────────────────────────────────────────
    if label == "Faulty":
        colour = "#ef4444"
        icon   = "🔴"
    elif label == "Normal":
        colour = "#22c55e"
        icon   = "🟢"
    else:
        colour = "#6b7280"
        icon   = "❓"

    st.markdown(
        f"<div style='text-align:center;padding:12px;border-radius:8px;"
        f"background:rgba(0,0,0,0.04)'>"
        f"<div style='font-size:2rem'>{icon}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{colour}'>{label}</div>"
        f"<div style='font-size:0.85rem;color:gray'>Fault Prediction</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Confidence metric ────────────────────────────────────────────────────
    st.metric("Confidence", f"{confidence*100:.1f}%")

    # ── Probability bar chart ─────────────────────────────────────────────────
    fig = go.Figure(go.Bar(
        x=[prob_normal, prob_faulty],
        y=["Normal", "Faulty"],
        orientation="h",
        marker_color=["#22c55e", "#ef4444"],
        text=[f"{prob_normal*100:.1f}%", f"{prob_faulty*100:.1f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        height=160,
        margin=dict(t=20, b=20, l=80, r=20),
        xaxis=dict(range=[0, 1], title="Probability", tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.caption(
        "⚠️ **Disclaimer:** RF model trained on heuristic temporal labels "
        "(final 20% of run = Faulty). Probabilities are not calibrated "
        "Bayesian fault probabilities."
    )
