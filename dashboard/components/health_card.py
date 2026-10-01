"""
dashboard/components/health_card.py — Phase 10

Renders the Health Overview panel:
  - Plotly gauge chart (0–100), colour-coded by status
  - st.metric() boxes for final score, mean score, trend
  - Status distribution bar

Input: result["health"] dict from OrchestratorAgent.build_final_result()
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


# ── Colour mapping ────────────────────────────────────────────────────────────

STATUS_COLOUR = {
    "Stable":    "#22c55e",  # green-500
    "Degrading": "#f97316",  # orange-500
    "Critical":  "#ef4444",  # red-500
}

TREND_EMOJI = {
    "Improving": "📈",
    "Stable":    "➡️",
    "Degrading": "📉",
}


def _gauge_colour(score: float) -> str:
    if score >= 70:
        return STATUS_COLOUR["Stable"]
    elif score >= 40:
        return STATUS_COLOUR["Degrading"]
    return STATUS_COLOUR["Critical"]


# ── Main render function ──────────────────────────────────────────────────────

def render_health_card(health: dict[str, Any]) -> None:
    """
    Render the health overview panel.

    Args:
        health: result["health"] dict with keys:
            final_health_score, final_health_status, mean_health_score,
            health_trend, status_counts
    """
    final_score  = health.get("final_health_score",  0.0) or 0.0
    mean_score   = health.get("mean_health_score",   0.0) or 0.0
    status       = health.get("final_health_status", "Unknown")
    trend        = health.get("health_trend",        "Unknown")
    counts       = health.get("status_counts",       {}) or {}

    colour = _gauge_colour(final_score)
    trend_icon = TREND_EMOJI.get(trend, "❓")

    # ── Gauge chart ───────────────────────────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(final_score, 1),
        delta = {"reference": mean_score, "valueformat": ".1f"},
        title = {"text": "Health Score (0–100)", "font": {"size": 16}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": colour},
            "steps": [
                {"range": [0,  40], "color": "#fecaca"},   # red tint
                {"range": [40, 70], "color": "#fed7aa"},   # orange tint
                {"range": [70, 100], "color": "#bbf7d0"},  # green tint
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": final_score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metric row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Final Score",  f"{final_score:.1f}")
    with c2:
        st.metric("Mean Score",   f"{mean_score:.1f}")
    with c3:
        status_col = STATUS_COLOUR.get(status, "#6b7280")
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='font-size:0.8rem;color:gray'>Status</span><br>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{status_col}'>"
            f"{status}</span></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='font-size:0.8rem;color:gray'>Trend</span><br>"
            f"<span style='font-size:1.4rem;font-weight:700'>"
            f"{trend_icon} {trend}</span></div>",
            unsafe_allow_html=True,
        )

    # ── Status distribution ───────────────────────────────────────────────────
    if counts:
        stable    = counts.get("Stable",    0)
        degrading = counts.get("Degrading", 0)
        critical  = counts.get("Critical",  0)
        total     = stable + degrading + critical or 1
        st.markdown(
            f"**Snapshot distribution:** "
            f"🟢 Stable {stable} ({100*stable/total:.0f}%)  "
            f"🟠 Degrading {degrading} ({100*degrading/total:.0f}%)  "
            f"🔴 Critical {critical} ({100*critical/total:.0f}%)"
        )
