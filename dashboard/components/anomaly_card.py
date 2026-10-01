"""
dashboard/components/anomaly_card.py — Phase 10

Renders the Anomaly Detection panel:
  - Anomaly label (Normal / Anomaly)
  - Anomaly score (Phase 6 semantics: higher = more anomalous)
  - Count of anomaly-predicted snapshots
  - Score semantics note

SCORE SEMANTICS (Phase 6, preserved exactly):
  decision_scores() = -score_samples() → HIGHER = MORE ANOMALOUS
  anomaly_pred: 1 = anomaly, 0 = normal

Input: result["anomaly"] dict from OrchestratorAgent.build_final_result()
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_anomaly_card(anomaly: dict[str, Any]) -> None:
    """
    Render the anomaly detection panel.

    Args:
        anomaly: result["anomaly"] dict with keys:
            anomaly_label, anomaly_score, n_anomaly_predicted,
            anomaly_prediction
    """
    label         = anomaly.get("anomaly_label",       "Unknown")
    score         = anomaly.get("anomaly_score",        0.0) or 0.0
    n_anomaly     = anomaly.get("n_anomaly_predicted",  0) or 0
    pred          = anomaly.get("anomaly_prediction",   None)

    # ── Status indicator ─────────────────────────────────────────────────────
    if pred == 1 or label == "Anomaly":
        colour = "#ef4444"
        icon   = "🚨"
        bg     = "rgba(239,68,68,0.08)"
    elif pred == 0 or label == "Normal":
        colour = "#22c55e"
        icon   = "✅"
        bg     = "rgba(34,197,94,0.08)"
    else:
        colour = "#6b7280"
        icon   = "❓"
        bg     = "rgba(0,0,0,0.04)"

    st.markdown(
        f"<div style='text-align:center;padding:12px;border-radius:8px;"
        f"background:{bg}'>"
        f"<div style='font-size:2rem'>{icon}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{colour}'>{label}</div>"
        f"<div style='font-size:0.85rem;color:gray'>Anomaly Status (Final Snapshot)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Metric row ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Anomaly Score",
            f"{score:.4f}",
            help="Higher = more anomalous. Phase 6 IF decision score.",
        )
    with c2:
        st.metric(
            "Anomaly Snapshots",
            f"{n_anomaly:,}",
            help="Number of snapshots classified as anomalous across the full run.",
        )

    # ── Score semantics note ──────────────────────────────────────────────────
    st.caption(
        "ℹ️ **Score semantics:** Anomaly score = `−score_samples()` "
        "(Isolation Forest). Higher values indicate more anomalous behaviour. "
        "IF trained on Normal-only data from the first 70% of the run."
    )
