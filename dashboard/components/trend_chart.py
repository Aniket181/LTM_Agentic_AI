"""
dashboard/components/trend_chart.py — Phase 10

Renders the Health Score timeline using Plotly:
  - health_score vs snapshot_index (line chart)
  - Horizontal threshold lines at 70 (Stable) and 40 (Critical)
  - Vertical marker at heuristic fault_start_idx (labelled clearly)
  - Optional component sub-plot (RMS / anomaly / fault contributions)

Input: pd.DataFrame from services.get_health_timeseries_df()
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def render_trend_chart(
    df: pd.DataFrame,
    fault_start: int,
    show_components: bool = False,
) -> None:
    """
    Render the health score trend chart.

    Args:
        df:              Scored DataFrame from HealthMonitor.score_dataset().
        fault_start:     Heuristic fault start snapshot index.
        show_components: If True, adds a subplot for component contributions.
    """
    if df is None or df.empty:
        st.warning("No health timeseries data available.")
        return

    rows = 2 if show_components else 1
    row_heights = [0.65, 0.35] if show_components else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.08,
        subplot_titles=(
            ["Health Score Timeline", "Component Contributions (0–100)"]
            if show_components else ["Health Score Timeline"]
        ),
    )

    # ── Health score line ─────────────────────────────────────────────────────
    # Colour each point by status
    colour_map = {"Stable": "#22c55e", "Degrading": "#f97316", "Critical": "#ef4444"}
    status_col = df["health_status"].map(colour_map).fillna("#6b7280")

    fig.add_trace(
        go.Scatter(
            x=df["snapshot_index"],
            y=df["health_score"],
            mode="lines",
            name="Health Score",
            line=dict(color="#3b82f6", width=2),
            hovertemplate=(
                "Snapshot: %{x}<br>"
                "Health: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ── Threshold lines ───────────────────────────────────────────────────────
    x_range = [df["snapshot_index"].min(), df["snapshot_index"].max()]

    fig.add_trace(
        go.Scatter(
            x=x_range, y=[70, 70],
            mode="lines",
            name="Stable threshold (70)",
            line=dict(color="#22c55e", width=1.5, dash="dash"),
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range, y=[40, 40],
            mode="lines",
            name="Critical threshold (40)",
            line=dict(color="#ef4444", width=1.5, dash="dash"),
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Heuristic label boundary ──────────────────────────────────────────────
    fig.add_vline(
        x=fault_start,
        line_width=2,
        line_dash="dot",
        line_color="#a855f7",
        annotation_text="⚠ Heuristic Label Boundary",
        annotation_position="top right",
        annotation_font_color="#a855f7",
        row=1, col=1,  # type: ignore[call-arg]
    )

    # ── Component sub-plot ────────────────────────────────────────────────────
    if show_components:
        for col_name, label, colour in [
            ("health_rms_component",     "RMS component",     "#06b6d4"),
            ("health_anomaly_component", "Anomaly component", "#f97316"),
            ("health_fault_component",   "Fault component",   "#8b5cf6"),
        ]:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["snapshot_index"],
                        y=df[col_name],
                        mode="lines",
                        name=label,
                        line=dict(width=1.5, color=colour),
                    ),
                    row=2, col=1,
                )

    # ── Layout ────────────────────────────────────────────────────────────────
    total_height = 480 if show_components else 360

    fig.update_layout(
        height=total_height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Snapshot Index", showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(
        title_text="Health Score",
        range=[0, 105],
        showgrid=True,
        gridcolor="#e5e7eb",
        row=1,
    )
    if show_components:
        fig.update_yaxes(
            title_text="Component (0–100)",
            range=[0, 105],
            row=2,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.caption(
        "⚠️ **Heuristic label boundary:** The dashed purple line marks the point "
        "where the temporal label changes from Normal to Faulty. "
        "This is not a verified fault-onset annotation."
    )
