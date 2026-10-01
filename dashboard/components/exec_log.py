"""
dashboard/components/exec_log.py — Phase 10

Renders the Agent Execution Log panel:
  - Collapsed expander (not shown by default)
  - Table of agent name, status, duration per execution entry
  - Red error display if any agents failed
  - Overall pipeline status indicator

Input: result["execution"] dict from OrchestratorAgent.build_final_result()
"""

from __future__ import annotations

from typing import Any

import streamlit as st


STATUS_ICON = {
    "completed": "✅",
    "failed":    "❌",
    "skipped":   "⏭️",
}


def render_exec_log(execution: dict[str, Any]) -> None:
    """
    Render the agent execution log as a collapsible panel.

    Args:
        execution: result["execution"] dict with keys:
            agents_executed, execution_status, errors, execution_log
    """
    status   = execution.get("execution_status", "UNKNOWN")
    errors   = execution.get("errors",           []) or []
    log      = execution.get("execution_log",    []) or []

    # ── Top-level status ──────────────────────────────────────────────────────
    status_map = {
        "COMPLETED":              ("✅ Pipeline completed successfully", "success"),
        "COMPLETED_WITH_ERRORS":  ("⚠️ Pipeline completed with errors", "warning"),
        "FAILED":                 ("❌ Pipeline failed", "error"),
        "RUNNING":                ("🔄 Pipeline running…", "info"),
    }
    status_text, status_level = status_map.get(
        status, (f"❓ Unknown status: {status}", "info")
    )

    if errors:
        st.error(f"**Pipeline errors detected:** {len(errors)} error(s)")
        for err in errors:
            st.error(err)

    with st.expander("🔧 Agent Execution Log", expanded=False):
        st.markdown(f"**Overall Status:** {status_text}")
        st.markdown("---")

        if not log:
            st.info("No execution log entries available.")
            return

        # Build log table
        rows = []
        for entry in log:
            agent_name = entry.get("agent",      "Unknown")
            entry_stat = entry.get("status",     "unknown")
            duration   = entry.get("duration_s", None)
            error_msg  = entry.get("error",      None)

            icon = STATUS_ICON.get(entry_stat, "❓")
            dur_str = f"{duration:.3f}s" if duration is not None else "—"

            rows.append({
                "Status":   f"{icon} {entry_stat}",
                "Agent":    agent_name,
                "Duration": dur_str,
                "Error":    error_msg or "",
            })

        import pandas as pd
        log_df = pd.DataFrame(rows)

        # Highlight rows with errors
        def _highlight(row):
            if "❌" in row["Status"]:
                return ["background-color: rgba(239,68,68,0.1)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            log_df.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )
