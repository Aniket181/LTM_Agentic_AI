"""
dashboard/views/v6_agent_activity.py
Renders the agent activity and execution log view.
"""
from typing import Any
import streamlit as st

def render(result: dict[str, Any], test_id: str) -> None:
    st.markdown("## ⚙️ Agent Activity & Analysis Workflow")
    st.caption("Overview of the AI agentic pipeline execution.")
    
    execution = result.get("execution", {})
    status = execution.get("execution_status", "UNKNOWN")
    log = execution.get("execution_log", []) or []
    
    status_map = {
        "COMPLETED":              ("Pipeline completed successfully", "#10b981"),
        "COMPLETED_WITH_ERRORS":  ("Pipeline completed with errors", "#f59e0b"),
        "FAILED":                 ("Pipeline failed", "#ef4444"),
        "RUNNING":                ("Pipeline running…", "#38bdf8"),
    }
    status_text, status_color = status_map.get(status, (f"Unknown status: {status}", "#94a3b8"))
    
    st.markdown(f"""
    <div style="border: 1px solid #334155; border-radius: 8px; padding: 1rem; background-color: #0f172a; display: inline-block; margin-bottom: 2rem;">
        <span style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; margin-right: 1rem;">System Status</span>
        <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Execution Workflow")
    
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 2rem 0; padding: 0 1rem;">
        <div style="text-align: center; flex: 1;">
            <div style="width: 50px; height: 50px; border-radius: 25px; background-color: #1e293b; border: 2px solid #38bdf8; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto; font-size: 1.5rem;">1</div>
            <div style="font-weight: bold; color: #e2e8f0;">Data Health Agent</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">Data loading, feature extraction</div>
        </div>
        <div style="flex: 0.5; height: 2px; background-color: #334155;"></div>
        <div style="text-align: center; flex: 1;">
            <div style="width: 50px; height: 50px; border-radius: 25px; background-color: #1e293b; border: 2px solid #38bdf8; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto; font-size: 1.5rem;">2</div>
            <div style="font-weight: bold; color: #e2e8f0;">Fault Diagnosis Agent</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">Random Forest evaluation</div>
        </div>
        <div style="flex: 0.5; height: 2px; background-color: #334155;"></div>
        <div style="text-align: center; flex: 1;">
            <div style="width: 50px; height: 50px; border-radius: 25px; background-color: #1e293b; border: 2px solid #38bdf8; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto; font-size: 1.5rem;">3</div>
            <div style="font-weight: bold; color: #e2e8f0;">Anomaly Agent</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">Isolation Forest detection</div>
        </div>
        <div style="flex: 0.5; height: 2px; background-color: #334155;"></div>
        <div style="text-align: center; flex: 1;">
            <div style="width: 50px; height: 50px; border-radius: 25px; background-color: #1e293b; border: 2px solid #10b981; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto; font-size: 1.5rem;">✓</div>
            <div style="font-weight: bold; color: #e2e8f0;">Final Assessment</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">Health assessment generated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⏱️ Agent Execution Log")
    
    if not log:
        st.info("No execution log entries available.")
    else:
        log_html = "<div style='display: flex; flex-direction: column; gap: 1rem;'>"
        for entry in log:
            agent = entry.get("agent", "Unknown")
            action = entry.get("action", "")
            step_status = entry.get("status", "")
            duration = entry.get("duration_s", 0.0)
            
            icon = "✅" if step_status == "completed" else "❌" if step_status == "failed" else "⏭️"
            color = "#10b981" if step_status == "completed" else "#ef4444" if step_status == "failed" else "#64748b"
            
            # Create human readable descriptions based on agent name
            purpose = ""
            if "Data/Health" in agent or "HealthMonitor" in agent:
                purpose = "Analyzed equipment condition and health indicators."
            elif "Fault" in agent:
                purpose = "Evaluated the current condition for fault-like behavior."
            elif "Anomaly" in agent:
                purpose = "Compared equipment behavior against learned normal baseline."
            else:
                purpose = action
                
            log_html += f"""
<div style="border: 1px solid #334155; border-radius: 8px; padding: 1.2rem; background-color: #0f172a; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">{icon}</span>
            <span style="font-weight: bold; font-size: 1.1rem; color: #e2e8f0;">{agent}</span>
        </div>
        <div style="color: #94a3b8; margin-top: 0.5rem; margin-left: 1.7rem;">{purpose}</div>
    </div>
    <div style="text-align: right;">
        <div style="color: {color}; font-weight: bold; text-transform: uppercase; font-size: 0.85rem;">{step_status}</div>
        <div style="color: #94a3b8; font-family: monospace; margin-top: 0.3rem;">{duration:.3f}s</div>
    </div>
</div>
"""
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
