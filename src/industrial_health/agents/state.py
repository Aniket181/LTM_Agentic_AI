"""
state.py — Shared Agent State for Phase 8 Agentic Orchestration

The AgentState dataclass is the single source of truth passed between
all agents. Each agent reads from it and writes results back.

Design principles:
  - All fields are optional after initial construction (agents populate them).
  - State is serializable to dict/JSON (model objects never stored here).
  - Execution log records each agent's name, status, and duration.
  - Errors are accumulated, not raised, so the orchestrator can decide
    whether to continue or stop.

IMPORTANT:
  No ML model objects are stored in state. Models are loaded by each agent
  independently and kept in agent instance variables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentState:
    """
    Shared state passed between orchestration agents.

    Lifecycle:
        1. OrchestratorAgent creates and owns the state object.
        2. Each agent receives the state, reads inputs, writes results.
        3. After all agents complete, the orchestrator reads the final state.

    Serialization:
        Call state.to_dict() to get a JSON-serializable representation.
        Model objects and DataFrames are NOT stored here.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    test_id: int = 1

    # ── Feature data (populated by DataHealthAgent) ───────────────────────────
    feature_csv_path: Optional[str] = None
    n_snapshots: Optional[int] = None
    n_features: Optional[int] = None
    fault_start_idx: Optional[int] = None

    # ── Health results (populated by DataHealthAgent, wrapping Phase 7) ───────
    health_results: Optional[dict[str, Any]] = None
    # Summary scalars extracted from health_results for easy access
    mean_health_score: Optional[float] = None
    final_health_score: Optional[float] = None
    final_health_status: Optional[str] = None
    final_health_trend: Optional[str] = None
    status_counts: Optional[dict[str, int]] = None

    # ── Fault results (populated by FaultDiagnosisAgent, wrapping Phase 5) ───
    fault_results: Optional[dict[str, Any]] = None
    # Summary scalars
    final_rf_pred: Optional[int] = None
    final_prob_normal: Optional[float] = None
    final_prob_faulty: Optional[float] = None
    final_fault_label: Optional[str] = None

    # ── Anomaly results (populated by AnomalyAgent, wrapping Phase 6) ────────
    anomaly_results: Optional[dict[str, Any]] = None
    # Summary scalars
    final_anomaly_score: Optional[float] = None
    final_anomaly_pred: Optional[int] = None
    n_anomaly_predicted: Optional[int] = None

    # ── Execution tracking ────────────────────────────────────────────────────
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "INITIALIZED"    # INITIALIZED | RUNNING | COMPLETED | FAILED

    def log_agent(
        self,
        agent_name: str,
        status: str,
        duration_s: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Append a structured entry to the execution log.

        Args:
            agent_name: Class name of the agent (e.g. "DataHealthAgent").
            status:     "completed" | "failed" | "skipped".
            duration_s: Wall-clock seconds taken by the agent.
            error:      Error message if status == "failed".
        """
        entry: dict[str, Any] = {
            "agent":  agent_name,
            "status": status,
        }
        if duration_s is not None:
            entry["duration_s"] = round(duration_s, 3)
        if error is not None:
            entry["error"] = error
        self.execution_log.append(entry)

    def add_error(self, agent_name: str, message: str) -> None:
        """Record an error and add it to the error list."""
        self.errors.append(f"{agent_name}: {message}")

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the state.

        Large objects (health_results, fault_results, anomaly_results) are
        included if they are present; DataFrames are never stored in state
        so no special handling is needed.
        """
        return {
            "test_id":              self.test_id,
            "feature_csv_path":     self.feature_csv_path,
            "n_snapshots":          self.n_snapshots,
            "n_features":           self.n_features,
            "fault_start_idx":      self.fault_start_idx,
            "health_summary": {
                "mean_health_score":  self.mean_health_score,
                "final_health_score": self.final_health_score,
                "final_health_status":self.final_health_status,
                "final_health_trend": self.final_health_trend,
                "status_counts":      self.status_counts,
            },
            "fault_summary": {
                "final_rf_pred":     self.final_rf_pred,
                "final_prob_normal": self.final_prob_normal,
                "final_prob_faulty": self.final_prob_faulty,
                "final_fault_label": self.final_fault_label,
            },
            "anomaly_summary": {
                "final_anomaly_score": self.final_anomaly_score,
                "final_anomaly_pred":  self.final_anomaly_pred,
                "n_anomaly_predicted": self.n_anomaly_predicted,
            },
            "execution_log": self.execution_log,
            "errors":         self.errors,
            "status":         self.status,
        }
