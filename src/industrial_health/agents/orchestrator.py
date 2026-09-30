"""
orchestrator.py — Agent Orchestrator

Controls the end-to-end analysis workflow.
Passes a shared state dictionary between agents in sequence.

Workflow:
  1. Data/Health Agent   → load features, compute health score + degradation
  2. Fault Diagnosis Agent → RF prediction + confidence
  3. RCA Agent            → root cause + evidence + feature importance
  4. Maintenance Agent    → RAG retrieval + risk + recommendation

The state dictionary is the single source of truth for the workflow.
Each agent reads from state, runs its logic, and writes results back.

This architecture is upgradeable to LangGraph if needed later.
"""

from pathlib import Path
from typing import Optional
from loguru import logger

from industrial_health.agents.health_agent import HealthAgent
from industrial_health.agents.diagnosis_agent import DiagnosisAgent
from industrial_health.agents.rca_agent import RCAAgent
from industrial_health.agents.maintenance_agent import MaintenanceAgent


class OrchestratorAgent:
    """
    Coordinates the multi-agent bearing health analysis pipeline.

    Args:
        models_path: Directory containing saved model files.
        features_path: Directory containing feature datasets.
        knowledge_base_path: Directory containing RAG knowledge files.
        use_llm: Whether to use LLM for enhanced explanations.
    """

    def __init__(
        self,
        models_path: Path,
        features_path: Path,
        knowledge_base_path: Path,
        use_llm: bool = False,
    ):
        self.models_path = Path(models_path)
        self.features_path = Path(features_path)
        self.knowledge_base_path = Path(knowledge_base_path)
        self.use_llm = use_llm

        # Initialize sub-agents
        self.health_agent = HealthAgent(
            models_path=self.models_path,
            features_path=self.features_path,
        )
        self.diagnosis_agent = DiagnosisAgent(
            models_path=self.models_path,
        )
        self.rca_agent = RCAAgent()
        self.maintenance_agent = MaintenanceAgent(
            knowledge_base_path=self.knowledge_base_path,
            use_llm=use_llm,
        )

        logger.info("OrchestratorAgent initialized (use_llm={use_llm})")

    def run(
        self,
        test_id: int,
        snapshot_index: int = -1,
        bearing_channel: Optional[str] = None,
    ) -> dict:
        """
        Run the complete analysis pipeline for a given test snapshot.

        Args:
            test_id: IMS test ID (1, 2, or 3).
            snapshot_index: Index of snapshot to analyze (-1 = latest).
            bearing_channel: Specific bearing channel to focus on (optional).

        Returns:
            Complete health report as a structured dictionary.
        """
        logger.info(
            f"=== Orchestrator: Analyzing Test {test_id}, "
            f"snapshot={snapshot_index} ==="
        )

        # ── Initialize state ───────────────────────────────────────────────
        state = {
            "test_id": test_id,
            "snapshot_index": snapshot_index,
            "bearing_channel": bearing_channel,
            "status": "RUNNING",
            "errors": [],
        }

        # ── Step 1: Health Agent ───────────────────────────────────────────
        try:
            logger.info("Step 1: Health Agent running...")
            state = self.health_agent.run(state)
        except Exception as e:
            logger.error(f"Health Agent failed: {e}")
            state["errors"].append(f"HealthAgent: {e}")

        # ── Step 2: Diagnosis Agent ────────────────────────────────────────
        try:
            logger.info("Step 2: Diagnosis Agent running...")
            state = self.diagnosis_agent.run(state)
        except Exception as e:
            logger.error(f"Diagnosis Agent failed: {e}")
            state["errors"].append(f"DiagnosisAgent: {e}")

        # ── Step 3: RCA Agent ──────────────────────────────────────────────
        try:
            logger.info("Step 3: RCA Agent running...")
            state = self.rca_agent.run(state)
        except Exception as e:
            logger.error(f"RCA Agent failed: {e}")
            state["errors"].append(f"RCAAgent: {e}")

        # ── Step 4: Maintenance Agent ──────────────────────────────────────
        try:
            logger.info("Step 4: Maintenance Agent running...")
            state = self.maintenance_agent.run(state)
        except Exception as e:
            logger.error(f"Maintenance Agent failed: {e}")
            state["errors"].append(f"MaintenanceAgent: {e}")

        # ── Finalize ───────────────────────────────────────────────────────
        state["status"] = "COMPLETED" if not state["errors"] else "COMPLETED_WITH_ERRORS"
        logger.info(f"=== Orchestrator: Analysis complete (status={state['status']}) ===")
        return state
