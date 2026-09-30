"""
run_agentic_pipeline.py — Phase 8: Agentic Pipeline CLI (Test 1)

Orchestrates DataHealthAgent → FaultDiagnosisAgent → AnomalyAgent using
existing Phase 5/6/7 model artifacts. No new ML training occurs.

Usage:
    python scripts/run_agentic_pipeline.py --test_id 1

DO NOT START PHASE 9 WITHOUT APPROVAL.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_health.agents import OrchestratorAgent, AgentState
from industrial_health.health.health_monitor import HealthConfig


# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
SCALERS_DIR  = PROJECT_ROOT / "models" / "scaler"
REPORTS_DIR  = PROJECT_ROOT / "reports"


def banner(title: str, w: int = 65) -> None:
    print(f"\n{'=' * w}\n  {title}\n{'=' * w}")


def section(title: str, w: int = 65) -> None:
    print(f"\n{'─' * w}\n  {title}\n{'─' * w}")


def check_artifact(path: Path, label: str) -> bool:
    ok = path.exists()
    print(f"  [{'✓' if ok else '✗'}] {label:<35} {path.name}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 — Agentic AI Pipeline (Test 1 only)"
    )
    parser.add_argument("--test_id", type=int, default=1, choices=[1])
    parser.add_argument("--weight_rms",     type=float, default=0.20)
    parser.add_argument("--weight_anomaly", type=float, default=0.40)
    parser.add_argument("--weight_fault",   type=float, default=0.40)
    args = parser.parse_args()

    test_id = args.test_id

    banner(f"PHASE 8 — AGENTIC AI PIPELINE  |  TEST {test_id}")
    print(f"""
  ARCHITECTURE:
    Deterministic orchestration of existing Phase 5/6/7 components.
    No new ML training. No LLM calls.

  EXECUTION ORDER:
    1. DataHealthAgent       → Phase 7 health monitoring
    2. FaultDiagnosisAgent   → Phase 5 RF fault prediction
    3. AnomalyAgent          → Phase 6 IF anomaly detection
""")

    # ── Pre-flight checks ──────────────────────────────────────────────────
    section("PRE-FLIGHT CHECKS")
    rf_model_path  = MODELS_DIR / f"random_forest_test{test_id}.joblib"
    rf_scaler_path = SCALERS_DIR / f"phase5_test{test_id}_scaler.pkl"
    if_model_path  = MODELS_DIR / f"isolation_forest_test{test_id}.joblib"
    if_scaler_path = SCALERS_DIR / f"phase6_test{test_id}_scaler.pkl"
    csv_path       = FEATURES_DIR / f"test{test_id}_features.csv"

    all_ok = all([
        check_artifact(csv_path,       "Feature CSV (Phase 3)"),
        check_artifact(rf_model_path,  "RF model (Phase 5)"),
        check_artifact(rf_scaler_path, "RF scaler (Phase 5)"),
        check_artifact(if_model_path,  "IF model (Phase 6)"),
        check_artifact(if_scaler_path, "IF scaler (Phase 6)"),
    ])
    if not all_ok:
        print("\n  STOP — Missing artifacts. Run Phases 3, 5, and 6 first.")
        sys.exit(1)
    print("\n  ✓ All artifacts present.")

    # ── Build HealthConfig ─────────────────────────────────────────────────
    try:
        cfg = HealthConfig(
            weight_rms=args.weight_rms,
            weight_anomaly=args.weight_anomaly,
            weight_fault=args.weight_fault,
        )
    except ValueError as e:
        print(f"  ERROR in HealthConfig: {e}")
        sys.exit(1)

    # ── Initialize Orchestrator ────────────────────────────────────────────
    section("INITIALIZING ORCHESTRATOR")
    orchestrator = OrchestratorAgent(
        models_dir=MODELS_DIR,
        features_dir=FEATURES_DIR,
        scalers_dir=SCALERS_DIR,
        test_id=test_id,
        health_config=cfg,
        stop_on_failure=True,
    )
    print(f"  Agents registered:  {orchestrator._EXECUTION_ORDER}")
    print(f"  Health weights:     RMS={cfg.weight_rms}, IF={cfg.weight_anomaly}, RF={cfg.weight_fault}")
    print(f"  Health thresholds:  Stable>={cfg.threshold_stable}, "
          f"Degrading>={cfg.threshold_degrading}, Critical<{cfg.threshold_degrading}")

    # ── Execute Pipeline ───────────────────────────────────────────────────
    section("EXECUTING PIPELINE")
    import time
    t0 = time.monotonic()
    state = orchestrator.run()
    total_s = time.monotonic() - t0

    # ── Execution Trace ────────────────────────────────────────────────────
    section("EXECUTION TRACE")
    print(f"  {'Agent':<30} {'Status':<12} {'Duration':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*10}")
    for entry in state.execution_log:
        dur = f"{entry.get('duration_s', 0):.2f}s" if "duration_s" in entry else "—"
        print(f"  {entry['agent']:<30} {entry['status']:<12} {dur:>10}")
        if "error" in entry:
            print(f"    ERROR: {entry['error']}")
    print(f"\n  Pipeline status:  {state.status}")
    print(f"  Total wall-clock:  {total_s:.2f}s")

    if state.errors:
        print(f"\n  ✗ Errors ({len(state.errors)}):")
        for e in state.errors:
            print(f"    {e}")

    if state.status in {"FAILED"}:
        print("\n  Pipeline failed — see errors above.")
        sys.exit(1)

    # ── Structured Final Result ────────────────────────────────────────────
    section("FINAL STRUCTURED RESULT")
    final = orchestrator.build_final_result(state)

    print(f"\n  TEST {final['test_id']}")
    h = final["health"]
    print(f"\n  HEALTH:")
    print(f"    Mean health score:    {h['mean_health_score']:.2f}" if h['mean_health_score'] else "    N/A")
    print(f"    Final health score:   {h['final_health_score']:.2f}" if h['final_health_score'] else "    N/A")
    print(f"    Final health status:  {h['final_health_status']}")
    print(f"    Health trend:         {h['health_trend']}")
    if h["status_counts"]:
        sc = h["status_counts"]
        print(f"    Status counts:        Stable={sc.get('Stable',0)}  "
              f"Degrading={sc.get('Degrading',0)}  Critical={sc.get('Critical',0)}")

    f = final["fault"]
    print(f"\n  FAULT DIAGNOSIS (final snapshot):")
    print(f"    Prediction:           {f['predicted_fault']}")
    print(f"    P(Normal):            {f['normal_probability']}")
    print(f"    P(Faulty):            {f['fault_probability']}")
    print(f"    Fault confidence:     {f['fault_confidence']}")

    a = final["anomaly"]
    print(f"\n  ANOMALY DETECTION (final snapshot):")
    print(f"    Anomaly score:        {a['anomaly_score']}")
    print(f"    Anomaly label:        {a['anomaly_label']}")
    print(f"    N anomalies in run:   {a['n_anomaly_predicted']}")

    ex = final["execution"]
    print(f"\n  EXECUTION:")
    print(f"    Agents executed:  {ex['agents_executed']}")
    print(f"    Status:           {ex['execution_status']}")
    print(f"    Errors:           {len(ex['errors'])}")

    # ── Save experiment JSON ───────────────────────────────────────────────
    section("SAVING EXPERIMENT JSON")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        **final,
        "timestamp":    datetime.now().isoformat(),
        "phase":        "8 — Agentic AI Orchestration",
        "methodology":  (
            "Phase 8 introduces orchestration around existing deterministic ML and "
            "health-monitoring components. It does not introduce autonomous LLM "
            "reasoning or new machine-learning training."
        ),
        "configuration": cfg.as_dict,
        "health_details": state.health_results,
        "fault_details":  state.fault_results,
        "anomaly_details":state.anomaly_results,
    }
    out_path = REPORTS_DIR / f"phase8_experiment_test{test_id}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    banner("PHASE 8 COMPLETE — STOP. DO NOT PROCEED TO PHASE 9 WITHOUT APPROVAL.")


if __name__ == "__main__":
    main()
