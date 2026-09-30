"""
rca_agent.py — Root Cause Analysis Agent

Responsibilities:
  1. Interpret anomaly score, fault prediction, and feature importance
  2. Identify top contributing sensor signals / features
  3. Produce structured evidence list
  4. Generate a probable root cause statement
  5. Clearly separate OBSERVED EVIDENCE from PROBABLE INTERPRETATION

The RCA agent does NOT invent causes.
Every conclusion is backed by actual model outputs and feature data.

Evidence structure:
  - Predicted fault class + confidence
  - Top important features (from RF model)
  - Anomaly score and status
  - RMS deviation from baseline
  - Trend direction

Reads from state:
  - fault_result, anomaly_result, feature_importance, health_result,
    rms_current, rms_baseline

Writes to state:
  - rca_result (dict with evidence + probable_cause + interpretation)
"""

import numpy as np
from loguru import logger


# ── Feature signal interpretation map ────────────────────────────────────────
# Maps feature keywords to bearing fault indicators
FEATURE_INTERPRETATION = {
    "kurtosis":       "impulsive shock events (early fault indicator)",
    "crest_factor":   "impulsive peaks relative to RMS (fault signature)",
    "rms":            "overall vibration energy level",
    "peak":           "maximum vibration amplitude",
    "spectral_energy":"frequency domain energy (FFT-based)",
    "dominant_freq":  "dominant vibration frequency",
    "std":            "signal variability",
    "skewness":       "signal asymmetry (impulse direction)",
}


def _interpret_feature(feature_name: str) -> str:
    """Return a human-readable interpretation for a feature name."""
    for keyword, interpretation in FEATURE_INTERPRETATION.items():
        if keyword in feature_name.lower():
            return interpretation
    return "signal characteristic"


def _extract_bearing_from_feature(feature_name: str) -> str:
    """Extract bearing/channel name from a feature name like 'bearing1_ch1__kurtosis'."""
    parts = feature_name.split("__")
    return parts[0] if parts else "unknown"


class RCAAgent:
    """
    Root Cause Analysis agent.
    Synthesizes model outputs into structured, evidence-based RCA.
    """

    def run(self, state: dict) -> dict:
        """
        Perform RCA and update state.

        Writes:
          state['rca_result'] = {
              'observed_evidence': [...],
              'probable_cause': str,
              'probable_interpretation': str,
              'affected_components': [...],
              'confidence_level': str,
          }
        """
        fault_result  = state.get("fault_result", {})
        anomaly_result = state.get("anomaly_result", {})
        health_result  = state.get("health_result", {})
        feat_importance = state.get("feature_importance", [])
        rms_current    = state.get("rms_current")
        rms_baseline   = state.get("rms_baseline")
        test_id        = state.get("test_id", "unknown")

        # ── Build Evidence List ────────────────────────────────────────────
        evidence = []

        # 1. Fault prediction evidence
        fault_label = fault_result.get("fault_label", "Unknown")
        fault_conf  = fault_result.get("confidence", 0.0)
        evidence.append({
            "type": "Fault Prediction",
            "observation": f"Random Forest predicted: {fault_label} "
                           f"(confidence: {fault_conf:.1%})",
            "significance": "HIGH" if fault_conf > 0.80 else "MEDIUM",
        })

        # 2. Anomaly evidence
        anomaly_status = anomaly_result.get("anomaly_status", "UNKNOWN")
        anomaly_score  = anomaly_result.get("anomaly_score", 0.5)
        evidence.append({
            "type": "Anomaly Detection",
            "observation": f"Isolation Forest: {anomaly_status} "
                           f"(normalized score: {anomaly_score:.3f})",
            "significance": "HIGH" if anomaly_status == "ANOMALOUS" else "LOW",
        })

        # 3. RMS degradation evidence
        if rms_current is not None and rms_baseline is not None:
            rms_increase = ((rms_current - rms_baseline) / (rms_baseline + 1e-10)) * 100
            evidence.append({
                "type": "Vibration Level (RMS)",
                "observation": (
                    f"RMS = {rms_current:.4f} | Baseline = {rms_baseline:.4f} | "
                    f"Change: {rms_increase:+.1f}%"
                ),
                "significance": "HIGH" if abs(rms_increase) > 50 else "MEDIUM",
            })

        # 4. Top feature importance evidence
        affected_bearings = set()
        for feat_info in feat_importance[:5]:
            feat_name = feat_info.get("feature", "")
            importance = feat_info.get("importance", 0.0)
            bearing = _extract_bearing_from_feature(feat_name)
            affected_bearings.add(bearing)
            interp = _interpret_feature(feat_name)
            evidence.append({
                "type": "Feature Importance",
                "observation": f"{feat_name} (importance: {importance:.4f}) — {interp}",
                "significance": "HIGH" if importance > 0.05 else "LOW",
            })

        # 5. Health score evidence
        health_score = health_result.get("health_score", 50.0)
        trend        = health_result.get("trend", "STABLE")
        evidence.append({
            "type": "Health Score",
            "observation": f"Overall health: {health_score}/100 | Trend: {trend}",
            "significance": "CRITICAL" if health_score < 40 else "HIGH" if health_score < 70 else "LOW",
        })

        # ── Determine Probable Cause ───────────────────────────────────────
        is_faulty  = fault_result.get("fault_class", 0) == 1
        is_anomaly = anomaly_result.get("is_anomaly", False)

        if is_faulty and is_anomaly:
            probable_cause = "Bearing degradation with active fault signature"
            interpretation = (
                "Both the fault classifier and anomaly detector indicate abnormal "
                "bearing behavior. The combination of high kurtosis and elevated "
                "vibration energy suggests a developing bearing fault. "
                "The specific bearing location is indicated by the top feature channels."
            )
        elif is_faulty and not is_anomaly:
            probable_cause = "Fault signature detected (anomaly detector within normal range)"
            interpretation = (
                "The Random Forest classifier detects fault characteristics in the "
                "feature patterns, though the Isolation Forest still rates this within "
                "normal operational bounds. This may indicate an early-stage fault "
                "developing within normal vibration amplitude limits."
            )
        elif not is_faulty and is_anomaly:
            probable_cause = "Abnormal operating conditions (not yet classified as fault)"
            interpretation = (
                "The Isolation Forest detects deviation from normal baseline, but "
                "the fault classifier does not yet confirm a fault pattern. "
                "This may indicate transient loading conditions or an early anomaly "
                "that has not developed into a recognizable fault signature."
            )
        else:
            probable_cause = "Normal operation — no fault or anomaly detected"
            interpretation = (
                "Both models indicate normal bearing behavior. Continue scheduled "
                "monitoring according to maintenance plan."
            )

        rca_result = {
            "observed_evidence":       evidence,
            "probable_cause":          probable_cause,
            "probable_interpretation": interpretation,
            "affected_components":     list(affected_bearings),
            "confidence_level":        (
                "HIGH"   if fault_conf > 0.85 else
                "MEDIUM" if fault_conf > 0.60 else
                "LOW"
            ),
            "note": (
                "OBSERVED EVIDENCE is based on measured model outputs. "
                "PROBABLE INTERPRETATION is a decision-support analysis — "
                "not a definitive engineering diagnosis."
            ),
        }

        state["rca_result"] = rca_result
        state["rca_agent_status"] = "COMPLETED"

        logger.info(f"RCA: {probable_cause}")
        return state
