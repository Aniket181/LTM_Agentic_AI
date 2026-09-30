# Bearing Fault Types and Characteristics

## Overview
Rolling element bearings are one of the most common mechanical components in industrial machinery.
Bearing failures account for approximately 40-50% of all electric motor failures.
Early detection of bearing faults is critical for preventing catastrophic equipment failure.

## Common Bearing Fault Types

### 1. Outer Race Fault (OR Fault)
**Description:** Damage or spalling on the stationary outer race of the bearing.
**Cause:** Overloading, contamination, inadequate lubrication, fatigue.
**Vibration Signature:**
  - Periodic impacts at the Ball Pass Frequency Outer Race (BPFO)
  - BPFO = (n/2) × f_r × (1 - d/D × cos(α))
  - Where n=number of balls, f_r=shaft frequency, d=ball diameter, D=pitch diameter
**Signal Characteristics:**
  - Elevated kurtosis (impulsive shocks)
  - Increased RMS at specific frequencies
  - Detectable in early stages via kurtosis monitoring
**Health Impact:** Progressive — moderate at onset, critical at advanced stage.

### 2. Inner Race Fault (IR Fault)
**Description:** Damage on the rotating inner race.
**Cause:** Excessive load, fatigue cracking, improper installation.
**Vibration Signature:**
  - Impacts at Ball Pass Frequency Inner Race (BPFI)
  - BPFI = (n/2) × f_r × (1 + d/D × cos(α))
  - Sidebands around BPFI due to modulation with shaft rotation
**Signal Characteristics:**
  - Kurtosis increase earlier than OR fault (rotating element amplifies defect)
  - Amplitude modulation pattern
**Health Impact:** Often more rapid progression than outer race faults.

### 3. Rolling Element (Ball) Fault
**Description:** Damage on the rolling ball or roller elements.
**Cause:** Fatigue, contamination, material defects.
**Vibration Signature:**
  - Impacts at Ball Spin Frequency (BSF)
  - BSF = (D/2d) × f_r × |1 - (d/D × cos(α))²|
  - Typically at 2×BSF
**Signal Characteristics:**
  - Less pronounced than race faults
  - Elevated crest factor and kurtosis
  - Energy in frequency bands around BSF and harmonics
**Health Impact:** Moderate — can remain stable for extended periods.

### 4. Cage Fault
**Description:** Damage to the bearing cage (retainer).
**Cause:** Excessive speed, lubricant starvation, mechanical impact.
**Vibration Signature:**
  - Fundamental Train Frequency (FTF)
  - FTF = (f_r/2) × (1 - d/D × cos(α))
**Signal Characteristics:**
  - Low-frequency signal component
  - Often masked by other vibrations
  - Can indicate broader bearing distress

## IMS Dataset Specific Fault Information

The IMS (Intelligent Maintenance Systems) Bearing Dataset contains:
- **Test 1:** Outer race fault (Bearing 3), Roller element fault (Bearing 4)
- **Test 2:** Outer race fault (Bearing 1)
- **Test 3:** Outer race fault (Bearing 3)

All tests are run-to-failure experiments with continuous data collection.
The test rig operates at 2000 RPM with a 6,000 lb radial load.

## Fault Severity Stages

### Stage 1: Incipient Fault
- Very faint high-frequency noise
- Kurtosis may begin rising
- RMS still within normal range
- Detection method: Kurtosis, high-frequency envelope analysis

### Stage 2: Developing Fault
- Bearing frequencies become detectable in spectrum
- RMS shows slight increase
- Kurtosis elevated (>4 indicates impulsive behavior)
- Detection method: FFT spectrum, envelope analysis

### Stage 3: Advanced Fault
- Clear fault frequencies and harmonics
- Significant RMS increase
- High kurtosis, elevated crest factor
- Bearing may be making audible noise
- Detection method: Broadband vibration increase

### Stage 4: Failure Imminent
- Wideband vibration increase
- All frequencies elevated
- Kurtosis may decrease (fault is now distributed)
- Immediate action required

## Key Vibration Metrics for Bearing Monitoring

| Metric | Normal Range | Alert Threshold | Critical Threshold |
|--------|-------------|-----------------|-------------------|
| RMS | Baseline | 1.5× baseline | 3× baseline |
| Kurtosis | 2.5–3.5 | >4.0 | >6.0 |
| Crest Factor | 2.5–3.0 | >3.5 | >5.0 |
| Peak-to-Peak | Baseline | 2× baseline | 4× baseline |

*Note: Thresholds are indicative. Actual values depend on machine type and operating conditions.*
