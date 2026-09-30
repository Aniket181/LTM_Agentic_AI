# Equipment Condition Classification

## Health Score Scale (0–100)

The health score integrates multiple signal analysis components:
- Anomaly Detection (40%): Isolation Forest deviation from normal baseline
- Fault Classification (40%): Random Forest fault prediction confidence
- Degradation Trend (20%): RMS change relative to healthy baseline

### HEALTHY (90–100)
**Description:** Equipment operating within normal parameters.
**Vibration:** Within or near baseline levels.
**Recommended action:** Continue standard monitoring.
**Monitoring frequency:** Per standard schedule.
**Maintenance:** Routine preventive maintenance only.

### NORMAL / SLIGHT DEGRADATION (70–89)
**Description:** Equipment showing minor deviations from baseline.
**Vibration:** Slightly elevated — within acceptable limits.
**Recommended action:** Increase monitoring frequency. Schedule inspection.
**Monitoring frequency:** Increase to weekly.
**Maintenance:** Plan preventive maintenance. Check lubrication.

### DEGRADED (40–69)
**Description:** Significant deviation from healthy baseline detected.
**Vibration:** Elevated RMS and/or elevated kurtosis.
**Recommended action:** Schedule bearing inspection within 1 week.
**Monitoring frequency:** Daily monitoring.
**Maintenance:** Prepare replacement bearing. Inspect lubrication and alignment.

### CRITICAL (0–39)
**Description:** Imminent failure risk. Immediate action required.
**Vibration:** Severely elevated. Active fault signature detected.
**Recommended action:** Immediate inspection. Plan emergency maintenance.
**Monitoring frequency:** Continuous (if possible).
**Maintenance:** Emergency bearing replacement may be required.

## Risk Level Definitions

### LOW Risk
- Equipment healthy, no anomaly detected
- Continue normal operation
- Standard preventive maintenance sufficient

### MEDIUM Risk
- Minor anomalies or slight degradation
- Increase monitoring frequency
- Plan preventive maintenance within 1 month
- Review operating conditions

### HIGH Risk
- Active degradation or fault signature present
- Urgent maintenance planning required
- Restrict equipment operation if overloaded
- Inspection within 1 week mandatory

### CRITICAL Risk
- Severe fault condition detected
- Equipment may fail imminently
- Immediate stop and inspect if conditions allow
- Do not operate without engineering review

## Degradation Trend Indicators

### STABLE
- Health score variance < ±2% over recent measurements
- RMS showing no consistent upward trend
- Continue monitoring at standard interval

### DEGRADING
- Health score consistently decreasing (>1% per measurement period)
- RMS showing clear upward trend
- Action: Increase monitoring, schedule inspection soon

### IMPROVING
- Health score increasing after maintenance or load reduction
- Post-maintenance verification showing positive trend
- Continue monitoring to confirm stabilization

## IMS Dataset Operating Conditions Reference

The IMS bearing dataset was collected under these conditions:
- **Test rig:** University of Cincinnati Intelligent Maintenance Systems Center
- **Shaft speed:** 2000 RPM
- **Radial load:** 6,000 lbs (applied via a spring mechanism)
- **Sampling frequency:** 20,480 Hz (20 kHz)
- **Snapshot interval:** Every 10 minutes
- **Data per snapshot:** 20,480 samples (approximately 1 second of data)
- **Bearings:** Rexnord ZA-2115 double row bearings
- **Lubricant:** Circulating oil system

These conditions are representative of medium-duty industrial bearing applications.
The test was run continuously until bearing failure, providing complete degradation data.
