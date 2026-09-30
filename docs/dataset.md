# Dataset Documentation

## Dataset: IMS Bearing Dataset

**Source:** NASA/University of Cincinnati Intelligent Maintenance Systems (IMS) Center
**Kaggle:** [vinayak123tyagi/bearing-dataset](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)

## Experimental Setup

- **Test rig:** University of Cincinnati IMS test rig
- **Bearings:** Rexnord ZA-2115 double row bearings (4 per test)
- **Shaft speed:** 2000 RPM
- **Radial load:** 6,000 lbs (via spring mechanism)
- **Lubrication:** Circulating oil system with magnetic plug to catch debris
- **Accelerometers:** PCB 353B33 High Sensitivity Quartz ICP (4 sensors)
- **Sampling rate:** 20,480 Hz (20 kHz)
- **Data per snapshot:** 20,480 samples (~1 second of vibration data)
- **Snapshot interval:** Every 10 minutes

## Test Descriptions

### Test 1 (1st_test/)
- **Date range:** October 22, 2003 – November 25, 2003
- **Duration:** ~35 days
- **Number of files:** ~983 files
- **Columns:** 8 (4 bearings × 2 accelerometers each)
- **Column mapping:**
  - bearing1_ch1, bearing1_ch2 (columns 0, 1)
  - bearing2_ch1, bearing2_ch2 (columns 2, 3)
  - bearing3_ch1, bearing3_ch2 (columns 4, 5)
  - bearing4_ch1, bearing4_ch2 (columns 6, 7)
- **Known fault outcomes:**
  - Bearing 3: Outer race failure
  - Bearing 4: Roller element failure
- **File size:** ~1.1 MB each (20480 rows × 8 cols × float)

### Test 2 (2nd_test/)
- **Date range:** February 12, 2004 – February 19, 2004
- **Duration:** ~7 days
- **Number of files:** ~984 files
- **Columns:** 4 (4 bearings × 1 accelerometer each)
- **Column mapping:** bearing1, bearing2, bearing3, bearing4
- **Known fault outcomes:**
  - Bearing 1: Outer race failure
- **File size:** ~553 KB each (20480 rows × 4 cols × float)

### Test 3 (3rd_test/ → 4th_test/txt/)
- **Date range:** March 4, 2004 – April 4, 2004
- **Duration:** ~31 days
- **Number of files:** ~1,000+ files
- **Columns:** 4 (4 bearings × 1 accelerometer each)
- **Known fault outcomes:**
  - Bearing 3: Outer race failure
- **File size:** ~553 KB each

## File Format

- **Extension:** None (no file extension)
- **Format:** Tab-delimited plain text
- **Header:** None (no column headers)
- **Filename:** Timestamp (YYYY.MM.DD.HH.MM.SS)
- **Example filename:** `2003.10.22.12.06.24`
- **Values:** Float32 vibration acceleration measurements (g units)

```
-0.022  -0.039  -0.183  -0.054  -0.105  -0.134  -0.129  -0.142
-0.105  -0.017  -0.164  -0.183  -0.049   0.029  -0.115  -0.122
...
```

## Labeling Strategy

The IMS dataset contains **no explicit fault labels** in the data files.
Labels are derived from known experimental outcomes:

- **Normal period:** First 80% of each test's chronological timeline
- **Faulty period:** Last 20% of each test's chronological timeline
- **Assumption:** Bearing failure develops progressively and manifests in the last 20% of the test

> This is the standard academic labeling approach for the IMS dataset.
> It is a simplification — actual fault onset may be earlier or later.
> More sophisticated labeling can use RMS threshold crossing times.

## Data Quality Observations

- No missing values observed in sampled files
- All files have consistent column counts (per test)
- Some timestamp gaps exist (equipment restart periods) — see loader.py
- File sizes are consistent within each test (~1.1 MB for test1, ~553 KB for tests 2 & 3)
- The column count difference (8 vs 4) between Test 1 and Tests 2/3 is a known dataset characteristic

## Degradation Information

**YES — this is a run-to-failure dataset.**

- Each test starts from a healthy bearing condition
- The test runs continuously until bearing failure
- Vibration amplitude increases measurably as degradation progresses
- RMS trend over time clearly shows degradation trajectory
- This allows:
  - Degradation analysis (RMS trend)
  - Approximate RUL estimation (fraction of remaining life)

## Important Note on RUL

RUL (Remaining Useful Life) can be approximated as:
```
RUL_fraction = 1 - (snapshot_index / total_snapshots)
```

This is a simple time-fraction approximation — not a physics-based RUL model.
It is documented as an approximation in all reports.
