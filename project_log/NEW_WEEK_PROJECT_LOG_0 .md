# NEW_WEEK_PROJECT_LOG_0 — Week0 Metric Baseline

## This Week's Objective

Convert qualitative weak points (slow convergence / oscillation / under-power / overshoot) into quantitative metrics and build the first reproducible baseline.

## Progress

* metrics_core.py implemented
* compute_metrics.py implemented (handles off‑by‑one obs/actions length)
* npz_inspector_v2.py implemented
* Day54 sample → metrics.json successfully generated
* action norm distribution analysed → calibrated thresholds (sat_high=5.84, under_a=1.24)
* first two diagnostic plots saved (radius.png / err_norm.png)

## Sample Results (Day54)

* convergence_step = 0
* steady_state_error ≈ 3.3e‑07
* oscillation_index = 0.0
* saturation_rate = 0.23 (with calibrated sat_high)
* under_power_ratio = 0.00
* drift_percent ≈ 1.26e‑07

## Interpretation (short)

Day54 is a **high‑thrust spiral‑in** with near‑instant convergence and negligible steady‑state error. After calibration of thresholds to the real thrust scale, saturation stabilises at ~23%, confirming decisive thrust usage without persistent bang‑bang behaviour.

