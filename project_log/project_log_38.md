# Project Log — Day 38 (Quick Baseline)

**Date:** 2025-08-22  
**Focus:** Quick expert comparison on a small fixed task set (≤1h)

## What I Ran
- Experts: `elliptic_strong`, `transfer_2phase`, `spiral_in`
- Tasks: fast task bundle (fallback to `ab/day36/task_specs_fast`)
- Pipeline:
  1) `script.run_baseline_complex` → `baseline_fast.csv`
  2) `make_summary_day37.py` → `summary.csv`
  3) `plot_day37_figs.py` → bar charts
  4) (Optional) `script.replay_worst` for worst-3 replay

## Key Outputs
- CSV: `ab/day38/csv/baseline_fast.csv`
- Summary: `ab/day38/csv/summary.csv`
- Figures:
  - `ab/day38/figs/day37_r_err_by_controller.png`
  - `ab/day38/figs/day37_return_by_controller.png`
- Replay (opt): `ab/day38/replay/`

## Observations
- `elliptic_strong`: lowest r_err on average; stable across tasks.
- `transfer_2phase`: slightly higher r_err; returns lower due to cost terms.
- `spiral_in`: occasional failures on tighter tolerances; higher r_err variance.

## Issues
- `eval.summary` missing → replaced with `tools/make_summary_day37.py`.
- Initial `ab/day38/task_specs_fast` not present → fell back to `day36` bundle.
