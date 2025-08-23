# Project Log — Day 39

**Date:** 2025-08-23  
**Focus:** Baseline fast run (reuse pipeline)

---

## Setup
- **Script:** `tools/day39_quickrun.py`
- **Tasks:** `ab/day36/task_specs_fast` (limited to 64 tasks)
- **Controllers:**  
  - `expert:elliptic_strong`  
  - `expert:transfer_2phase`  
  - `expert:spiral_in`  
- **Output root:** `ab/day39`
- **Command used:**
  ```powershell
  python tools/day39_quickrun.py `
    --tasks_dir ab\day36\task_specs_fast `
    --out_dir ab\day39 `
    --controllers elliptic_strong transfer_2phase spiral_in `
    --limit 64
  ```

---

## Outputs
- **CSV:**  
  - `ab/day39/csv/baseline_fast.csv`  
  - `ab/day39/csv/summary.csv`
- **Figures:**  
  - `ab/day39/figs/day39_r_err_by_controller.png`  
  - `ab/day39/figs/day39_return_by_controller.png`

---

## Observations
- **elliptic_strong**
  - Very stable across circular and elliptic tasks.
  - `r_err` often as low as `6.9e-05 ~ 5.9e-04`.
  - Always `agent_success=1` in top summary entries.
- **transfer_2phase**
  - Generally accurate (`r_err` ~1e-04 on many tasks).
  - Achieved lowest error on transfer-type tasks (e.g., `4.0e-05`).
  - Some failures observed in tight-radius transfers (`r_err > 3e-03`).
- **spiral_in**
  - Mixed performance: successful in some large-radius and transfer cases.
  - Higher error in tighter orbits (`r_err > 1e-03`).
  - Less stable compared to the other two.

---

## Summary Preview
```text
   name             controller    r_err       ret  agent_success
task_42 expert:elliptic_strong 0.000069  -480.096              1
task_39 expert:elliptic_strong 0.000112  -300.060              1
task_33 expert:elliptic_strong 0.000136  -300.060              1
task_30 expert:elliptic_strong 0.000151  -420.084              1
task_57 expert:elliptic_strong 0.000168  -480.096              1
```

---

## Aggregated Results
| Controller          | Mean r_err | Mean return | Success rate |
|---------------------|------------|-------------|--------------|
| elliptic_strong     | ~1e-04     | -300 ~ -4800 | ~100%        |
| transfer_2phase     | ~1e-04–1e-03 | -600 ~ -6000 | ~85–90%      |
| spiral_in           | >1e-03     | -240 ~ -3600 | ~70–80%      |

---

## Notes
- Day39 run confirms the pipeline is fully reusable with a new output root.  
- Results align with expectations from Day36–38, with consistent baseline behavior.  
- Figures generated for quick visual comparison between controllers.

---

## Next Steps
- (Optional) Run `replay_worst_complex` on Day39 baseline to capture hardest tasks.  
- (Optional) Export aggregated controller stats into `agg_by_controller.csv`.  
- Prepare for Day40: README writing and summarizing progress so far.
