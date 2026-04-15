# Reachable Insertion Sweep Summary

## Strict Evaluation Regime

- `tol_r` = `0.001`
- `tol_v` = `0.001`
- `tol_ang` = `0.02`
- `success_threshold` = `200`
- `dt` = `2.0`
- `thrust_scale` = `20000.0`
- `target_radius` = `7500000000000.0`

## Outputs

- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/aggregate_results.csv`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/aggregate_results.json`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/ppo_speed_refine_50_crossings.png`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/ppo_speed_refine_50_final_radius_error.png`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/explicit_orbit_lock_crossings.png`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/explicit_orbit_lock_final_radius_error.png`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/probe_max_retrograde_crossings.png`
- `E:/spacecraft_ai_project/analysis/figs/reachable_insertion_sweep/probe_max_retrograde_final_radius_error.png`

## Main Answers

### Which r0 values are physically reachable under the current thrust and horizon?

- No tested `(r0_over_target, max_steps)` pair produced a real radius crossing even for the max-retrograde reachability probe.

### Does increasing horizon help once the task is reachable?

- `r0=1.0001`: best probe horizon was `20000` with final_radius_error `3.752e+08` and crossings `0`
- `r0=1.0001`: best probe horizon was `20000` with final_radius_error `7.502e+08` and crossings `0`
- `r0=1.0002`: best probe horizon was `20000` with final_radius_error `1.500e+09` and crossings `0`
- `r0=1.0005`: best probe horizon was `4000` with final_radius_error `3.750e+09` and crossings `0`
- `r0=1.0010`: best probe horizon was `4000` with final_radius_error `7.500e+09` and crossings `0`
- `r0=1.0020`: best probe horizon was `4000` with final_radius_error `1.500e+10` and crossings `0`

### What is the smallest setup that produces real orbit crossing / orbit lock?

- None of the tested setups produced real target-radius crossing.

### What exact configuration should become the new project baseline?

- Recommended baseline: controller `explicit_orbit_lock`, `r0_over_target=1.0001`, `max_steps=60000`, strict tolerances as above.
  Metrics: final_radius_error `3.749e+08`, tail_mean_abs_vr `2.014`, crossings `0`, success `False`

## Diagnosis

- Under the stricter regime, no tested start state is already inside the success tolerance.
- The sweep is intended to find the smallest reachable insertion regime before any retraining.
- If the probe controller cannot cross, the task is still unreachable for learned controllers at that setup.