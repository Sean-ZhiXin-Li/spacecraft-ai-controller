# Orbit Lock Validation

## Baseline Setup

- `dt = 50`
- `max_steps = 100000`
- `r0_over_target = 1.00005`
- `thrust_scale = 20000.0`
- Strict tolerances:
  - `tol_r = 0.001`
  - `tol_v = 0.001`
  - `tol_ang = 0.02`

## Controllers

- `probe_max_retrograde`
- `explicit_orbit_lock`
- `ppo_speed_refine_50`

## Results

- `probe_max_retrograde`: crossings `1`, tail_crosses `True`, first_crossing_step `96173`, tail_mean_abs_vr `46.367`, amplitude_shrinks `True`, final_radius_error `8.143e+06`
- `explicit_orbit_lock`: crossings `1`, tail_crosses `True`, first_crossing_step `96173`, tail_mean_abs_vr `47.098`, amplitude_shrinks `True`, final_radius_error `3.143e+04`
- `ppo_speed_refine_50`: crossings `0`, tail_crosses `False`, first_crossing_step `None`, tail_mean_abs_vr `690.724`, amplitude_shrinks `True`, final_radius_error `5.621e+08`

## Main Answers

- Can any controller achieve sustained orbit lock? `No`
- Best controller under this setup: `probe_max_retrograde`
- What prevents stability after crossing: The controllers do not sustain phase-aware control after the first crossing, so the trajectory falls back into one-sided drift or a single-pass transit.
- What control behavior is missing: The controllers do not sustain phase-aware control after the first crossing, so the trajectory falls back into one-sided drift or a single-pass transit.

## Diagnosis

- No controller maintains tail crossings. The failure is not reachability anymore; it is post-crossing state regulation.
- Probe crossing count: `1`
- Explicit crossing count: `1`
- PPO crossing count: `0`