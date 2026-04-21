# Orbit Lock Validation

## Baseline Setup

- `dt = 100`
- `max_steps = 100000`
- `r0_over_target = 1.00005`
- `thrust_scale = 10000.0`
- Strict tolerances:
  - `tol_r = 0.001`
  - `tol_v = 0.001`
  - `tol_ang = 0.02`

## Controllers

- `probe_max_retrograde`
- `explicit_orbit_lock`
- `ppo_speed_refine_50`

## Results

- `probe_max_retrograde`: crossings `1`, tail_crosses `False`, first_crossing_step `48269`, tail_mean_abs_vr `30.103`, amplitude_shrinks `True`, final_radius_error `3.116e+07`
- `explicit_orbit_lock`: crossings `1`, tail_crosses `True`, first_crossing_step `48269`, tail_mean_abs_vr `48.418`, amplitude_shrinks `True`, final_radius_error `2.766e+04`
- `ppo_speed_refine_50`: crossings `0`, tail_crosses `False`, first_crossing_step `None`, tail_mean_abs_vr `688.923`, amplitude_shrinks `True`, final_radius_error `7.432e+08`

## Main Answers

- Can any controller achieve sustained orbit lock? `No`
- Best controller under this setup: `explicit_orbit_lock`
- What prevents stability after crossing: The explicit controller achieves a successful single-crossing insertion on this baseline, but it does not demonstrate repeated orbit-lock cycling across the target radius.
- What control behavior is missing: The explicit controller achieves a successful single-crossing insertion on this baseline, but it does not demonstrate repeated orbit-lock cycling across the target radius.

## Diagnosis

- No controller shows repeated target crossings into the tail. Reachability is solved for the explicit controller, but repeated post-crossing cycling is still missing.
- Probe crossing count: `1`
- Explicit crossing count: `1`
- PPO crossing count: `0`