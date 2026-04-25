# Robustness Summary

## Setup

- trials per setting: `20`
- baseline: `dt=100.0`, `max_steps=100000`, `r0_over_target=1.00005`, `thrust_scale=10000.0`
- perturbations are applied without changing environment physics or controller gains

## Results

- `initial_velocity_noise_pm_1pct`: success_rate `0.30`, crossing_rate `0.25`, mean_final_radius_error `5.198e+08`, std_final_radius_error `6.105e+08`, mean_tail_mean_abs_vr `1.438e+02`
- `small_position_perturbation`: success_rate `1.00`, crossing_rate `0.75`, mean_final_radius_error `1.355e+05`, std_final_radius_error `7.499e+04`, mean_tail_mean_abs_vr `5.210e+01`
- `small_action_noise`: success_rate `0.00`, crossing_rate `0.35`, mean_final_radius_error `2.864e+08`, std_final_radius_error `3.137e+08`, mean_tail_mean_abs_vr `3.084e+02`

## Interpretation

- Most robust setting in this scan: `small_position_perturbation`.
- Weakest setting in this scan: `small_action_noise`.
- Treat these as local robustness checks around the validated 2D baseline, not as a global robustness proof.