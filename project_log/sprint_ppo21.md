# Day21 Validation

## Setup

- Fixed comparison regime: `dt = 100`, `thrust_scale = 10000`, `max_steps = 100000`
- `r0_over_target` sweep: `1.00002, 1.00005, 1.00010, 1.00020`
- Strict tolerances:
  - `tol_r = 0.001`
  - `tol_v = 0.001`
  - `tol_ang = 0.02`

## Stability Sweep | Explicit Controller

- `r0=1.00002`: success `True`, crossings `1`, final_radius_error `7.914e+04`, tail_mean_abs_vr `97.264`
- `r0=1.00005`: success `True`, crossings `1`, final_radius_error `2.766e+04`, tail_mean_abs_vr `48.418`
- `r0=1.00010`: success `False`, crossings `0`, final_radius_error `3.451e+08`, tail_mean_abs_vr `30.397`
- `r0=1.00020`: success `False`, crossings `0`, final_radius_error `1.098e+09`, tail_mean_abs_vr `31.036`

## Controller Comparison

- `explicit_orbit_lock`: success `True`, crossings `1`, final_radius_error `2.766e+04`, tail_mean_abs_vr `48.418`
- `ppo_speed_refine_50`: success `False`, crossings `0`, final_radius_error `7.432e+08`, tail_mean_abs_vr `688.923`

### Radius Error Diagnostic

![radius error](figs/day21_validation/radius_error_vs_time.png)

This plot makes the structural difference visible directly:

- the explicit controller drives the system through the target radius and into closed-loop correction
- PPO remains on one side of the target radius and never enters the capture regime

This supports the main mechanism-level conclusion of the project:

> PPO learns to stop moving, not to stabilize motion.

### Imitation Learning Result

Based on the minimal imitation learning test:

- the cloned MLP achieves low MSE on the dataset
- but fails to produce crossing and orbit insertion

This shows that:

Matching expert actions locally is not sufficient to reproduce
the global phase-structured behavior required for orbit insertion.

This further supports the main conclusion:

> The failure is structural, not due to optimization or model capacity.

## Optional Thrust Generalization | Explicit Controller

- `thrust=8000`: success `False`, crossings `0`, final_radius_error `1.004e+09`, tail_mean_abs_vr `116.479`
- `thrust=10000`: success `True`, crossings `1`, final_radius_error `2.766e+04`, tail_mean_abs_vr `48.418`
- `thrust=12000`: success `False`, crossings `0`, final_radius_error `4.621e+08`, tail_mean_abs_vr `145.712`

## Main Answers

- Stability region under this sweep: success in `2` / `4` explicit-controller setups.
- Crossing appears in `2` / `4` explicit-controller setups.
- Best explicit setup in this sweep: `r0_over_target = 1.00005`.
- Explicit vs PPO on the same setup: explicit is structurally better because it reaches crossing and success, while PPO does not.
- Optional thrust variation result: success in `1` / `3` tested thrust values.

## Structural Diagnosis

- The explicit controller has a narrow but real stability region near the reachable insertion regime.
- PPO remains a continuous reactive policy and does not reproduce the phase-structured crossing behavior on the same setup.
- The main difference is control structure, not just gain magnitude.
