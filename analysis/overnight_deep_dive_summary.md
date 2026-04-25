# Overnight Explicit-Controller Deep Dive Summary

## What Was Run

Added and ran `scripts/explicit_controller_deep_dive.py` for the same 2D single-orbit explicit-controller scenario.

The script completed:

- local stability basin scan over `r0_over_target = [1.00001, 1.00005, 1.0001, 1.001, 1.005, 1.01]`
- timestep sensitivity scan over `dt = [50, 100, 200, 500]`
- robustness trials with 20 trials per perturbation setting
- orbital physics diagnostics for energy, angular momentum, and phase-wise statistics
- output verification and a small README cleanup

Generated outputs:

- `analysis/local_stability_map.csv`
- `analysis/local_stability_map.png`
- `analysis/dt_sensitivity.csv`
- `analysis/dt_sensitivity.png`
- `analysis/robustness_results.csv`
- `analysis/robustness_summary.md`
- `analysis/energy_vs_time.png`
- `analysis/angular_momentum.png`
- `analysis/phase_statistics.json`
- `analysis/project_audit_phase3_extended.md`
- `analysis/overnight_deep_dive_summary.md`

## Key Findings

The explicit controller is real but sharply local.

On the requested local stability scan, only two initial radius offsets succeeded:

- `r0_over_target=1.00001`: success `True`, crossings `1`, final_radius_error `1.246e5`, tail_mean_abs_vr `108.36`
- `r0_over_target=1.00005`: success `True`, crossings `1`, final_radius_error `2.766e4`, tail_mean_abs_vr `48.42`

All larger tested offsets failed to cross:

- `1.0001`, `1.001`, `1.005`, and `1.01`: success `False`, crossings `0`

## Stability Insights

The stability basin is narrower than the already conservative Phase 3 narrative implied.

The transition between success and failure occurs between:

- successful: `r0_over_target=1.00005`
- failed: `r0_over_target=1.0001`

That means doubling the validated offset from `0.005%` above target to `0.01%` above target is enough to lose first crossing within `100000` steps under the same controller and thrust.

The successful cases are single-crossing insertion successes, not repeated target-radius cycling.

## Robustness Insights

Robustness is strongly perturbation-dependent.

From 20 trials per setting:

- initial velocity noise `plus/minus 1%`: success_rate `0.30`, crossing_rate `0.25`, mean_final_radius_error `5.198e8`
- small position perturbation `plus/minus 1e-5 target_radius`: success_rate `1.00`, crossing_rate `0.75`, mean_final_radius_error `1.355e5`
- small action noise `std=0.005`: success_rate `0.00`, crossing_rate `0.35`, mean_final_radius_error `2.864e8`

The controller tolerates small position perturbations much better than velocity or action perturbations. Action noise is especially damaging because the successful solution relies on a long, coherent retrograde descent followed by a narrow capture/lock window.

## Numerical Sensitivity Insights

The controller is timestep-sensitive.

Requested dt scan:

- `dt=50`: success `False`, crossings `0`, final_radius_error `1.953e8`
- `dt=100`: success `True`, crossings `1`, final_radius_error `2.766e4`
- `dt=200`: success `False`, crossings `0`, final_radius_error `5.417e8`
- `dt=500`: success `False`, crossings `0`, final_radius_error `6.965e9`

Only `dt=100` succeeded in this scan. The result should be interpreted as a coupled controller-plus-integrator behavior, not as a continuous-time robustness proof.

## Orbit Physics Insights

The phase statistics show a highly asymmetric controller:

- `DESCENT`: `48276` steps, action norm almost exactly `1.0`, tangential command `-1.0`
- `CAPTURE`: `17` steps, mean action norm `0.235`, short transition phase near the target radius
- `LOCK`: `203` steps, mean action norm `4.27e-5`, near-zero radial velocity, final radius error around `2.766e4 m`

This confirms that the solution is dominated by a long full-retrograde energy-removal phase, followed by a very short capture adjustment and a low-authority lock phase.

## Top 5 Conclusions About The Controller

1. The explicit phase controller is still the strongest verified controller in this 2D scenario.
2. Its success basin is very narrow: success is lost between `r0_over_target=1.00005` and `1.0001`.
3. The controller is numerically sensitive: only `dt=100` succeeded among `[50, 100, 200, 500]`.
4. Robustness is uneven: small position perturbations are tolerated, but `plus/minus 1%` velocity noise and small action noise sharply reduce success.
5. The controller's mechanism is structurally clear: long full-retrograde descent, brief capture, then very low-authority lock.

## Final Assessment

This deep dive strengthens the project narrative without changing the core problem.

The explicit controller is valid as a local 2D insertion solution, but it should not be described as broadly robust. The next useful work inside the same scenario is not PPO retraining or residual tuning; it is controlled characterization of the narrow basin and the phase-transition conditions that make the single successful regime work.
