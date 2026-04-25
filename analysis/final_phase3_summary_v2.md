# Final Phase 3 Summary v2

## Refined Stability Boundary

The v2 phase map tested `180` grid points and found `38` strict-success points.

The validated `dt=100` line is narrow:

- `r0_over_target=1.00005` succeeds.
- `r0_over_target=1.00006` is the nearest tested failure.

Adaptive boundary v2 confirms that success is first lost on the validated line at `1.00006`.

The boundary is not a smooth global curve. It has timestep-dependent success bands and re-entrant pockets.

## Timestep Sensitivity

The controller is strongly coupled to `dt`.

Successful bands appear around:

- small offsets at `dt=50`
- an isolated pocket near `dt=90`
- the validated band at `dt=100`
- broader pockets around `dt=130..150`

The same `r0_over_target=1.00005` that succeeds at `dt=100` fails at `dt=200`.

## Mechanism Findings

Mechanism v2 compared exactly three cases:

- validated success: `r0=1.00005`, `dt=100`
- nearest boundary failure: `r0=1.00006`, `dt=100`
- dt-induced failure: `r0=1.00005`, `dt=200`

Only the validated success enters `CAPTURE` and `LOCK`.

Phase durations:

- success: `DESCENT=48276`, `CAPTURE=17`, `LOCK=203`
- nearest boundary failure: `DESCENT=100000`, `CAPTURE=0`, `LOCK=0`
- dt-induced failure: `DESCENT=100000`, `CAPTURE=0`, `LOCK=0`

The mechanism is event access: if the long retrograde descent does not reach the crossing event, the controller never leaves `DESCENT`.

## Why Learning Failed

Learning-only policies failed because the task is not one homogeneous feedback law. It requires a long-horizon sequence:

- energy removal
- crossing-triggered phase transition
- capture
- lock

Behavior cloning and short PPO fine-tuning did not reliably reproduce that sequence on rollout.

## Why Residual Failed

Residual approaches failed because small action perturbations can disturb the explicit controller's structure.

Observed Phase 3 residual result:

- zero residual preserves success exactly
- nonzero unconstrained residuals harmed success
- magnitude-only residual kept the accepted checkpoint at zero because nonzero bias did not improve the rollout objective

The safe lesson is structure-preserving residuals only, with strict rollout acceptance gates.

## Top 5 Conclusions

1. The explicit controller is the strongest verified controller in the current 2D setup.
2. Its success basin is narrow at the validated `dt=100` line: failure starts at `r0_over_target=1.00006`.
3. Success is non-monotonic across timestep; `dt` is part of the controller/integrator behavior.
4. Success requires reaching the phase transition. Failures stay in `DESCENT`.
5. Phase 3 is complete for this single-orbit 2D scenario; the next useful step is multi-regime 2D generalization.
