# Final Phase 3 Summary

## What Was Run

Phase 3 final characterization stayed in the same 2D single-orbit explicit-controller scenario.

Completed:

- 2D phase map over 12 `r0_over_target` values and 8 `dt` values.
- Dense boundary refinement over 11 near-boundary `r0_over_target` values and 7 `dt` values.
- Mechanism comparison between one success, one near-boundary failure, and one dt-induced failure.
- Lightweight robustness quick check with 5 trials per perturbation.
- Project audit and low-risk README cleanup.

No PPO was retrained. The explicit controller and environment physics were not redesigned.

## Stability Boundary Findings

The success region is narrow and non-monotonic.

Coarse phase-map results:

- successes: `17 / 96`
- failures: `79 / 96`
- coarse highest successful `r0_over_target`: `1.00007`
- coarse next higher tested failure: `1.00010`

Dense boundary refinement:

- successes: `24 / 77`
- failures: `53 / 77`
- first dense-grid loss after success: `dt=90`, success through `r0_over_target=1.00003`, first failure at `1.00004`
- validated baseline frontier: at `dt=100`, success through `1.00005`, first failure at `1.00006`
- best dense-grid final-radius-error success: `dt=140`, `r0_over_target=1.00003`, final_radius_error `7.205e3`

Important nuance: some strict-success cases have zero sampled radius sign-crossings. These should be interpreted as strict tolerance success without a recorded sampled sign flip, not as repeated orbit crossing.

## Dt Sensitivity Findings

The controller is strongly coupled to timestep.

The coarse phase map found success pockets at:

- `dt=50`: only very small offsets
- `dt=75`: small offsets
- `dt=100`: validated baseline and nearby small offsets
- `dt=125` and `dt=150`: additional success pockets

No coarse-grid successes appeared at `dt=200` or `dt=500`. Some `dt=300` cases crossed but did not satisfy strict success.

Dense refinement confirmed that the boundary is not a simple monotonic curve: `dt=110` and `dt=120` had no successful dense-grid points, while `dt=130` and `dt=140` regained success pockets.

## Robustness Findings

The lightweight quick check used 5 trials per perturbation at the validated baseline:

- velocity noise plus/minus `1%`: success_rate `0.60`, crossing_rate `0.60`
- action noise std `0.001`: success_rate `0.80`, crossing_rate `0.80`
- small position perturbation: success_rate `1.00`, crossing_rate `0.60`

This is not a statistical robustness proof. It shows that the deterministic controller has some tolerance to small perturbations, but success remains fragile and perturbation-dependent.

## Mechanism Findings

Mechanism comparison cases:

- `validated_success`: `r0=1.00005`, `dt=100`, success `True`, crossings `1`, first_crossing_step `48269`
- `near_boundary_failure`: `r0=1.00006`, `dt=100`, success `False`, crossings `0`
- `dt_induced_failure`: `r0=1.00005`, `dt=200`, success `False`, crossings `0`

Phase durations:

- success: `DESCENT=48276`, `CAPTURE=17`, `LOCK=203`
- near-boundary failure: `DESCENT=100000`, `CAPTURE=0`, `LOCK=0`
- dt-induced failure: `DESCENT=100000`, `CAPTURE=0`, `LOCK=0`

The mechanism is clear: success requires a long coherent retrograde descent that reaches the transition, followed by a very short capture phase and low-authority lock. The failures never trigger the crossing-driven phase transition and remain in `DESCENT` until truncation.

## Why Learning Failed

Learning-only policies failed because the task is phase-structured rather than a single homogeneous feedback problem.

The successful behavior depends on:

- long-horizon energy removal
- crossing-triggered transition
- post-crossing capture
- near-target low-authority stabilization

Behavior cloning and short PPO fine-tuning did not reliably reproduce that sequence on rollout. Low one-step action error was not enough to recover the long-horizon phase transition.

## Why Residual Failed

Residual learning failed because perturbing the explicit action can easily disturb the structure that makes the controller work.

Phase 3 residual results showed:

- zero residual exactly preserves explicit-controller success
- alpha sweep showed no effect when residual output was zero
- tiny unconstrained nonzero residuals harmed success
- magnitude-only residual preserved the accepted zero checkpoint, but nonzero magnitude bias did not improve the objective

The implication is that hybrid learning must be structure-preserving and rollout-gated. It cannot freely perturb the full action and should not be accepted unless it preserves success and improves measured objectives.

## Top 5 Conclusions

1. The explicit controller is the strongest verified controller in the current 2D scenario.
2. The success basin is narrow and timestep-dependent, with success lost at `r0=1.00006` for the validated `dt=100` line.
3. The controller succeeds through phase structure: long `DESCENT`, brief `CAPTURE`, then low-authority `LOCK`.
4. Learning-only and naive residual approaches fail because they do not reliably preserve the required phase transition mechanism.
5. Phase 3 is complete for this single-scenario characterization; broader claims require multi-regime testing, not more tuning on the same point.
