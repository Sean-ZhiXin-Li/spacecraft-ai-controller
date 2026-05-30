# Phase36C Planner Search Space

## Purpose

Phase36C does not define a new controller and does not rerun the simulator. It diagnoses the `16 / 24` baseline Phase36B non-crossing cases and prepares a planner-level search space for upstream crossing-generation in the simplified 2D sandbox.

Phase36B showed that manual transfer-family variants matched the baseline crossing count but did not expand it. The next step should therefore be parameterized trajectory search, not another manually named heuristic family.

## Why Manual Family Invention Is No Longer Enough

The baseline non-crossing set contains `16` cases. Failure labels are: `near_crossing=8`, `over_conservative_transfer=8`.

Across non-baseline families on those same cases, `28` family-case rows improved closest approach, `47` improved crossing potential, and `21` worsened at least one geometry metric. These mixed local changes did not create new target-radius crossings.

This indicates that crossing-generation should be searched as a structured trajectory-geometry problem. Manual labels are useful for interpretation, but the next experiment needs explicit parameters that control timing, energy, angular momentum, and handoff geometry.

## Candidate Search Variables

- `coast_duration`: number of steps or time fraction before active crossing commitment.
- `radial_push_timing`: when radial motion toward target radius is introduced.
- `radial_push_magnitude`: bounded radial component during the shaping or commit phase.
- `tangential_shaping_magnitude`: bounded tangential correction before crossing.
- `crossing_commit_time`: planned transition from shaping to target-radius crossing attempt.
- `angular_momentum_correction_weight`: weight on tangential velocity or angular momentum proxy alignment.
- `energy_correction_weight`: weight on energy-like transfer proximity.
- `max_action_norm`: cap on the pre-cross action vector.
- `handoff_window_length`: allowed window for crossing and subsequent Phase34 compatibility check.

## Candidate Objective Terms

- minimize `min_abs_radius_error_ratio`
- maximize `best_crossing_potential`
- penalize `overspeed`
- penalize `instability`
- penalize bad tangential corridor entry
- reward Phase34-compatible crossing
- reward low crossing sync error
- penalize crossings that occur only with poor Phase34 handoff quality

## Candidate Search Methods

- grid search
- random search
- coarse-to-fine search

Do not use MPC-lite yet. The current need is to map which coarse transfer parameters generate crossings, not to build a receding-horizon controller.

## Recommended First Search

Start with a small coarse grid over only 3 to 4 variables:

1. `coast_duration`
2. `radial_push_timing`
3. `radial_push_magnitude`
4. `tangential_shaping_magnitude`

Keep Phase34 as the fixed terminal controller. Evaluate only whether the candidate parameter set creates target-radius crossings and whether those crossings are Phase34-compatible. Do not tune after seeing the first result set; treat the first grid as a hypothesis test.

## Search Discipline

- Use the same 24-case reduced benchmark.
- Preserve Phase34 post-cross synchronization unchanged.
- Report `simulator_success_label`, not plain success.
- Separate geometric crossing from recoverable crossing.
- Keep near-crossing cases visible instead of hiding them.
- Do not present planner search as real spacecraft validation.

## Family Delta Signals

| Transfer family | Improved closest approach rows | Improved crossing potential rows | Worsened geometry rows |
|---|---:|---:|---:|
| `grazing_corridor` | 12 | 15 | 5 |
| `redesigned_delayed_crossing` | 8 | 16 | 8 |
| `spiral_approach` | 8 | 16 | 8 |

## Bottom Line

The next technical step should be a small parameterized planner search over transfer timing and shaping variables. It should prepare the project for planner-level reasoning without escalating yet to MPC-lite.
