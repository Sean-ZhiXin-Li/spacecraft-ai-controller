# Orbit Lock Phase Controller

## Baseline Setup

- `dt = 100`
- `max_steps = 100000`
- `r0_over_target = 1.00005`

## Main Answers

- Does the phase controller produce repeated crossings? `No`
- Does it stabilize after crossing into the target band? `Yes`
- What phase transition behavior is critical? `DESCENT -> CAPTURE` must occur early enough to cross, and `CAPTURE -> LOCK` only matters after radial damping and tangential support are both active near the target.`

## Explicit Controller Metrics

- crossings `1`
- tail_crosses_target_radius `True`
- sustained_crossing_score `1`
- tail_mean_abs_vr `48.418`
- phase_transition_count `2`

## Diagnosis

- The explicit controller achieves a successful single-crossing insertion on this baseline, but it does not demonstrate repeated orbit-lock cycling across the target radius.