# Orbit Lock Phase Controller

## Baseline Setup

- `dt = 50`
- `max_steps = 100000`
- `r0_over_target = 1.00005`

## Main Answers

- Does the phase controller produce repeated crossings? `No`
- Does it stabilize after crossing? `Yes`
- What phase transition behavior is critical? `DESCENT -> CAPTURE` must occur early enough to cross, and `CAPTURE -> LOCK` only matters after radial damping and tangential support are both active near the target.`

## Explicit Controller Metrics

- crossings `1`
- tail_crosses_target_radius `True`
- sustained_crossing_score `1`
- tail_mean_abs_vr `47.098`
- phase_transition_count `2`

## Diagnosis

- The controllers do not sustain phase-aware control after the first crossing, so the trajectory falls back into one-sided drift or a single-pass transit.