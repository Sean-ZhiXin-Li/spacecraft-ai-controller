# PL35 — Crossing Basin Expansion Failed Under Local Bias Architectures

## Phase35 Pre-Cross Basin Expansion Study

## 1. Objective

Phase34 answered the downstream question:

- what to do after a target-radius crossing exists
- how to convert crossing-producing trajectories into recoverable crossings
- how to use post-cross synchronization as a terminal control law

Phase35 tested the upstream question:

- can more initial conditions be routed into crossing-producing trajectories before the first crossing?

The core scientific question was:

Can local upstream steering biases expand the crossing basin while preserving Phase34 downstream recoverability?

This made Phase35 a pre-cross basin expansion study, not a replacement for Phase34.

## 2. Architecture Setup

The Phase35 architecture was:

```text
pre-cross module
->
first target-radius crossing
->
Phase34 post-cross synchronization
```

Phase34 terminal behavior was intentionally preserved unchanged. The downstream controller remained the Phase34 `radius_priority` post-cross synchronization mode, with the same simulator physics, CAPTURE/LOCK thresholds, reward assumptions, and recoverability thresholds.

Only the upstream behavior before the first target-radius crossing was changed.

## 3. Variants Tested

Four upstream variants were evaluated on the same 24-case reduced benchmark:

- `baseline_phase34`: the Phase34 reference behavior, used to preserve the known 8 crossing-producing cases and provide the direct comparison point.
- `radial_energy_push`: a local pre-cross bias intended to increase useful radial motion toward the target radius before first crossing.
- `tangential_corridor_entry`: a local pre-cross bias intended to adjust tangential velocity and angular-momentum corridor entry before crossing.
- `predictive_crossing_bias`: a simple local action selector that chose among candidate pre-cross actions using a crossing-potential score.

All variants handed off to the Phase34 post-cross synchronization controller after first target-radius crossing.

## 4. Benchmark Results

| Variant | Crossings | Recoverable Crossings | Simulator Success Label |
|---|---:|---:|---:|
| `baseline_phase34` | 8 / 24 | 8 / 24 | 8 / 24 |
| `radial_energy_push` | 0 / 24 | 0 / 24 | 0 / 24 |
| `tangential_corridor_entry` | 0 / 24 | 0 / 24 | 0 / 24 |
| `predictive_crossing_bias` | 8 / 24 | 8 / 24 | 8 / 24 |

The baseline Phase34 architecture produced 8 / 24 geometric crossings and 8 / 24 recoverable crossings.

The `predictive_crossing_bias` variant did not improve the crossing count above the Phase34 baseline. It preserved the same 8 / 24 crossing-producing cases and the same 8 / 24 recoverable crossings.

The `radial_energy_push` variant collapsed crossing performance to 0 / 24 and produced 5 overspeed cases. This indicates that simply adding radial motion toward the target radius can damage the transfer geometry rather than expand it.

The `tangential_corridor_entry` variant also collapsed crossing performance to 0 / 24. This indicates that local tangential correction alone is not sufficient to create new target-radius crossings.

No Phase35 variant improved the crossing count above Phase34.

The success-label column is the simulator-defined success label. It is not a real spacecraft mission-success claim.

## 5. Failure Mode Analysis

The baseline Phase34 non-crossing cases split into two tied dominant labels:

- `near_crossing`: 8 cases
- `over_conservative_transfer`: 8 cases

This is a useful structural diagnosis. The remaining 16 cases are not simply featureless dead cases. Half came close enough that timing or trajectory commitment may matter, and half stayed near the target-radius boundary without committing to crossing.

At the same time, the local-bias variants did not repair these cases. The result suggests that the remaining trajectories are not trivially dead, but also cannot be repaired through simple local steering adjustments.

## 6. Most Important Scientific Insight

Phase35 suggests that crossing-generation is likely a global trajectory geometry problem.

It does not look like:

- simple radial push
- simple tangential-velocity correction
- a local steering artifact

The evidence points toward crossing families as structured trajectory objects. Producing a useful target-radius crossing appears to require the whole transfer arc to be shaped coherently, not just a late local adjustment toward radius or velocity alignment.

Crossing families appear to require:

- long-horizon geometry shaping
- timing coordination
- transfer-family-level structure

This is consistent with the project trajectory from Phase31 through Phase35. Phase34 showed that post-cross synchronization can solve the downstream recoverability problem once crossing exists. Phase35 shows that creating additional crossings is a different upstream problem.

## 7. Interpretation

The negative result is scientifically valuable because it eliminates an incorrect architectural assumption:

> local pre-cross steering is sufficient for crossing-basin expansion.

Phase35 therefore acts as:

- architecture elimination
- upstream bottleneck diagnosis
- hypothesis-space narrowing

It should not be read as a metric result only. The key result is structural: local upstream biases did not create new crossing-producing trajectory families while preserving Phase34 downstream behavior.

## 8. Limitations

Phase35 remains bounded by the same simplified research setting:

- reduced 24-case benchmark
- simplified 2D environment
- no planner-level search
- no true MPC
- no direct trajectory optimization inside Phase35 itself

The result eliminates a local-bias hypothesis under this benchmark. It does not prove that the crossing basin cannot be expanded by stronger long-horizon methods.

## 9. Next Step

Phase36 should likely move away from stronger local biases and toward trajectory-level planning.

Promising directions:

- planner-level transfer architectures
- trajectory-family search
- MPC-lite
- direct trajectory optimization
- long-horizon geometry shaping

The next question should not be how to push harder locally. It should be which transfer family can route a non-crossing initial condition into the Phase34 terminal controller.

## 10. Bottom Line

Phase35 did not expand the crossing basin, but it revealed that crossing-generation is likely a global trajectory-structure problem rather than a local steering problem.
