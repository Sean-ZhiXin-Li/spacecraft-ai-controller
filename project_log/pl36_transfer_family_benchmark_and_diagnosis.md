# PL36 - Transfer Families Did Not Expand the Crossing Basin

## Phase36B Benchmark and Phase36C Non-Crossing Diagnosis

## 1. Objective

Phase36 tested the upstream problem left open by Phase34 and Phase35.

Phase34 showed that once a target-radius crossing exists, post-cross synchronization can convert crossing-producing cases into recoverable crossings. Phase35 showed that simple local pre-cross biases did not expand the crossing basin.

Phase36 asked a narrower research question:

Can interpretable transfer-family variants route additional initial conditions into Phase34-compatible crossings while preserving the fixed Phase34 terminal controller?

## 2. Background

The project's current architecture is:

```text
pre-cross transfer generation
->
first target-radius crossing
->
Phase34 post-cross synchronization
->
recoverability basin
```

Phase34 is treated as the fixed terminal/post-cross controller. Phase36 does not change Phase34 recovery logic, CAPTURE/LOCK thresholds, physics, or recoverability definitions.

The hypothesis entering Phase36 was that crossing-generation might require trajectory-family structure rather than local radial or tangential steering bias.

## 3. Phase36B Method

Phase36B ran the full 24-case reduced benchmark used by Phase34 and Phase35.

Tested transfer families:

- `baseline_phase34`
- `spiral_approach`
- `grazing_corridor`
- `redesigned_delayed_crossing`

The terminal controller was fixed:

- Phase34 `radius_priority`
- unchanged post-cross synchronization
- unchanged physics
- unchanged CAPTURE/LOCK thresholds
- unchanged recoverability thresholds
- unchanged overspeed and instability checks

The benchmark measured geometric crossings, Phase34-compatible crossings, recoverable crossings, simulator success labels, overspeed, instability, and crossing-state quality.

## 4. Phase36B Results

| Transfer family | Cases | Geometric crossings | Phase34-compatible crossings | Recoverable crossings | Simulator success label | Overspeed | Instability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_phase34` | 24 | 8 | 8 | 8 | 8 | 0 | 0 |
| `spiral_approach` | 24 | 8 | 8 | 8 | 8 | 0 | 0 |
| `grazing_corridor` | 24 | 8 | 8 | 8 | 8 | 0 | 0 |
| `redesigned_delayed_crossing` | 24 | 8 | 8 | 8 | 8 | 0 | 0 |

No Phase36B transfer family improved the crossing count above the Phase34 baseline.

The result is negative but useful: manually designed transfer-family variants changed trajectory-quality metrics, but they did not create additional target-radius crossings.

## 5. Phase36C Diagnosis

Phase36C did not run a new controller. It read the Phase36B CSV outputs and isolated the `16 / 24` baseline non-crossing cases.

Baseline non-crossing failure labels:

| Failure label | Count |
|---|---:|
| `near_crossing` | 8 |
| `over_conservative_transfer` | 8 |

Across non-baseline families on the same non-crossing cases:

- `28` family-case rows improved closest approach.
- `47` family-case rows improved crossing potential.
- `21` family-case rows worsened at least one geometry metric.
- `24` family-case rows changed the diagnostic failure label.

These changes did not produce new crossings.

## 6. Interpretation

Phase36B and Phase36C support a conservative conclusion:

Crossing-generation remains the current bottleneck.

The result does not prove that the crossing basin cannot be expanded. It shows that the tested manually named transfer families did not expand it. The remaining non-crossing cases are not all dead geometry; half are labeled `near_crossing`, which makes them useful candidates for a more structured search.

The project should therefore move from manual family invention to a small parameterized planner-level transfer search.

## 7. Next Step

The next experiment should search a small coarse grid over transfer timing and shaping variables:

- `coast_duration`
- `radial_push_timing`
- `radial_push_magnitude`
- `tangential_shaping_magnitude`

Phase34 `radius_priority` should remain fixed as the terminal/post-cross controller.

This is not MPC-lite yet. The immediate question is whether coarse transfer parameters can create new target-radius crossings at all.

## 8. Bottom Line

Phase36 did not expand the crossing basin, but it narrowed the hypothesis space. The next responsible step is a small parameterized planner-level transfer search for upstream crossing-generation, with Phase34 preserved as the fixed terminal controller.
