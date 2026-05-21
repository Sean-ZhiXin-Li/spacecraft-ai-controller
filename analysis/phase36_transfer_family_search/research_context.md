# Phase36 Research Context — Transfer Family Search

## 1. Why Phase36 Exists

Phase34 solved the downstream post-cross problem. It showed that if a trajectory already produces a target-radius crossing, a smooth post-cross synchronization mode can convert that crossing-producing case into a recoverable crossing under the current 2D simulator benchmark.

Phase35 then tested the upstream question: can local pre-cross steering biases expand the crossing-producing basin? The result was negative. Local radial push, local tangential corridor correction, and a simple predictive crossing-potential bias did not increase crossing count above the Phase34 baseline.

The project therefore needs trajectory-space exploration rather than another reactive controller. Phase36 should investigate long-horizon transfer families that naturally generate target-radius crossings suitable for Phase34 handoff.

Central hypothesis:

> Crossing-generation is likely a global trajectory-geometry problem, not a local steering problem.

This means crossing-producing trajectories may depend on:

- long-horizon transfer shape
- timing coordination
- energy/angular-momentum evolution
- tangential corridor entry
- controlled coast duration
- family-level trajectory structure

## 2. What Phase35 Ruled Out

Phase35 eliminated the assumption that local upstream steering is sufficient for crossing-basin expansion.

| Variant | Crossings | Recoverable Crossings | Interpretation |
|---|---:|---:|---|
| `baseline_phase34` | 8 / 24 | 8 / 24 | reference terminal architecture |
| `radial_energy_push` | 0 / 24 | 0 / 24 | local radial push damaged transfer geometry |
| `tangential_corridor_entry` | 0 / 24 | 0 / 24 | local vt correction did not create crossings |
| `predictive_crossing_bias` | 8 / 24 | 8 / 24 | matched baseline but did not improve |

The result is not simply that three variants scored poorly. The structural lesson is that crossing-production did not respond to local steering patches. Even the predictive local selector improved the crossing-potential score but did not create new crossing-producing cases.

The Phase35 non-crossing diagnosis also matters. The 16 baseline non-crossing cases split into two tied labels:

- `near_crossing`: 8 cases
- `over_conservative_transfer`: 8 cases

These labels suggest the remaining cases are not featureless dead trajectories. Some approach the target-radius neighborhood, but they do not commit to crossing. That points toward transfer timing and family-level geometry rather than a missing local gain.

## 3. New Scientific Question

Which trajectory families can route non-crossing initial conditions into Phase34-compatible crossing states?

Subquestions:

- What makes a trajectory family crossing-producing?
- Which families approach target radius but fail to commit?
- Which families cross but enter bad sync states?
- Which families are likely compatible with Phase34 terminal recovery?

The objective is not immediate recoverability at first crossing. The objective is to generate crossing states that Phase34 can plausibly convert through post-cross synchronization.

## 4. Candidate Transfer Families

### Family A — Spiral Approach

Hypothesis: A slow inward or outward spiral can gradually reshape radius without forcing a high-speed crossing.

Expected advantage: Gradual radius shaping may reduce overspeed risk and create many opportunities to approach the target-radius band.

Expected failure mode: The trajectory may become over-conservative, orbit near the target radius, and still fail to cross.

Metrics to track:

- crossing_count
- min_abs_radius_error
- closest_approach_step
- crossing_step
- crossing_vr_ratio
- crossing_vt_error_ratio
- energy_error_proxy
- angular_momentum_proxy
- overspeed
- handoff quality into Phase34

### Family B — Delayed Crossing

Hypothesis: Avoiding the earliest crossing opportunity may allow better tangential alignment before committing to the target-radius crossing.

Expected advantage: Later crossings may enter the Phase34 terminal controller with lower tangential error and better radial velocity.

Expected failure mode: The delay may miss the crossing window entirely or drift into the same over-conservative non-crossing behavior seen in Phase35.

Metrics to track:

- controlled coast duration
- crossing_step
- crossing_sync_error
- crossing_vr_ratio
- crossing_vt_error_ratio
- best crossing potential before crossing
- recoverable_crossing_count after handoff

### Family C — Energy Bleed Then Cross

Hypothesis: Reducing excess tangential energy before crossing can create a later target-radius crossing with a better state corridor.

Expected advantage: This may address cases where local tangential correction alone fails because it is applied without a coherent transfer arc.

Expected failure mode: Excess energy bleed can collapse radial motion, over-damp the trajectory, or prevent crossing entirely.

Metrics to track:

- energy_error_proxy over time
- angular_momentum_proxy over time
- burn_duration
- coast_duration
- min_abs_radius_error
- crossing_count
- overspeed
- instability
- Phase34-compatible crossing_count

### Family D — Overshoot Return

Hypothesis: A controlled overshoot past the target-radius region followed by a return crossing may produce better radial-velocity timing than direct approach.

Expected advantage: The return leg can create a second opportunity for crossing with different velocity geometry.

Expected failure mode: Overshoot can become out-of-range, overspeed, or return with excessive radial velocity that Phase34 cannot recover.

Metrics to track:

- maximum radius excursion
- overshoot duration
- return crossing_step
- crossing_vr_ratio
- crossing_vt_error_ratio
- crossing_sync_error
- max_speed_ratio
- instability
- recoverable_crossing_count after handoff

### Family E — Grazing Corridor

Hypothesis: Remaining near the target radius for a long duration may reveal soft crossing windows that are missed by short local heuristics.

Expected advantage: It directly targets Phase35's `near_crossing` and `over_conservative_transfer` labels by searching for a commitment point instead of forcing immediate crossing.

Expected failure mode: The controller may spend long horizons near the target radius without crossing, producing high computation cost and no structural improvement.

Metrics to track:

- time inside near-target band
- min_abs_radius_error
- closest_approach_step
- best crossing potential
- crossing_count
- crossing_step
- crossing_sync_error
- family overspeed rate

### Family F — Two-Stage Transfer Family

Hypothesis: A larger geometry-shaping phase followed by a smaller crossing-commit phase can separate global transfer setup from final crossing production.

Expected advantage: This mirrors the emerging architecture: upstream trajectory family selection first, Phase34 terminal synchronization after crossing.

Expected failure mode: The first phase may create a promising geometry that the second phase fails to convert, or the second phase may destroy the prepared corridor.

Metrics to track:

- geometry-shaping burn_duration
- crossing-commit burn_duration
- coast_duration
- energy/angular-momentum evolution
- crossing_count
- Phase34-compatible crossing_count
- recoverable_crossing_count after handoff
- handoff quality into Phase34

## 5. Phase36 Design Principles

- Phase34 remains the terminal controller.
- Phase36 should generate transfer candidates, not replace post-cross sync.
- The objective is not immediate recoverability; it is Phase34-compatible crossing.
- Avoid local gain tweaking.
- Compare trajectory families, not only individual controllers.

The correct unit of comparison is a transfer family with interpretable geometry, timing, and handoff behavior. A single tuned controller score is less informative than knowing which transfer families create crossings and which fail to commit.

## 6. Proposed Phase36 Metrics

Primary metrics:

- `crossing_count`
- `Phase34-compatible crossing_count`
- `recoverable_crossing_count after handoff`
- `non_crossing_count`

Trajectory-family metrics:

- `min_abs_radius_error`
- `closest_approach_step`
- `crossing_step`
- `crossing_vr_ratio`
- `crossing_vt_error_ratio`
- `crossing_sync_error`
- `energy_error_proxy`
- `angular_momentum_proxy`
- `coast_duration`
- `burn_duration`
- `overspeed`
- `instability`

Family-level metrics:

- mean crossing potential
- best crossing potential
- family success rate
- family overspeed rate
- handoff quality into Phase34

Phase36 should distinguish geometric crossing from Phase34-compatible crossing. A family that increases crossings but produces unrecoverable handoff states is still scientifically useful, but it is not yet the desired architecture.

## 7. Proposed Phase36 Outputs

Recommended implementation directory:

```text
analysis/phase36_transfer_family_search/
```

Potential files:

- `summary.md`
- `phase36_results.csv`
- `transfer_family_dataset.csv`
- `family_comparison.md`
- `phase36_vs_phase35.md`
- `best_family_examples.md`

Potential plots:

- `family_crossing_count_comparison.png`
- `family_recoverable_handoff_comparison.png`
- `transfer_family_geometry_map.png`
- `crossing_state_by_family.png`
- `energy_h_trajectory_family_map.png`
- `best_family_trajectory_examples.png`

## 8. Risks and Honesty Rules

Phase36 should be written so negative results remain useful.

If no transfer family improves crossing count:

> Phase36 did not find a better transfer family under the tested family set. The crossing basin may require planner-level optimization.

If crossing improves but recoverability does not:

> Phase36 expanded geometric crossing but did not produce Phase34-compatible crossing states.

If both improve:

> Phase36 found a transfer family that feeds new trajectories into the Phase34 terminal controller.

The report should avoid claiming that a geometric crossing is insertion. It should also avoid treating a higher crossing count as sufficient unless Phase34 handoff quality improves.

## 9. Recommended Phase36 Implementation Direction

Recommended order:

1. Transfer-family search.
2. Handoff evaluation into Phase34.
3. MPC-lite only if family search identifies promising geometry.
4. Direct optimization only after family-level structure is understood.

Transfer-family grid search should come first because it is interpretable and directly tests the Phase35 conclusion. MPC-lite should be second because it can exploit promising geometry once the family space is better understood. Direct trajectory optimization should come later because it is more powerful but less explanatory if used before the project understands which transfer structures matter.

Phase36 should therefore start by mapping families, not by adding stronger local gains.

## 10. Bottom Line

Phase36 should move the project from local controller biasing to transfer-family discovery, because Phase35 showed that crossing-generation is a global trajectory-structure problem.
