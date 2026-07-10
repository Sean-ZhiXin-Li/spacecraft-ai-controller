# Recoverability Benchmark v1

## Status

Week 1 milestone document.

Completed: 2026-07-04

Scope: post-paper baseline for the 2D spacecraft orbital-control research platform.

This document freezes the current scientific baseline after the IAI2O paper and quad-chart stage. It explains what the repository evidence supports, what it does not support, and what future experiments must report before they can be treated as benchmark progress.

The central lesson is:

````text
Crossing is not insertion.
````

The broader benchmark principle is:

````text
Intermediate success is not recoverable task completion.
````

## Benchmark Purpose

Recoverability Benchmark v1 exists to prevent future experiments from confusing proxy progress with real task progress.

The benchmark has four near-term jobs:

- Preserve the current Phase34/36/37 evidence baseline.
- Separate target-radius crossing from recoverable crossing and final simulator success.
- Make negative and diagnostic results usable without overstating them.
- Define reporting requirements for future controller, planner, learning, and runtime-assurance experiments.

This is a 2D simulator benchmark. It is not real spacecraft validation, not hardware validation, and not a sim-to-real result.

## Current Scientific Claim

The current repository evidence supports this claim:

Phase34-style post-cross synchronization improved recoverability for the existing crossing-producing cases in the reduced 24-case 2D benchmark. In that comparison, the reduced Phase31-style reference produced `8 / 24` target-radius crossings and `0 / 24` recoverable crossings, while Phase34 `radius_priority` preserved `8 / 24` crossings and produced `8 / 24` recoverable crossings.

The current repository evidence also supports this limitation:

Upstream crossing generation remains unresolved. Phase36B, Phase36C, Phase37A, and Phase37B did not establish new full-benchmark crossing or recoverable-crossing wins beyond the Phase34 baseline.

## Non-Claims

Do not use this benchmark to claim:

- Flight readiness.
- Real spacecraft validation.
- Hardware readiness.
- Sim-to-real transfer.
- Docking readiness.
- Robotics or plug-insertion validation.
- That Phase34 solves orbital insertion in general.
- That upstream crossing generation is solved.
- That learning baselines solve the task unless they beat explicit-controller baselines under the same benchmark.
- That subset results are full-benchmark wins.
- That closest approach is success.
- That crossing count is recoverability.
- That simulator `CAPTURE`, `LOCK`, or `success` labels are mission success.

## Benchmark Scope

The current protected evidence centers on the reduced 24-case benchmark used across Phase34, Phase36B, and Phase37A-style comparisons:

- `r0_over_target`: `0.98`, `1.00`, `1.02`
- `initial_velocity_angle_deg`: `150`, `165`, `170`, `175`
- `thrust_scale`: `8000`, `10000`

Phase37B is a subset diagnostic, not a full 24-case benchmark win. Its selected-case and regression-case results must remain labeled as subset evidence.

## Definitions

### Target-Radius Crossing

A target-radius crossing is a geometric event where the simulated trajectory reaches or crosses the target radius according to the experiment's crossing detector.

Crossing is an intermediate event. It does not by itself imply recoverability, stable insertion, CAPTURE, LOCK, or final success.

### Recoverable Crossing

A recoverable crossing is a target-radius crossing followed by entry into the simulator-defined recoverability basin or successful post-cross continuation under the specified downstream controller, horizon, thresholds, and failure rules.

Recoverable crossing is controller-relative and benchmark-relative. It is not an absolute statement that any possible controller could recover from the state.

### Final Simulator Success

Final simulator success is the simulator's task-completion label under the experiment's current termination and success criteria.

It must be reported as a simulator-defined label. It must not be described as mission success, real insertion, flight validation, or hardware success.

### Closest Approach

Closest approach is the minimum distance or minimum absolute radius error reached during a rollout.

Closest approach is diagnostic. It can indicate that a controller moved closer to the target region, but it is not a crossing, not recoverability, and not final task success.

### Overspeed

Overspeed is a simulator-defined safety or failure condition where velocity exceeds the experiment's configured speed threshold.

Overspeed must be reported separately from crossing and recoverability. A crossing that requires overspeed is not an accepted recoverability improvement.

### Instability

Instability is a simulator-defined failure condition indicating divergent, oscillatory, invalid, or otherwise unacceptable closed-loop behavior under the experiment's criteria.

Instability must be reported separately from crossing and recoverability. A controller that improves a proxy metric while increasing instability has not produced clean benchmark progress.

### Diagnostic Subset Result

A diagnostic subset result is an experiment run on selected cases rather than the full benchmark.

Subset results can guide mechanism analysis and future experiment design. They cannot be reported as full-benchmark progress unless the same change is evaluated on the full benchmark and preserves known-success regression cases.

### Accepted Benchmark Progress

A future result can be considered accepted benchmark progress only if it:

- Runs on the declared benchmark or clearly states that it is a diagnostic subset.
- Reports crossing count and recoverable-crossing count separately.
- Preserves known Phase34 recoverable cases before claiming improvement.
- Reports overspeed and instability.
- Reports termination or failure labels.
- Improves a primary benchmark metric, not only a proxy metric.
- Does not convert subset evidence into full-benchmark claims.

## Current Protected Evidence Summary

The protected guard is `python scripts/check_phase_results.py`. It reads the historical CSV artifacts and checks current aggregate claims.

### Phase34: Post-Cross Synchronization

Protected sources:

- `analysis/phase34_post_cross_sync/summary.md`
- `analysis/phase34_post_cross_sync/phase34_results.csv`
- `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`

Guarded results:

| Configuration | Cases | Crossings | Recoverable crossings |
| --- | ---: | ---: | ---: |
| Reduced Phase31-style baseline | 24 | 8 | 0 |
| Phase34 `radius_priority` | 24 | 8 | 8 |

Interpretation:

Phase34 is the current terminal-controller foundation. It improved post-cross recoverability for crossing-producing cases. It did not expand the crossing basin because the crossing count stayed `8 / 24`.

### Phase36B: Transfer-Family Benchmark

Protected sources:

- `analysis/phase36b_transfer_family_benchmark/summary.md`
- `analysis/phase36b_transfer_family_benchmark/phase36b_results.csv`
- `analysis/phase36b_transfer_family_benchmark/phase36b_family_summary.csv`

Guarded families:

- `baseline_phase34`
- `grazing_corridor`
- `redesigned_delayed_crossing`
- `spiral_approach`

Guarded results:

| Family | Cases | Crossings | Recoverable crossings | Overspeed | Instability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_phase34` | 24 | 8 | 8 | 0 | 0 |
| `grazing_corridor` | 24 | 8 | 8 | 0 | 0 |
| `redesigned_delayed_crossing` | 24 | 8 | 8 | 0 | 0 |
| `spiral_approach` | 24 | 8 | 8 | 0 | 0 |

Interpretation:

Phase36B narrowed the transfer-family hypothesis space. No tested transfer family expanded crossing count or recoverable-crossing count beyond the Phase34 baseline.

### Phase36C: Non-Crossing Geometry Diagnosis

Protected sources:

- `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`
- `analysis/phase36c_non_crossing_geometry_diagnosis/non_crossing_case_set.csv`
- `analysis/phase36c_non_crossing_geometry_diagnosis/planner_search_space.md`

Guarded results:

| Category | Count |
| --- | ---: |
| Baseline non-crossing cases | 16 |
| `near_crossing` labels | 8 |
| `over_conservative_transfer` labels | 8 |

Interpretation:

Phase36C explains the unresolved non-crossing set. It is diagnostic evidence that some geometry and closest-approach metrics changed without creating new target-radius crossings.

### Phase37A: Radial Commitment Timing Sweep

Protected sources:

- `analysis/phase37a_radial_commit_timing/phase37a_results.csv`
- `analysis/phase37a_radial_commit_timing/phase37a_summary.md`
- `analysis/phase37a_radial_commit_timing/phase37a_comparison.png`
- `project_log/phase37a_radial_commit_timing.md`

Guarded results:

| Metric | Result |
| --- | ---: |
| Total rows | 144 |
| New crossings on baseline non-crossing cases | 0 |
| Total overspeed | 0 |
| Total instability | 0 |
| `delayed_commit_low` crossings | 8 / 24 |
| `delayed_commit_low` recoverable crossings | 8 / 24 |
| `delayed_commit_medium` crossings | 8 / 24 |
| `delayed_commit_medium` recoverable crossings | 8 / 24 |

Interpretation:

Delayed radial commitment preserved known recoverable behavior, but Phase37A did not create new crossings on the baseline non-crossing cases.

### Phase37B: Weak Tangential Subset Diagnostic

Protected sources:

- `analysis/phase37b_weak_tangential_subset/phase37b_results.csv`
- `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`
- `analysis/phase37b_weak_tangential_subset/phase37b_comparison.png`
- `project_log/phase37b_weak_tangential_postmortem.md`

Guarded results:

| Group | Cases | Crossings | Recoverable crossings |
| --- | ---: | ---: | ---: |
| Weak selected cases | 4 | 0 | 0 |
| Weak regression cases | 8 | 4 | 4 |

Additional guarded results:

- Total rows: `24`
- Total overspeed: `0`
- Total instability: `0`

Interpretation:

Phase37B is a subset diagnostic. Weak tangential shaping did not create selected-case crossings and did not preserve all regression crossings. Any closest-approach improvement remains diagnostic, not accepted benchmark progress.

## Why Crossing Count Is Not Enough

Crossing count measures only whether a trajectory reached the target-radius event. It does not measure whether the state at or after that event can be stabilized or converted into simulator-defined task completion.

The Phase34 comparison shows why this matters:

- The reduced Phase31-style baseline produced `8 / 24` crossings but `0 / 24` recoverable crossings.
- Phase34 `radius_priority` produced the same `8 / 24` crossings but `8 / 24` recoverable crossings.

The crossing count alone would hide the main scientific result.

## Why Closest Approach Is Not Enough

Closest approach can improve while the controller still fails to cross, fails to recover after crossing, or damages regression preservation.

Closest approach should be treated as a diagnostic metric for search-space design, not as a primary success metric. A result that improves closest approach but produces no new crossings, no new recoverable crossings, or regression on known recoverable cases is false progress unless clearly labeled as diagnostic.

## Why Subset Improvement Is Not Full-Benchmark Progress

Subset experiments are useful for mechanism analysis. They are not full benchmark results.

A subset result can be reported only as:

- selected-case diagnostic evidence,
- regression-case diagnostic evidence,
- hypothesis-generation evidence, or
- preparation for a full benchmark run.

It cannot be reported as accepted benchmark progress until the same controller or rule is evaluated on the declared benchmark and passes regression protection for known Phase34 recoverable cases.

## Known Bottleneck

The known bottleneck is upstream crossing generation.

Phase34 improved post-cross synchronization for cases that already cross. Phase36B transfer-family variants, Phase36C geometry diagnosis, Phase37A radial timing, and Phase37B weak tangential shaping did not solve the baseline non-crossing set. Future work should treat new crossing generation and new recoverable crossing generation as separate claims.

## Protected Files And Evidence Artifacts

Do not overwrite these files casually:

- `analysis/artifact_manifest.md`
- `docs/benchmark_contract.md`
- `scripts/check_phase_results.py`
- `analysis/phase34_post_cross_sync/summary.md`
- `analysis/phase34_post_cross_sync/phase34_results.csv`
- `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`
- `analysis/phase36b_transfer_family_benchmark/summary.md`
- `analysis/phase36b_transfer_family_benchmark/phase36b_results.csv`
- `analysis/phase36b_transfer_family_benchmark/phase36b_family_summary.csv`
- `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`
- `analysis/phase36c_non_crossing_geometry_diagnosis/non_crossing_case_set.csv`
- `analysis/phase36c_non_crossing_geometry_diagnosis/planner_search_space.md`
- `analysis/phase37a_radial_commit_timing/phase37a_results.csv`
- `analysis/phase37a_radial_commit_timing/phase37a_summary.md`
- `analysis/phase37b_weak_tangential_subset/phase37b_results.csv`
- `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`

If a benchmark is rerun, write outputs to a new artifact directory or document the regeneration explicitly. Do not silently replace historical evidence.

## Required Reporting Fields For Future Experiments

Every future experiment that claims progress against this benchmark should report at least:

- `benchmark_id`
- `benchmark_version`
- `controller_id`
- `controller_family`
- `experiment_id`
- `case_id`
- `r0_over_target`
- `initial_velocity_angle_deg`
- `thrust_scale`
- `crossed_target_radius`
- `crossing_time` or `first_crossing_step`
- `state_at_crossing` or a documented crossing-state summary
- `recoverable_crossing`
- `final_simulator_success`
- `closest_approach`
- `max_speed`
- `overspeed`
- `instability`
- `termination_label`
- `dominant_failure_label`
- `control_effort` if available
- `fuel_proxy` if available
- `representative_subset_note` when applicable
- `regression_set_membership`
- `accepted_as_progress`
- `acceptance_reason`
- `artifact_path`

Recommended but not required for v1:

- `recovery_time`
- `recovery_cost`
- `minimum_recovery_margin`
- `phase34_compatible_crossing`
- `controller_mode_at_crossing`
- `post_cross_mode`
- `seed`
- `git_commit`
- `environment_summary`

## Regression Rule

Future controller changes must preserve known Phase34 recoverable cases before claiming progress.

Minimum rule:

- If a controller claims to improve upstream crossing generation, it must also report performance on the known Phase34 crossing-producing cases.
- A result that creates proxy improvement on selected non-crossing cases but regresses known recoverable cases is diagnostic only.
- A result that increases crossing count but does not increase recoverable-crossing count must be reported as crossing-only progress, not recoverability progress.
- A result that increases recoverable crossings while increasing overspeed or instability must report the tradeoff and cannot be treated as clean progress without justification.

## False-Progress Refusal Rules

The benchmark refuses to count the following as accepted progress by themselves:

- Better closest approach without new target-radius crossings.
- More target-radius crossings without more recoverable crossings.
- Subset gains without full-benchmark evaluation.
- Learning loss reduction without rollout improvement under the same benchmark.
- Higher reward without improved crossing, recoverability, safety, or final simulator outcome.
- Perception or state-estimation accuracy without improved task recoverability.
- Controller changes that improve selected failures but regress known Phase34 recoverable cases.
- Results that omit overspeed, instability, or termination labels.
- Results that move thresholds after seeing outcomes.

Diagnostic results are still valuable. They should be preserved and labeled as diagnostic rather than converted into benchmark claims.

## Acceptance Checklist For Future Results

Before a future result is described as benchmark progress, answer:

- Did it run on the full benchmark or is it clearly labeled as a subset diagnostic?
- Did it preserve known Phase34 recoverable cases?
- Did crossing count improve?
- Did recoverable-crossing count improve?
- Did overspeed remain controlled?
- Did instability remain controlled?
- Are failure labels reported?
- Are artifact paths recorded?
- Are proxy metrics separated from primary metrics?
- Is the claim scoped to the 2D simulator?

If any answer is missing, the result should remain diagnostic until resolved.

## Open Questions For Week 2 And Later

Week 2 failure-label work should answer:

- What is the standard termination-label priority order?
- How should labels distinguish `no_crossing`, `crossing_unrecoverable`, `recoverable_crossing_failed_late`, `overspeed`, `instability`, `timeout`, `resource_depletion`, `invalid_simulation`, and `unknown`?
- Should precursor labels be recorded separately from terminal labels?
- How should ambiguous cases be labeled without inventing false precision?

Later benchmark/schema work should answer:

- What is the minimal `state_at_crossing` representation that is useful and stable?
- Can recovery margin be computed consistently across Phase34-style and future controllers?
- What fuel or control-effort proxy is easiest to populate without refactoring old scripts?
- How should future held-out or randomized 2D benchmarks relate to the protected 24-case benchmark?
- What small regression gate should run before any new controller experiment is accepted?

## Week 1 Baseline Rule

Week 1 is complete when this document exists, the protected regression guard still passes, and no historical evidence has been modified.

