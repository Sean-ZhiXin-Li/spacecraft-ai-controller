# Recoverability Regression Policy v0

## Status

Week 4 milestone document.

Completed: 2026-07-08

Scope: future controller, planner, learning-baseline, and runtime-assurance experiments for the 2D spacecraft recoverability benchmark.

This document defines a regression policy that protects known successful recoverability behavior before any future experiment can claim progress. It is a documentation and policy artifact only. It does not edit historical CSVs, change Phase34/36/37 scripts, change controller logic, regenerate old analysis artifacts, or implement a full experiment manager.

The policy principle is:

```text
Future experiments must not destroy known recoverable cases while chasing new proxy improvements.
```

The operational rule is:

```text
A new controller cannot claim progress unless it preserves the known Phase34 recoverable cases and reports crossing, recoverability, safety, labels, and subset status separately.
```

## Purpose

The purpose of this policy is to prevent false progress.

Future experiments should be able to explore new transfer logic, planners, learning baselines, and runtime-assurance wrappers. But those experiments must not claim benchmark progress if they:

- lose known Phase34 recoverable cases,
- improve closest approach without crossing,
- create crossings that are not recoverable,
- improve a subset while damaging regression cases,
- hide overspeed or instability,
- omit failure labels,
- omit subset or regression status.

The policy protects current known-success behavior while keeping diagnostic failures scientifically useful.

## Relationship To Existing Benchmark Documents

### Recoverability Benchmark v1

Recoverability Benchmark v1 defines the scientific baseline:

- crossing is not insertion,
- intermediate success is not recoverable task completion,
- closest approach is diagnostic,
- subset results are not full-benchmark wins,
- known Phase34 recoverable cases must be preserved before broader progress is claimed.

This policy turns those benchmark principles into regression-protection rules.

### Failure Label Taxonomy v0

Failure Label Taxonomy v0 defines controlled terminal labels:

- `success`
- `no_crossing`
- `crossing_unrecoverable`
- `recoverable_crossing_failed_late`
- `overspeed`
- `instability`
- `timeout`
- `resource_depletion`
- `unsafe_state`
- `invalid_simulation`
- `unknown`

This policy requires future progress claims to report valid `terminal_label` values, plus `precursor_labels`, `diagnostic_labels`, and `manual_audit_note` when needed. Labels must describe mechanism and must not be replaced by phase-specific diagnostic labels such as `near_crossing`.

### Result Schema v1

Result Schema v1 defines the row-level fields future artifacts should use.

This policy depends on the following schema concepts:

- crossing and recoverability are separate fields,
- final simulator success is simulator-defined,
- safety fields are explicit,
- failure labels are controlled,
- subset status and regression status are explicit,
- `accepted_as_progress` and `acceptance_reason` explain whether a row contributes to a claim.

The policy does not require historical CSVs to be rewritten into Result Schema v1.

### `scripts/check_phase_results.py`

The protected guard `python scripts/check_phase_results.py` verifies the current historical facts from Phase34, Phase36B, Phase36C, Phase37A, and Phase37B artifacts.

This policy treats that guard as the first historical baseline check. A future regression gate should run it before validating any new result artifact.

The current guard reads legacy fields such as:

- Phase34: `controller_name`, `post_cross_mode`, `crossing_occurs`, `crossing_step`, `recoverable_crossing`, `success`, `overspeed`.
- Phase36B: `transfer_family`, `crossing_occurs`, `first_crossing_step`, `phase34_compatible_crossing`, `recoverable_crossing`, `dominant_failure_label`, `overspeed`, `instability`, `simulator_success_label`.
- Phase37A: `variant_name`, `baseline_non_crossing_case`, `crossing_occurs`, `first_crossing_step`, `phase34_compatible_crossing`, `recoverable_crossing`, `new_crossing_on_baseline_non_crossing_case`, `dominant_failure_label`, `overspeed`, `instability`, `simulator_success_label`.
- Phase37B: `case_id`, `group`, `setting`, `crossing_occurs`, `phase34_compatible_crossing`, `recoverable_crossing`, `simulator_success_label`, `overspeed`, `instability`, `failure_label`.

Future normalized artifacts should use Result Schema v1 fields, not rely on these legacy names except through explicit compatibility rules.

## Protected Evidence Table

These facts are protected by `scripts/check_phase_results.py` and are the baseline for this policy.

| Phase | Protected fact | Interpretation |
| --- | --- | --- |
| Phase34 `radius_priority` | 24 cases, 8 crossings, 8 recoverable crossings | Phase34 improved post-cross recoverability for crossing-producing cases. |
| Phase34 reduced Phase31-style baseline | 24 cases, 8 crossings, 0 recoverable crossings | Crossing existed, but recoverability did not. |
| Phase34 comparison | Phase34 kept 8 crossings while improving recoverable crossings from 0 to 8 | Phase34 did not expand crossing generation. |
| Phase36B `baseline_phase34` | 24 cases, 8 crossings, 8 recoverable crossings, 0 overspeed, 0 instability | Transfer-family baseline preserved Phase34 behavior. |
| Phase36B `grazing_corridor` | 24 cases, 8 crossings, 8 recoverable crossings, 0 overspeed, 0 instability | No full-benchmark improvement beyond Phase34. |
| Phase36B `redesigned_delayed_crossing` | 24 cases, 8 crossings, 8 recoverable crossings, 0 overspeed, 0 instability | No full-benchmark improvement beyond Phase34. |
| Phase36B `spiral_approach` | 24 cases, 8 crossings, 8 recoverable crossings, 0 overspeed, 0 instability | No full-benchmark improvement beyond Phase34. |
| Phase36C baseline non-crossing set | 16 cases, 8 `near_crossing`, 8 `over_conservative_transfer` | Diagnostic geometry evidence, not accepted crossing or recoverability progress. |
| Phase37A all rows | 144 total rows, 0 new crossings on baseline non-crossing cases, 0 overspeed, 0 instability | Radial timing variants did not solve baseline non-crossing cases. |
| Phase37A `delayed_commit_low` | 8 / 24 crossings, 8 / 24 recoverable crossings | Preserved known behavior but did not expand crossing generation. |
| Phase37A `delayed_commit_medium` | 8 / 24 crossings, 8 / 24 recoverable crossings | Preserved known behavior but did not expand crossing generation. |
| Phase37B all rows | 24 total rows, 0 overspeed, 0 instability | Subset diagnostic only. |
| Phase37B weak selected cases | 4 cases, 0 crossings, 0 recoverable crossings | Weak tangential shaping did not solve selected non-crossing cases. |
| Phase37B weak regression cases | 8 cases, 4 crossings, 4 recoverable crossings | Weak tangential shaping did not preserve all known recoverable behavior in the regression subset. |

## Protected Known-Success Case

A known Phase34 recoverable case is a case that belongs to the Phase34 `radius_priority` set and is known to produce a recoverable crossing under the protected Phase34 evidence.

The current protected aggregate count is:

- controller: `phase34_post_cross_sync`
- post-cross mode: `radius_priority`
- cases: `24`
- crossings: `8`
- recoverable crossings: `8`

Future experiments must preserve these known Phase34 recoverable cases before claiming broader progress. A future method that creates a new proxy improvement while losing these cases is diagnostic only.

The exact row-level membership should be represented in future Result Schema v1 artifacts using:

- `known_phase34_recoverable_case`
- `regression_set_membership`
- `case_id`

Historical artifacts do not need to be rewritten to add those fields.

## Regression Set

A regression set is a group of cases used to check that a new method preserves known behavior.

At minimum, future experiments should distinguish:

- known Phase34 recoverable cases,
- known baseline non-crossing cases,
- diagnostic selected subsets when relevant.

Regression sets are not only for success cases. The known baseline non-crossing cases protect the interpretation of new crossing-generation claims, while selected diagnostic subsets protect the scope of subset claims.

## Diagnostic Subset

A diagnostic subset is a deliberately selected set of cases used for mechanism analysis, not a full-benchmark claim.

Examples:

- Phase37B selected non-crossing cases,
- Phase37B regression crossing cases,
- future handpicked difficult cases,
- future ablation subsets,
- future held-out mechanism probes.

Subset rows must report `is_full_benchmark=false`, a `subset_id`, a `representative_subset_note`, and regression status. A subset can motivate the next experiment, but it cannot be reported as full-benchmark progress unless the same method is evaluated on the declared benchmark and passes regression preservation.

## Accepted Progress

Accepted progress requires:

- declared benchmark or explicitly scoped subset,
- crossing and recoverability reported separately,
- overspeed and instability reported,
- failure labels reported,
- subset status reported,
- regression status reported,
- known Phase34 recoverable cases preserved,
- artifact path recorded,
- claim scoped to the 2D simulator.

Accepted progress does not mean flight readiness, hardware readiness, docking readiness, real spacecraft validation, or sim-to-real transfer.

## Regression

Regression includes:

- fewer known Phase34 recoverable crossings,
- loss of known crossing cases,
- increased overspeed,
- increased instability,
- invalid simulation,
- unsafe-state increase,
- missing required reporting fields,
- subset improvement with damage to known-success cases,
- proxy improvement without primary metric improvement.

Any regression must be reported directly. A method with a regression may still be useful as diagnostic evidence, but it cannot be described as clean benchmark progress.

## Diagnostic-Only Evidence

Diagnostic-only evidence includes:

- closest approach improvement without crossing,
- crossing potential improvement without crossing,
- subset-only improvement,
- learning loss reduction without rollout improvement,
- reward increase without recoverability improvement,
- new crossing without recoverable crossing,
- results missing labels or safety fields.

Diagnostic-only results should be preserved and labeled. They should not be upgraded into accepted progress by optimistic wording.

## Minimum Reporting Requirements

Future controller, planner, learning, or runtime-assurance experiments must report at least:

- `schema_version`
- `benchmark_id`
- `benchmark_version`
- `experiment_id`
- `controller_id`
- `controller_family`
- `case_id`
- `artifact_path`
- `source_script`
- `r0_over_target`
- `initial_velocity_angle_deg`
- `thrust_scale`
- `crossed_target_radius`
- `first_crossing_step`
- `recoverable_crossing`
- `final_simulator_success`
- `overspeed`
- `instability`
- `unsafe_state`
- `invalid_simulation`
- `terminal_label`
- `precursor_labels`
- `diagnostic_labels`
- `manual_audit_note`
- `label_taxonomy_version`
- `is_full_benchmark`
- `subset_id`
- `representative_subset_note`
- `regression_set_membership`
- `known_phase34_recoverable_case`
- `accepted_as_progress`
- `acceptance_reason`

If current scripts cannot populate a field, the field should be left empty or null under Result Schema v1 rules, and the result should not claim accepted progress until the missing field is resolved.

## Progress Claim Type Table

| Claim type | Minimum evidence required | Allowed claim if evidence is incomplete |
| --- | --- | --- |
| New crossing generation | More target-radius crossings on declared benchmark or scoped subset, known Phase34 recoverable cases preserved, safety controlled, labels valid, subset status explicit | Diagnostic crossing-generation signal only. |
| New recoverable crossing generation | More recoverable crossings, crossing and recoverability separated, known Phase34 recoverable cases preserved, no unacceptable safety regression, benchmark or subset status explicit | Diagnostic recoverability signal only. |
| Safer behavior | Lower overspeed, instability, unsafe-state, or invalid-simulation rate, with no hidden loss of recoverability | Safety diagnostic only. |
| Runtime-assurance / final-veto improvement | Avoided failures, blocked successes, unnecessary veto rate, recoverability preserved, performance cost, comparison against no-monitor baseline | Monitor diagnostic only. |
| Learning baseline improvement | Rollout improvement under same benchmark, explicit-controller baseline comparison, labels, safety metrics, known-success regression status | Training diagnostic only. |

## Claim Rules

### New Crossing Generation

Must show:

- more target-radius crossings on the declared benchmark or clearly scoped subset,
- known Phase34 recoverable cases preserved,
- overspeed and instability controlled,
- terminal labels valid,
- subset status explicit.

If crossings increase but recoverable crossings do not, claim only crossing-generation progress, not recoverability progress.

Closest approach improvement without a new target-radius crossing is diagnostic only.

### New Recoverable Crossing Generation

Must show:

- more recoverable crossings,
- crossing and recoverability reported separately,
- known Phase34 recoverable cases preserved,
- no unacceptable safety regression,
- full benchmark or clearly scoped subset status.

New recoverable crossings should be reported separately from final simulator success. A recoverable crossing can still fail later, and that distinction must be labeled.

### Safer Behavior

Must show:

- lower overspeed, instability, unsafe-state, or invalid-simulation rate,
- no hidden loss of recoverability,
- no conversion of doing nothing into safety success unless safe abort or degraded mission is defined.

A controller that avoids overspeed by never attempting the task has not improved recoverability safety unless the benchmark explicitly defines safe abort behavior and reports the performance cost.

### Runtime-Assurance / Final-Veto Improvement

Must show:

- avoided failures,
- blocked successes,
- unnecessary veto rate,
- recoverability preserved,
- performance cost,
- comparison against no-monitor baseline,
- no claim of formal safety unless formally proven.

Runtime-assurance improvements must report false veto behavior. A monitor that blocks successful recoverable cases is a regression unless the claim is explicitly about a different safety envelope and the tradeoff is accepted.

### Learning Baseline Improvement

Must show:

- rollout improvement under the same benchmark,
- explicit-controller baseline comparison,
- failure labels,
- safety metrics,
- known-success regression status,
- not only lower imitation loss, reward increase, or training curve improvement.

Learning curves, validation loss, reward, and policy confidence are diagnostics. They are not benchmark progress without rollout evidence.

## Regression Trigger Table

| Trigger | Why it is a regression | Required treatment |
| --- | --- | --- |
| Fewer known Phase34 recoverable crossings | Destroys protected recoverability behavior | Mark diagnostic only; explain loss. |
| Loss of known crossing cases | Damages protected event behavior | Mark diagnostic only unless explicitly studying failure. |
| Increased `overspeed` | Safety worsened | Block clean accepted progress. |
| Increased `instability` | Closed-loop behavior worsened | Block clean accepted progress. |
| Increased `unsafe_state` | Safety envelope worsened | Block clean accepted progress. |
| `invalid_simulation=true` | Output cannot be trusted | Block accepted progress. |
| Missing `terminal_label` | Failure mechanism not auditable | Block accepted progress. |
| Missing safety fields | Safety tradeoff hidden | Block accepted progress. |
| Missing subset status | Scope is unclear | Block accepted progress. |
| Subset gain with known-success damage | Local improvement destroys baseline behavior | Mark diagnostic only. |
| Closest approach gain without crossing | Proxy-only improvement | Mark diagnostic only. |
| More crossings without more recoverable crossings | Event-only improvement | Allow crossing-only claim, not recoverability claim. |

## Diagnostic-Only Result Table

| Result pattern | Diagnostic interpretation | Not allowed claim |
| --- | --- | --- |
| Closest approach improves but `crossed_target_radius=false` | Search-space or geometry signal | Crossing or recoverability progress. |
| `best_crossing_potential` improves but no crossing occurs | Planner hypothesis signal | Crossing progress. |
| Subset improves but full benchmark is not run | Hypothesis-generation evidence | Full-benchmark progress. |
| New crossing with `recoverable_crossing=false` | Crossing-generation signal | Recoverability progress. |
| Reward increases without rollout metric improvement | Learning diagnostic | Benchmark progress. |
| Imitation loss decreases without rollout metric improvement | Training diagnostic | Controller improvement. |
| Safety improves while known recoverable cases are lost | Safety tradeoff | Clean recoverability progress. |
| Labels or safety fields are missing | Incomplete evidence | Accepted progress. |

## Handling Specific Evidence Types

### Closest Approach Improvements

Closest approach is diagnostic. It can justify more search or planner work, but it is not target-radius crossing, recoverability, or final simulator success.

Use closest approach improvements to explain mechanism, not to claim progress.

### Crossing-Only Improvements

Crossing-only improvement is allowed as a scoped claim when:

- target-radius crossings increase,
- recoverable crossings are reported separately,
- known Phase34 recoverable cases are preserved,
- safety fields are reported,
- labels are valid,
- subset status is explicit.

Crossing-only improvement must not be called recoverability progress.

### Subset Improvements

Subset improvement is diagnostic unless:

- the subset is explicitly declared,
- regression cases are preserved,
- the result is later evaluated on the declared benchmark,
- the full-benchmark result passes the required reporting and preservation checks.

Phase37B is the current warning example: weak tangential shaping did not create selected-case crossings and did not preserve all regression crossings.

### Safety Tradeoffs

Safety tradeoffs must be reported explicitly.

An increase in `overspeed`, `instability`, `unsafe_state`, or `invalid_simulation` blocks clean accepted progress. If a future method improves recoverability while worsening safety, the result can be reported only as a tradeoff or diagnostic unless the benchmark has a predeclared acceptance rule for that tradeoff.

### Missing Fields

Missing fields block accepted progress when the missing field is required to evaluate the claim.

Examples:

- missing `overspeed` blocks safety-clean progress claims,
- missing `instability` blocks safety-clean progress claims,
- missing `terminal_label` blocks auditable failure interpretation,
- missing `is_full_benchmark` blocks full-benchmark claims,
- missing `regression_set_membership` blocks known-success preservation claims,
- missing `known_phase34_recoverable_case` blocks upstream progress claims.

Use `unknown` and `manual_audit_note` when the available artifact cannot support a more precise label.

## Suggested First Regression Gate Design

The first gate should be lightweight and documentation-driven. It should not become a full experiment manager.

Suggested future script name:

```text
scripts/check_recoverability_regression_gate.py
```

The first implementation should be a schema and consistency validator for new artifacts, not a generator of old artifacts.

Suggested gate stages:

1. Run `python scripts/check_phase_results.py`.
2. Read one declared future result artifact.
3. Verify required Result Schema v1 columns.
4. Validate controlled `terminal_label` values.
5. Run logical consistency checks.
6. Check accepted-progress rows for safety and subset violations.
7. Check known Phase34 recoverable case preservation for upstream progress claims.
8. Print explicit pass/fail reasons.

## Suggested Gate Checks Table

| Gate check | Required behavior |
| --- | --- |
| Historical guard | Run `python scripts/check_phase_results.py` and fail if it fails. |
| Required columns | Require `schema_version`, `benchmark_id`, `benchmark_version`, `controller_id`, `case_id`, `crossed_target_radius`, `recoverable_crossing`, `overspeed`, `instability`, `terminal_label`, `is_full_benchmark`, `regression_set_membership`, `known_phase34_recoverable_case`, `accepted_as_progress`, and `acceptance_reason`. |
| Controlled labels | `terminal_label` must come from Failure Label Taxonomy v0. |
| Crossing consistency | `crossed_target_radius=false` must not have `recoverable_crossing=true`. |
| Crossing step consistency | `crossed_target_radius=false` must not have `first_crossing_step` populated. |
| Overspeed veto | `overspeed=true` should block clean `accepted_as_progress=true`. |
| Instability veto | `instability=true` should block clean `accepted_as_progress=true`. |
| Invalid simulation veto | `invalid_simulation=true` should block clean `accepted_as_progress=true`. |
| Subset scope | Subset rows should not set `accepted_as_progress=true` for full-benchmark claims. |
| Unknown label audit | `terminal_label=unknown` should require non-empty `manual_audit_note`. |
| Known-success preservation | Rows marked `known_phase34_recoverable_case=true` should preserve crossing and recoverability before upstream progress can be claimed. |

## Suggested Future Validation Script Behavior

The future script should:

- accept a path to a new result CSV or JSON artifact,
- leave protected historical artifacts untouched,
- run the protected historical guard first,
- validate schema presence and controlled labels,
- reject inconsistent boolean combinations,
- reject clean progress claims with safety vetoes,
- reject full-benchmark claims from subset-only rows,
- summarize known Phase34 recoverable preservation,
- summarize crossing count and recoverable-crossing count separately,
- report diagnostic-only reasons without deleting or rewriting evidence.

It should not:

- rerun old Phase34/36/37 experiments,
- rewrite historical CSVs,
- change controller code,
- infer missing fields silently,
- decide scientific publication claims automatically.

## Week 5 Handoff Questions For Decision Evidence Logging

Week 5 should answer:

- What decision log format should record why a result was accepted, rejected, or kept diagnostic?
- Should decision evidence live beside result artifacts or in a central benchmark log?
- What fields should be required for a decision record: claim type, artifact path, regression status, reviewer, date, and rationale?
- How should manual-audit notes be carried from Result Schema v1 rows into decision evidence?
- Should each accepted progress claim point to the exact gate output that passed?
- How should rejected claims preserve useful diagnostic evidence without encouraging false progress?
- Should runtime-assurance veto decisions have their own event log for blocked action, blocked success, and avoided failure?
- How should learning baseline decisions record training metrics separately from rollout metrics?
- Should the first decision log be plain Markdown, CSV, or JSONL?
- How should decision logs reference protected historical facts without duplicating or rewriting them?

## Week 4 Completion Rule

Week 4 is complete when this document exists, the protected regression guard still passes, and no historical evidence has been modified.
