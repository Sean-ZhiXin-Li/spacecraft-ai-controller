# Result Schema v1

Status: Week 3 draft.

Date: July 15-21, 2026.

Scope: future 2D spacecraft recoverability experiments after Recoverability Benchmark v1 and Failure Label Taxonomy v0.

This document defines a repeatable result schema for future spacecraft recoverability experiments. It is a documentation and schema-design artifact only. It does not rewrite historical CSVs, change controller logic, regenerate old artifacts, or implement a metrics engine.

The schema principle is:

```text
Report event, recoverability, simulator outcome, safety, effort, labels, and benchmark status as separate fields.
```

## Purpose

`result_schema_v1` exists to make future experiment rows auditable and comparable.

It separates:

- event metrics,
- recoverability metrics,
- final simulator outcome,
- safety metrics,
- resource and effort metrics,
- failure labels,
- diagnostic, subset, and regression metadata.

The schema should prevent future results from treating a target-radius crossing, closest approach, legacy simulator success, or subset improvement as recoverability progress by itself.

## Relationship To Recoverability Benchmark v1

Recoverability Benchmark v1 defines the benchmark claim structure:

- crossing is not insertion,
- intermediate success is not recoverable task completion,
- closest approach is diagnostic,
- subset results are not full-benchmark wins,
- known Phase34 recoverable cases must be preserved before claiming progress.

`result_schema_v1` is the row-level reporting contract for future outputs that claim to run against that benchmark.

The schema does not change the protected Phase34/36/37 evidence. Historical CSVs remain evidence sources with legacy column names.

## Relationship To Failure Label Taxonomy v0

Failure Label Taxonomy v0 defines the controlled terminal labels and the difference between terminal labels, precursor labels, diagnostic labels, and manual-audit notes.

`result_schema_v1` imports those fields:

- `terminal_label`
- `precursor_labels`
- `diagnostic_labels`
- `manual_audit_note`
- `label_taxonomy_version`

The `terminal_label` field must use one of the controlled labels from `failure_label_taxonomy_v0`:

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

`precursor_labels` and `diagnostic_labels` must not override `terminal_label`.

## Schema Design Principles

- One row should describe one rollout for one controller, one benchmark case, and one experiment configuration.
- Primary metrics should be directly populated by the script or declared empty; do not infer missing values after the fact.
- Event metrics, recoverability metrics, and simulator success must remain separate.
- `crossed_target_radius=True` does not imply `recoverable_crossing=True`.
- `recoverable_crossing=True` does not automatically imply `final_simulator_success=True` unless the benchmark criteria say so.
- `final_simulator_success=True` remains simulator-defined.
- Closest approach is diagnostic, not success.
- Safety fields must be reported explicitly.
- Failure labels must use the controlled taxonomy.
- Subset status and regression status must be explicit.
- Historical CSVs must not be rewritten into this schema.
- Legacy fields may be mapped only through explicit compatibility rules.
- Missing optional fields should be empty in CSV and `null` in JSON, not guessed.

## Required Fields

Future rows that claim to run under `result_schema_v1` should include the following fields.

### Identity / Provenance

- `schema_version`
- `benchmark_id`
- `benchmark_version`
- `experiment_id`
- `controller_id`
- `controller_family`
- `case_id`
- `artifact_path`
- `source_script`

### Benchmark Case Parameters

- `r0_over_target`
- `initial_velocity_angle_deg`
- `thrust_scale`

### Event Metrics

- `crossed_target_radius`
- `first_crossing_step`
- `crossing_time`
- `state_at_crossing_summary`

### Recoverability Metrics

- `recoverable_crossing`
- `recovery_time`

### Final Outcome

- `final_simulator_success`

### Safety Metrics

- `overspeed`
- `instability`
- `unsafe_state`
- `invalid_simulation`
- `max_speed`

### Resource / Effort Metrics

- `control_effort`
- `fuel_proxy`

### Failure / Diagnostic Labels

- `terminal_label`
- `precursor_labels`
- `diagnostic_labels`
- `manual_audit_note`
- `label_taxonomy_version`

### Subset / Regression Metadata

- `is_full_benchmark`
- `subset_id`
- `representative_subset_note`
- `regression_set_membership`
- `known_phase34_recoverable_case`
- `accepted_as_progress`
- `acceptance_reason`

## Optional Fields

Optional fields should be included when scripts can populate them directly and consistently.

### Identity / Provenance

- `git_commit`
- `environment_summary`

### Benchmark Case Parameters

- `seed`

### Event Metrics

- `phase34_compatible_crossing`
- `controller_mode_at_crossing`

### Recoverability Metrics

- `recovery_cost`
- `minimum_recovery_margin`
- `post_cross_mode`

### Final Outcome

- `legacy_success_label`
- `capture_label`
- `lock_label`

### Resource / Effort Metrics

- `peak_action`
- `saturation_count`

### Diagnostic Metrics

- `closest_approach`
- `closest_approach_step`
- `crossing_potential`

These diagnostic metrics can help explain mechanism, but they must not be used as success criteria by themselves.

## Deprecated / Legacy Compatibility Fields

Historical CSVs use several names that overlap with this schema but are not normalized `result_schema_v1` fields.

| Legacy field | Seen in current artifacts | Compatibility rule |
| --- | --- | --- |
| `crossing_occurs` | Phase34, Phase36B, Phase37A, Phase37B | May map to `crossed_target_radius` when the script's crossing detector matches the benchmark detector. |
| `crossing_step` | Phase34 | May map to `first_crossing_step`. |
| `first_crossing_step` | Phase36B, Phase37A | Same normalized name as this schema. |
| `success` | Phase34 | May map only to `legacy_success_label`; do not treat as recoverability success by itself. |
| `simulator_success_label` | Phase36B, Phase37A, Phase37B | May map to `final_simulator_success` if the experiment declares the same simulator criterion. |
| `capture_entered` | Phase34 | May map to `capture_label`. |
| `lock_entered` | Phase34 | May map to `lock_label`. |
| `recoverable_state` | Phase34, Phase36B, Phase37A | Diagnostic compatibility field; `recoverable_crossing` remains the normalized benchmark field. |
| `max_speed_ratio` | Phase34, Phase36B, Phase37A | May map to `max_speed` only if documented as a ratio. |
| `dominant_failure_label` | Phase36B, Phase37A | Legacy diagnostic label; may map to `diagnostic_labels`, not directly to `terminal_label`. |
| `failure_label` | Phase37B | Legacy diagnostic label; may map to `diagnostic_labels`, not directly to `terminal_label`. |
| `termination_reason` | Phase34, Phase36B, Phase37A | Compatibility evidence for timeout or simulator outcome; not a controlled terminal label. |
| `min_abs_radius_error_ratio` | Phase36B, Phase37A, Phase37B | May map to `closest_approach`; remains diagnostic. |
| `best_crossing_potential` | Phase36B, Phase37A | May map to `crossing_potential`; remains diagnostic. |

Compatibility mapping must be documented by the script that writes new rows. Do not silently rename historical columns in place.

## Field Type Table

| Field | Type | Required | Empty value | Definition |
| --- | --- | --- | --- | --- |
| `schema_version` | string | yes | never empty | Result schema identifier, normally `result_schema_v1`. |
| `benchmark_id` | string | yes | never empty | Stable benchmark family, for example `recoverability_benchmark`. |
| `benchmark_version` | string | yes | never empty | Benchmark version, for example `v1`. |
| `experiment_id` | string | yes | never empty | Unique experiment or run identifier. |
| `controller_id` | string | yes | never empty | Specific controller or variant identifier. |
| `controller_family` | string | yes | never empty | Coarser family, such as `phase34_post_cross_sync` or `transfer_family`. |
| `case_id` | string | yes | never empty | Stable case identifier. |
| `artifact_path` | string | yes | never empty | Path to the artifact containing or supporting the row. |
| `source_script` | string | yes | never empty | Script that generated the row. |
| `git_commit` | string | no | empty/null | Git commit used for the run, when captured. |
| `environment_summary` | string | no | empty/null | Short runtime environment summary. |
| `r0_over_target` | number | yes | never empty | Initial radius divided by target radius. |
| `initial_velocity_angle_deg` | number | yes | never empty | Initial velocity angle in degrees. |
| `thrust_scale` | number | yes | never empty | Thrust scale used for the case. |
| `seed` | integer/string | no | empty/null | Random seed when the rollout is stochastic. |
| `crossed_target_radius` | boolean | yes | never empty | Whether the target-radius crossing event occurred. |
| `first_crossing_step` | integer | yes | empty/null allowed when no crossing | First step at which crossing occurred. |
| `crossing_time` | number | yes | empty/null allowed when no crossing | First crossing time in simulator time units. |
| `state_at_crossing_summary` | string/object | yes | empty/null allowed when no crossing | Stable summary of state at first crossing. |
| `phase34_compatible_crossing` | boolean | no | empty/null | Whether the crossing is compatible with the Phase34-style post-cross controller. |
| `controller_mode_at_crossing` | string | no | empty/null | Controller mode active at first crossing. |
| `recoverable_crossing` | boolean | yes | never empty | Whether a target-radius crossing entered the declared recoverable condition. |
| `recovery_time` | number | yes | empty/null allowed when not recoverable | Time from crossing to recoverable condition or benchmark-defined recovery. |
| `recovery_cost` | number | no | empty/null | Cost accumulated during recovery window. |
| `minimum_recovery_margin` | number | no | empty/null | Smallest margin to recoverability or safety boundary. |
| `post_cross_mode` | string | no | empty/null | Post-cross mode, when applicable. |
| `final_simulator_success` | boolean | yes | never empty | Simulator-defined final success under declared criteria. |
| `legacy_success_label` | boolean/string | no | empty/null | Historical script success value preserved for compatibility. |
| `capture_label` | boolean/string | no | empty/null | Simulator or legacy capture state. |
| `lock_label` | boolean/string | no | empty/null | Simulator or legacy lock state. |
| `overspeed` | boolean | yes | never empty | Whether speed threshold was violated. |
| `instability` | boolean | yes | never empty | Whether instability criterion was violated. |
| `unsafe_state` | boolean | yes | never empty | Whether a non-overspeed unsafe state was entered. |
| `invalid_simulation` | boolean | yes | never empty | Whether simulator output was invalid. |
| `max_speed` | number | yes | empty/null only if invalid simulation prevents measurement | Maximum speed or documented speed ratio. |
| `control_effort` | number | yes | empty/null if not yet measured | Integrated or aggregate control effort under documented definition. |
| `fuel_proxy` | number | yes | empty/null if not yet measured | Fuel or impulse proxy under documented definition. |
| `peak_action` | number | no | empty/null | Peak action magnitude. |
| `saturation_count` | integer | no | empty/null | Count of action saturation events. |
| `terminal_label` | enum string | yes | never empty | Controlled terminal label from Failure Label Taxonomy v0. |
| `precursor_labels` | list/string | yes | empty list/string allowed | Events that happened before termination. |
| `diagnostic_labels` | list/string | yes | empty list/string allowed | Diagnostic mechanism labels. |
| `manual_audit_note` | string | yes | empty allowed | Free-text audit note. |
| `label_taxonomy_version` | string | yes | never empty | Normally `failure_label_taxonomy_v0`. |
| `is_full_benchmark` | boolean | yes | never empty | Whether the row belongs to a full benchmark run. |
| `subset_id` | string | yes | empty allowed for full benchmark rows | Identifier for diagnostic or representative subset. |
| `representative_subset_note` | string | yes | empty allowed | Scope note for subset rows. |
| `regression_set_membership` | string/list | yes | empty allowed | Regression group membership, if any. |
| `known_phase34_recoverable_case` | boolean | yes | never empty | Whether this case is in the known Phase34 recoverable set. |
| `accepted_as_progress` | boolean | yes | never empty | Whether this row contributes to an accepted progress claim. |
| `acceptance_reason` | string | yes | empty allowed only when `accepted_as_progress=False` and reason is obvious from labels | Reason for acceptance or rejection. |
| `closest_approach` | number | no | empty/null | Diagnostic closest approach metric. |
| `closest_approach_step` | integer | no | empty/null | Step at closest approach. |
| `crossing_potential` | number | no | empty/null | Diagnostic crossing-potential metric. |

## CSV Representation

CSV output should use one header row with stable field names.

Rules:

- Booleans should be `true` or `false` in lowercase.
- Missing optional values should be empty strings.
- Missing required conditional values, such as `first_crossing_step` for a no-crossing row, should be empty strings.
- List fields should use semicolon-separated values with no surrounding brackets.
- Numeric fields should use plain decimal notation where practical.
- JSON-like nested state should be avoided in CSV unless it is quoted and documented.

Minimal CSV header:

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
```

CSV writers may append optional fields after the required block.

## JSON Representation

JSON output should use one object per rollout or a top-level array of rollout objects.

Rules:

- Booleans should be native JSON booleans.
- Missing optional values should be `null`.
- Missing conditional required values should be `null`.
- List fields should be arrays.
- `state_at_crossing_summary` may be a short object when the writer can keep keys stable.

Minimal JSON shape:

```json
{
  "schema_version": "result_schema_v1",
  "benchmark_id": "recoverability_benchmark",
  "benchmark_version": "v1",
  "experiment_id": "example_experiment",
  "controller_id": "example_controller",
  "controller_family": "example_family",
  "case_id": "case_r1p00_a150_t8000",
  "artifact_path": "analysis/example/result.csv",
  "source_script": "scripts/example.py",
  "r0_over_target": 1.0,
  "initial_velocity_angle_deg": 150.0,
  "thrust_scale": 8000.0,
  "crossed_target_radius": true,
  "first_crossing_step": 1716,
  "crossing_time": null,
  "state_at_crossing_summary": "vr_ratio=-0.0137;vt_error_ratio=-0.7502",
  "recoverable_crossing": true,
  "recovery_time": null,
  "final_simulator_success": true,
  "overspeed": false,
  "instability": false,
  "unsafe_state": false,
  "invalid_simulation": false,
  "max_speed": 1.0,
  "control_effort": null,
  "fuel_proxy": null,
  "terminal_label": "success",
  "precursor_labels": ["crossed_target_radius", "entered_recoverable_basin"],
  "diagnostic_labels": [],
  "manual_audit_note": "",
  "label_taxonomy_version": "failure_label_taxonomy_v0",
  "is_full_benchmark": true,
  "subset_id": "",
  "representative_subset_note": "",
  "regression_set_membership": "known_phase34_recoverable",
  "known_phase34_recoverable_case": true,
  "accepted_as_progress": false,
  "acceptance_reason": "baseline preservation row, not a new progress claim"
}
```

## Minimal Example Rows

These examples are schema examples for future outputs. They are not historical CSV rewrites.

### No Crossing

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
result_schema_v1,recoverability_benchmark,v1,example_full_run,baseline_phase34,phase34_family,case_r0p98_a150_t8000,analysis/example/result.csv,scripts/example.py,0.98,150.0,8000.0,false,,,,false,,false,false,false,false,false,1.2998,,,no_crossing,,near_crossing,closest approach remains diagnostic only,failure_label_taxonomy_v0,true,,,baseline_non_crossing,false,false,no target-radius crossing
```

### Crossing But Unrecoverable

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
result_schema_v1,recoverability_benchmark,v1,example_full_run,phase31_reference,baseline_reference,case_r1p00_a150_t8000,analysis/example/result.csv,scripts/example.py,1.0,150.0,8000.0,true,631,,vr_ratio=-0.0060;vt_error_ratio=-0.7801,false,,false,false,false,false,false,1.0000,,,crossing_unrecoverable,crossed_target_radius,,crossed target radius but never entered recoverable basin,failure_label_taxonomy_v0,true,,,known_crossing_regression,true,false,crossing without recoverability is not accepted progress
```

### Recoverable Crossing / Success

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
result_schema_v1,recoverability_benchmark,v1,example_full_run,phase34_radius_priority,phase34_post_cross_sync,case_r1p00_a150_t8000,analysis/example/result.csv,scripts/example.py,1.0,150.0,8000.0,true,1716,,vr_ratio=-0.0137;vt_error_ratio=-0.7502,true,,true,false,false,false,false,1.0000,,,success,crossed_target_radius;entered_recoverable_basin,,clean known recoverable case under declared simulator criteria,failure_label_taxonomy_v0,true,,,known_phase34_recoverable,true,false,baseline preservation row not a new progress claim
```

### Overspeed

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
result_schema_v1,recoverability_benchmark,v1,example_safety_run,aggressive_transfer,experimental_transfer,case_r1p00_a165_t10000,analysis/example/result.csv,scripts/example.py,1.0,165.0,10000.0,true,900,,vr_ratio=-0.0200;vt_error_ratio=-0.6000,false,,false,true,false,false,false,1.8000,,,overspeed,crossed_target_radius,,overspeed outranks crossing and recoverability evidence,failure_label_taxonomy_v0,true,,,known_phase34_recoverable,true,false,safety violation blocks accepted progress
```

### Diagnostic Subset Row

```csv
schema_version,benchmark_id,benchmark_version,experiment_id,controller_id,controller_family,case_id,artifact_path,source_script,r0_over_target,initial_velocity_angle_deg,thrust_scale,crossed_target_radius,first_crossing_step,crossing_time,state_at_crossing_summary,recoverable_crossing,recovery_time,final_simulator_success,overspeed,instability,unsafe_state,invalid_simulation,max_speed,control_effort,fuel_proxy,terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version,is_full_benchmark,subset_id,representative_subset_note,regression_set_membership,known_phase34_recoverable_case,accepted_as_progress,acceptance_reason
result_schema_v1,recoverability_benchmark,v1,example_subset_run,weak_tangential,phase37b_like_subset,selected_01,analysis/example/subset.csv,scripts/example_subset.py,1.02,150.0,10000.0,false,,,,false,,false,false,false,false,false,1.0002,,,no_crossing,,near_crossing;selected_subset,diagnostic subset row cannot establish full-benchmark progress,failure_label_taxonomy_v0,false,weak_tangential_selected,selected non-crossing diagnostic subset,selected_non_crossing,false,false,subset diagnostic only
```

## Field Definitions

### Identity / Provenance

`schema_version` identifies the schema used by the row. For this document, use `result_schema_v1`.

`benchmark_id` identifies the benchmark family. Use a stable value such as `recoverability_benchmark`.

`benchmark_version` identifies the benchmark contract version. Use `v1` for Recoverability Benchmark v1.

`experiment_id` identifies the run or experiment. It should be stable enough to group rows from the same run.

`controller_id` identifies the exact controller or variant.

`controller_family` identifies a broader family for aggregation.

`case_id` identifies the benchmark case. It should not depend on row order.

`artifact_path` points to the result artifact or supporting artifact.

`source_script` names the script that produced the row.

`git_commit` and `environment_summary` are optional provenance fields. Record them when available from the run process.

### Benchmark Case Parameters

`r0_over_target`, `initial_velocity_angle_deg`, and `thrust_scale` define the current reduced-grid case parameters used by the protected benchmark evidence.

`seed` is optional and should be populated for stochastic controllers, randomized initial conditions, or sampled disturbances.

### Event Metrics

`crossed_target_radius` records whether the target-radius crossing event occurred.

`first_crossing_step` records the first crossing step. It should be empty/null when `crossed_target_radius=false`.

`crossing_time` records first crossing time if the simulator exposes time. It should be empty/null when unavailable or when no crossing occurred.

`state_at_crossing_summary` records a stable summary of the state at first crossing. It should be empty/null when no crossing occurred. A compact string is acceptable for CSV; JSON may use an object.

`phase34_compatible_crossing` is optional and records whether the crossing can be handed to the Phase34-style terminal controller under the experiment's compatibility rule.

`controller_mode_at_crossing` is optional and records the active controller mode when crossing occurred.

### Recoverability Metrics

`recoverable_crossing` records whether a target-radius crossing reached the declared recoverability condition.

`recovery_time` records time from crossing to recoverability, when available. It should be empty/null when no crossing occurred or when the crossing was not recoverable.

`recovery_cost`, `minimum_recovery_margin`, and `post_cross_mode` are optional fields for richer recovery analysis.

### Final Outcome

`final_simulator_success` records the simulator-defined final outcome under the experiment's declared success criterion. It is not mission success.

`legacy_success_label`, `capture_label`, and `lock_label` preserve historical or simulator-specific labels without collapsing them into recoverability.

### Safety Metrics

`overspeed` records whether the speed threshold was violated.

`instability` records whether the instability criterion was violated.

`unsafe_state` records safety-envelope violations not already covered by overspeed or instability.

`invalid_simulation` records structurally or numerically invalid simulator output.

`max_speed` records maximum speed or a documented speed ratio.

### Resource / Effort Metrics

`control_effort` records an aggregate control-effort metric under a declared definition. If a current script cannot compute it, leave it empty/null.

`fuel_proxy` records a fuel, impulse, or resource proxy under a declared definition. If a current script cannot compute it, leave it empty/null.

`peak_action` and `saturation_count` are optional effort diagnostics.

### Failure / Diagnostic Labels

`terminal_label` is the single controlled label assigned after applying Failure Label Taxonomy v0.

`precursor_labels` records events that happened before termination.

`diagnostic_labels` records analysis labels such as `near_crossing`, `over_conservative_transfer`, or subset-specific diagnostic tags.

`manual_audit_note` records free-text caveats.

`label_taxonomy_version` should be `failure_label_taxonomy_v0` for this schema version.

### Subset / Regression Metadata

`is_full_benchmark` states whether the row belongs to a full benchmark run.

`subset_id` identifies diagnostic subsets. It should be empty for full benchmark rows.

`representative_subset_note` explains subset selection.

`regression_set_membership` records whether the case belongs to a regression group such as known Phase34 recoverable cases, selected non-crossing cases, or baseline non-crossing cases.

`known_phase34_recoverable_case` records whether the case is one of the known Phase34 recoverable cases that future progress claims must preserve.

`accepted_as_progress` records whether this row contributes to an accepted progress claim. Most baseline preservation rows and diagnostic subset rows should be `false`.

`acceptance_reason` explains why the row does or does not count toward progress.

## Rules For Missing Fields

- Required identity and benchmark parameter fields must not be missing.
- Required booleans must be populated as `true` or `false`; do not leave them blank.
- Conditional event fields may be empty/null when their precondition is false. For example, `first_crossing_step` is empty/null when `crossed_target_radius=false`.
- Optional fields should be empty/null when unavailable.
- Do not estimate `control_effort`, `fuel_proxy`, `recovery_cost`, or `minimum_recovery_margin` unless the writer has a documented computation.
- Do not infer `unsafe_state=false` from absence of an unsafe-state field in a legacy artifact. Future `result_schema_v1` rows must populate it explicitly.
- Do not infer `invalid_simulation=false` from a successful CSV write alone. Future writers should explicitly validate or set the field under documented rules.
- If a field required for a precise terminal label is missing, use `unknown` or a more conservative supported label and explain the gap in `manual_audit_note`.

## Rules For Accepted Progress

A result claiming accepted progress must report:

- `crossed_target_radius`
- `recoverable_crossing`
- `overspeed`
- `instability`
- `terminal_label`
- `is_full_benchmark`
- `regression_set_membership`
- `known_phase34_recoverable_case`
- `accepted_as_progress`
- `acceptance_reason`

Acceptance rules:

- New crossing generation and new recoverable crossing generation are separate claims.
- A row with `crossed_target_radius=true` and `recoverable_crossing=false` may support crossing-only analysis, but not recoverability progress.
- A row with `recoverable_crossing=true` but `final_simulator_success=false` must be labeled according to the failure taxonomy and explained.
- Any `overspeed=true`, `instability=true`, `unsafe_state=true`, or `invalid_simulation=true` blocks clean accepted progress unless the claim is explicitly about diagnosing that failure.
- A full-benchmark claim must report full benchmark status and regression preservation.
- Known Phase34 recoverable cases must be preserved before claiming upstream crossing-generation progress.
- A result that improves closest approach, crossing potential, reward, or loss without new accepted crossing or recoverability metrics remains diagnostic.
- `accepted_as_progress=true` should require a non-empty `acceptance_reason`.

## Rules For Diagnostic-Only Results

Diagnostic-only rows are valid and useful, but they must not be reported as full-benchmark progress.

Use diagnostic-only status when:

- `is_full_benchmark=false`,
- a selected subset was run,
- closest approach improved without a new crossing,
- crossing potential improved without a new crossing,
- known Phase34 recoverable cases regressed,
- safety metrics worsened,
- required progress fields are missing,
- labels are `unknown`,
- the result is hypothesis-generation evidence.

For diagnostic-only rows:

- set `accepted_as_progress=false`,
- populate `subset_id` when the row is from a subset,
- explain the scope in `representative_subset_note`,
- preserve `terminal_label` and diagnostic labels separately,
- do not use `success` unless the row actually satisfies declared simulator and benchmark criteria.

## Compatibility With Current Protected Evidence

Current scripts can populate many but not all fields:

- Phase34 has event, post-cross, legacy success, capture/lock, speed, overspeed, and some effort-like control norm fields.
- Phase36B and Phase37A have crossing, recoverability, diagnostic labels, overspeed, instability, max speed ratio, and simulator success labels.
- Phase37B has subset groups, crossing, recoverability, simulator success, overspeed, instability, and legacy `failure_label`.

Current scripts do not consistently provide:

- `schema_version`
- `benchmark_id`
- `benchmark_version`
- `experiment_id`
- stable `case_id` across all phases
- `artifact_path`
- `source_script`
- `git_commit`
- `environment_summary`
- `crossing_time`
- normalized `state_at_crossing_summary`
- `recovery_time`
- `recovery_cost`
- `minimum_recovery_margin`
- normalized `unsafe_state`
- normalized `invalid_simulation`
- consistent `control_effort`
- consistent `fuel_proxy`
- `terminal_label`
- `precursor_labels`
- `label_taxonomy_version`
- normalized full-benchmark/subset/regression progress fields

Do not backfill these fields into historical artifacts unless a future migration task explicitly creates a new derived artifact.

## Week 4 Handoff Questions For Regression Gate

Week 4 should answer:

- Which `result_schema_v1` fields should the first regression gate require for every new artifact?
- Should the gate validate all controlled `terminal_label` values against Failure Label Taxonomy v0?
- Should the gate fail when `accepted_as_progress=true` appears on subset rows?
- Should the gate fail when `accepted_as_progress=true` and regression-set rows are missing?
- How should the gate verify that known Phase34 recoverable cases are preserved?
- Should the gate require `overspeed`, `instability`, `unsafe_state`, and `invalid_simulation` to be explicit booleans?
- Should the gate reject rows where `crossed_target_radius=false` but `first_crossing_step` is populated?
- Should the gate reject rows where `crossed_target_radius=false` and `recoverable_crossing=true`?
- Should the gate require non-empty `manual_audit_note` when `terminal_label=unknown`?
- Should the first gate be documentation-only, schema-lint-only, or connected to a new example artifact directory?

## Week 3 Completion Rule

Week 3 is complete when this document exists, the protected regression guard still passes, and no historical evidence has been modified.
