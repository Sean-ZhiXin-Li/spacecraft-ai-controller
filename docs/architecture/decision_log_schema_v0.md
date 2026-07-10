# Decision Log Schema v0

## Status

Week 5 milestone document.

Completed: 2026-07-09

Scope: minimal decision-evidence logging schema for future recoverability experiments.

This document defines a small decision-log schema that makes future decisions observable before building decision automation. It is not a full Decision Manager implementation, not a runtime autonomy executive, not a final-veto implementation, and not a formal-safety claim.

The schema question is:

```text
What information would justify continue, retry, retreat, re-observe, safe mode, abort, controller switch, or veto?
```

The reconstruction goal is:

```text
A future reader should be able to reconstruct why the system or evaluator continued, retried, retreated, re-observed, entered safe mode, aborted, selected a controller, vetoed an action or mode, accepted a result, rejected a result, or kept a result diagnostic.
```

## Purpose

`decision_log_schema_v0` exists to record decision evidence separately from result metrics.

The schema should make visible:

- why a runtime decision was made,
- what evidence supported the decision,
- what safety and recoverability levels were active,
- whether trust was degraded,
- whether a fallback existed,
- whether a veto was evaluated,
- why a research result was accepted, rejected, or kept diagnostic.

Decision logs are required because aggregate metrics can hide failed retries, vetoes, aborts, low-trust observations, and manual audit decisions.

## Relationship To Existing Documents

### Recoverability Benchmark v1

Recoverability Benchmark v1 establishes that crossing is not insertion and that intermediate success is not recoverable task completion.

Decision logs carry that rule into decision evidence. A target-radius crossing should trigger a decision record explaining whether the system continued, stabilized, retried, retreated, re-observed, or aborted.

### Failure Label Taxonomy v0

Failure Label Taxonomy v0 defines what happened through controlled labels such as `no_crossing`, `crossing_unrecoverable`, `overspeed`, and `unknown`.

Decision logs explain how the system or evaluator responded. A failure label is evidence for a decision, not the decision itself.

### Result Schema v1

Result Schema v1 defines rollout result rows: crossing, recoverability, final simulator outcome, safety, labels, and benchmark status.

Decision logs link to result rows through:

- `result_schema_version`
- `result_row_ref`
- `terminal_label`
- `accepted_as_progress`
- `acceptance_reason`

Decision logs should not replace result rows.

### Recoverability Regression Policy v0

Recoverability Regression Policy v0 defines when future results can be accepted, rejected, or marked diagnostic.

Decision logs provide the evidence trail for evaluator decisions such as `accept_progress`, `reject_progress`, and `mark_diagnostic`.

### Decision And Runtime Assurance Architecture

`docs/architecture/decision_and_runtime_assurance.md` defines the broader architecture: Decision Manager, Runtime Assurance / Final Veto, trust, recoverability, fallback, and logging.

This schema is the minimal logging layer for that architecture. It records decisions but does not implement the Decision Manager, Runtime Assurance, final veto, trust manager, or controller switching logic.

## What A Decision Log Is

A decision log is one structured record for one decision event.

It answers:

- What decision was made?
- Who or what had authority for the decision?
- What event or evidence triggered it?
- What safety level and recoverability level were active?
- What trust flags were present?
- Was fallback available?
- Was a veto evaluated?
- Why was the result accepted, rejected, or marked diagnostic?

Decision logs may describe runtime-intent decisions or evaluator/research decisions.

## What A Decision Log Is Not

A decision log is not:

- a replacement for a result row,
- a replacement for a failure label,
- a controller implementation,
- a final-veto implementation,
- a formal proof of safety,
- a hardware, ROS, sensor, or robotics interface,
- a mechanism for rewriting historical Phase34/36/37 artifacts,
- a way to turn diagnostic evidence into accepted progress.

Historical Phase34/36/37 artifacts should not be rewritten to add decision logs.

## Concept Boundaries

| Concept | Meaning | Example field |
| --- | --- | --- |
| Result row | What a rollout measured under Result Schema v1. | `crossed_target_radius`, `recoverable_crossing`. |
| Failure label | What happened or why the rollout terminated. | `terminal_label=no_crossing`. |
| Regression decision | Evaluator decision about progress or diagnostic status. | `decision_type=reject_progress`. |
| Runtime decision | System-intent decision during a run. | `decision_type=re_observe`. |
| Veto decision | Runtime-assurance response to a proposed action or mode. | `decision_type=veto_action`, `veto_status=safe_mode`. |
| Manual audit note | Free-text caveat when fields cannot explain the whole case. | `manual_audit_note`. |

`accept_progress`, `reject_progress`, and `mark_diagnostic` are evaluator or research decisions, not runtime control decisions.

`continue`, `retry`, `retreat`, `re_observe`, `safe_mode`, `abort`, and `switch_controller` are runtime-intent decisions.

## Required Fields

### Identity / Provenance

- `decision_schema_version`
- `decision_id`
- `benchmark_id`
- `benchmark_version`
- `experiment_id`
- `rollout_id`
- `case_id`
- `artifact_path`
- `source_script`

Optional:

- `timestamp`
- `git_commit`

### Link To Result Schema v1

- `result_schema_version`
- `result_row_ref`
- `terminal_label`
- `accepted_as_progress`
- `acceptance_reason`

### Mission / Task Context

- `mission_mode`
- `task_phase`
- `event_detected`
- `event_type`
- `event_step`
- `selected_controller`
- `controller_family`

### Evidence Summary

- `state_summary`
- `safety_level`
- `recoverability_level`
- `trust_flags`
- `failure_label`
- `regression_status`
- `subset_status`
- `known_phase34_recoverable_case`

### Decision

- `decision_type`
- `decision_reason`
- `decision_scope`
- `decision_authority`
- `fallback_available`
- `fallback_action`
- `veto_status`
- `veto_reason`
- `manual_audit_note`

## Optional Future Fields

These fields should remain optional until a future implementation can populate them consistently:

- `belief_summary`
- `uncertainty_summary`
- `recoverability_estimate`
- `recovery_margin`
- `risk_budget_remaining`
- `planner_options`
- `selected_option`
- `proposed_action`
- `thresholds_used`
- `resource_usage`
- `trust_scores`

Optional fields should be empty in CSV and `null` in JSONL when unavailable.

## Decision Type Enum

Allowed `decision_type` values:

- `continue`
- `retry`
- `retreat`
- `re_observe`
- `safe_mode`
- `abort`
- `switch_controller`
- `degrade_goal`
- `accept_progress`
- `reject_progress`
- `mark_diagnostic`
- `veto_action`
- `modify_action`
- `no_decision_recorded`
- `unknown`

## Decision Reason Enum

Allowed `decision_reason` values:

- `recoverable_crossing_detected`
- `no_crossing`
- `crossing_unrecoverable`
- `known_success_preserved`
- `known_success_regressed`
- `closest_approach_only`
- `subset_only`
- `safety_violation`
- `overspeed_risk`
- `instability_risk`
- `invalid_simulation`
- `missing_required_fields`
- `low_trust`
- `low_recoverability_margin`
- `fallback_available`
- `fallback_missing`
- `manual_audit_required`
- `unknown`

## Safety Level Enum

Allowed `safety_level` values:

- `nominal`
- `caution`
- `warning`
- `critical`
- `failed`
- `unknown`

## Recoverability Level Enum

Allowed `recoverability_level` values:

- `irrecoverable`
- `marginal`
- `recoverable`
- `robustly_recoverable`
- `unknown`

## Trust Flag Representation

`trust_flags` should be a semicolon-separated list in CSV and an array in JSONL.

Recommended initial flags:

- `low_perception_trust`
- `low_estimator_trust`
- `low_planner_trust`
- `low_controller_trust`
- `low_hardware_trust`
- `low_recoverability_trust`
- `prediction_mismatch`
- `sensor_dropout`
- `latency_spike`
- `controller_saturation`
- `none`
- `unknown`

Trust flags are evidence. They do not by themselves prove failure.

## Veto Status Representation

Allowed `veto_status` values:

- `not_applicable`
- `not_evaluated`
- `allow`
- `modify_action`
- `switch_controller`
- `retreat`
- `re_observe`
- `safe_mode`
- `abort`
- `blocked`
- `unknown`

A veto decision must include `veto_reason` and an evidence summary. A veto must not be hidden inside aggregate success or failure metrics.

## Fallback Representation

Fallback should be represented by:

- `fallback_available`: boolean
- `fallback_action`: enum-like string or short description

Recommended `fallback_action` values:

- `none`
- `continue_current_controller`
- `switch_controller`
- `reduce_action`
- `retreat`
- `re_observe`
- `safe_mode`
- `abort`
- `degrade_goal`
- `manual_review`
- `unknown`

If no fallback exists, set `fallback_available=false` and explain the implication in `manual_audit_note` when it affects the decision.

## Field Type Table

| Field | Type | Required | Definition |
| --- | --- | --- | --- |
| `decision_schema_version` | string | yes | Schema identifier, normally `decision_log_schema_v0`. |
| `decision_id` | string | yes | Unique decision record identifier. |
| `timestamp` | string | no | Event time or log time when available. |
| `benchmark_id` | string | yes | Benchmark family, such as `recoverability_benchmark`. |
| `benchmark_version` | string | yes | Benchmark version, such as `v1`. |
| `experiment_id` | string | yes | Experiment or run group identifier. |
| `rollout_id` | string | yes | Rollout identifier within the experiment. |
| `case_id` | string | yes | Benchmark case identifier. |
| `artifact_path` | string | yes | Path to the result or decision artifact. |
| `source_script` | string | yes | Script or tool that produced the decision record. |
| `git_commit` | string | no | Git commit when captured. |
| `result_schema_version` | string | yes | Linked result schema, normally `result_schema_v1`. |
| `result_row_ref` | string | yes | Reference to the result row, such as row id, row number, or artifact path plus row key. |
| `terminal_label` | enum string | yes | Controlled terminal label from Failure Label Taxonomy v0. |
| `accepted_as_progress` | boolean | yes | Whether linked result is accepted as progress. |
| `acceptance_reason` | string | yes | Reason for accepted, rejected, or diagnostic status. |
| `mission_mode` | string | yes | High-level mission mode. |
| `task_phase` | string | yes | Task phase such as `approach`, `event_review`, or `stabilize`. |
| `event_detected` | boolean | yes | Whether a relevant event was detected. |
| `event_type` | string | yes | Event type, such as `target_radius_crossing` or `none`. |
| `event_step` | integer | yes | Step of event, empty/null if not applicable. |
| `selected_controller` | string | yes | Selected controller or mode. |
| `controller_family` | string | yes | Controller family. |
| `state_summary` | string/object | yes | Compact state evidence used for the decision. |
| `safety_level` | enum string | yes | Safety level enum. |
| `recoverability_level` | enum string | yes | Recoverability level enum. |
| `trust_flags` | list/string | yes | Trust flags. |
| `failure_label` | string | yes | Failure or diagnostic label used as decision evidence. |
| `regression_status` | string | yes | Regression status such as `preserved`, `regressed`, or `not_applicable`. |
| `subset_status` | string | yes | Subset status such as `full_benchmark`, `diagnostic_subset`, or `not_applicable`. |
| `known_phase34_recoverable_case` | boolean | yes | Whether the case is a known Phase34 recoverable case. |
| `decision_type` | enum string | yes | Decision type enum. |
| `decision_reason` | enum string | yes | Decision reason enum. |
| `decision_scope` | string | yes | Scope such as `runtime`, `evaluator`, `veto`, or `manual_audit`. |
| `decision_authority` | string | yes | Authority such as `decision_manager`, `runtime_assurance`, `benchmark_evaluator`, or `human_auditor`. |
| `fallback_available` | boolean | yes | Whether a fallback was available. |
| `fallback_action` | string | yes | Selected or available fallback. |
| `veto_status` | enum string | yes | Veto status enum. |
| `veto_reason` | string | yes | Reason for veto status, empty only when `veto_status=not_applicable`. |
| `manual_audit_note` | string | yes | Free-text caveat or audit note. |

## CSV Representation

CSV is useful for simple spreadsheet inspection.

Rules:

- Booleans should be `true` or `false`.
- Missing optional values should be empty.
- List fields should use semicolon-separated values.
- JSON-like state summaries should be avoided unless quoted.
- `veto_reason` may be empty only when `veto_status=not_applicable`.

Recommended CSV header:

```csv
decision_schema_version,decision_id,timestamp,benchmark_id,benchmark_version,experiment_id,rollout_id,case_id,artifact_path,source_script,git_commit,result_schema_version,result_row_ref,terminal_label,accepted_as_progress,acceptance_reason,mission_mode,task_phase,event_detected,event_type,event_step,selected_controller,controller_family,state_summary,safety_level,recoverability_level,trust_flags,failure_label,regression_status,subset_status,known_phase34_recoverable_case,decision_type,decision_reason,decision_scope,decision_authority,fallback_available,fallback_action,veto_status,veto_reason,manual_audit_note
```

## JSONL Representation

Prefer JSONL for future machine-readable decision logs because each line can represent one decision event.

Rules:

- One JSON object per line.
- Booleans should be native JSON booleans.
- Missing optional values should be `null`.
- List fields should be arrays.
- Keep `decision_id` stable and unique.

Recommended future artifact examples:

- `analysis/example_decision_log_v0/decision_log.jsonl`
- `analysis/example_decision_log_v0/decision_log.csv`
- `analysis/example_decision_log_v0/summary.md`

Do not create those artifacts until a future implementation task explicitly asks for them.

## Minimal Example Logs

These examples are schema examples only. They are not historical artifact rewrites.

### Example 1: Continue After Recoverable Crossing

```json
{"decision_schema_version":"decision_log_schema_v0","decision_id":"dec_continue_001","timestamp":null,"benchmark_id":"recoverability_benchmark","benchmark_version":"v1","experiment_id":"example_run","rollout_id":"rollout_001","case_id":"case_r1p00_a150_t8000","artifact_path":"analysis/example/result.csv","source_script":"scripts/example.py","git_commit":null,"result_schema_version":"result_schema_v1","result_row_ref":"row:1","terminal_label":"success","accepted_as_progress":false,"acceptance_reason":"baseline preservation row, not a new progress claim","mission_mode":"nominal","task_phase":"event_review","event_detected":true,"event_type":"target_radius_crossing","event_step":1716,"selected_controller":"phase34_radius_priority","controller_family":"phase34_post_cross_sync","state_summary":"target radius crossed; recoverable basin reached","safety_level":"nominal","recoverability_level":"recoverable","trust_flags":["none"],"failure_label":"","regression_status":"preserved","subset_status":"full_benchmark","known_phase34_recoverable_case":true,"decision_type":"continue","decision_reason":"recoverable_crossing_detected","decision_scope":"runtime","decision_authority":"decision_manager","fallback_available":true,"fallback_action":"switch_controller","veto_status":"allow","veto_reason":"runtime assurance allowed continuation","manual_audit_note":""}
```

### Example 2: Reject False Progress

Closest approach improved but no target-radius crossing occurred.

```json
{"decision_schema_version":"decision_log_schema_v0","decision_id":"dec_reject_001","timestamp":null,"benchmark_id":"recoverability_benchmark","benchmark_version":"v1","experiment_id":"example_geometry_run","rollout_id":"rollout_002","case_id":"case_r0p98_a150_t8000","artifact_path":"analysis/example/result.csv","source_script":"scripts/example.py","git_commit":null,"result_schema_version":"result_schema_v1","result_row_ref":"row:2","terminal_label":"no_crossing","accepted_as_progress":false,"acceptance_reason":"closest approach improved but target radius was not crossed","mission_mode":"analysis","task_phase":"post_run_review","event_detected":false,"event_type":"none","event_step":null,"selected_controller":"example_transfer","controller_family":"transfer_family","state_summary":"closest approach improved; crossed_target_radius=false","safety_level":"nominal","recoverability_level":"unknown","trust_flags":["none"],"failure_label":"no_crossing","regression_status":"not_applicable","subset_status":"full_benchmark","known_phase34_recoverable_case":false,"decision_type":"reject_progress","decision_reason":"closest_approach_only","decision_scope":"evaluator","decision_authority":"benchmark_evaluator","fallback_available":false,"fallback_action":"none","veto_status":"not_applicable","veto_reason":"","manual_audit_note":"Closest approach is diagnostic, not success."}
```

### Example 3: Mark Subset Result As Diagnostic

```json
{"decision_schema_version":"decision_log_schema_v0","decision_id":"dec_diag_001","timestamp":null,"benchmark_id":"recoverability_benchmark","benchmark_version":"v1","experiment_id":"example_subset_run","rollout_id":"rollout_003","case_id":"selected_01","artifact_path":"analysis/example/subset.csv","source_script":"scripts/example_subset.py","git_commit":null,"result_schema_version":"result_schema_v1","result_row_ref":"row:3","terminal_label":"no_crossing","accepted_as_progress":false,"acceptance_reason":"subset diagnostic only","mission_mode":"analysis","task_phase":"post_run_review","event_detected":false,"event_type":"none","event_step":null,"selected_controller":"weak_tangential","controller_family":"phase37b_like_subset","state_summary":"selected subset row; no crossing","safety_level":"nominal","recoverability_level":"unknown","trust_flags":["none"],"failure_label":"near_crossing","regression_status":"not_applicable","subset_status":"diagnostic_subset","known_phase34_recoverable_case":false,"decision_type":"mark_diagnostic","decision_reason":"subset_only","decision_scope":"evaluator","decision_authority":"benchmark_evaluator","fallback_available":false,"fallback_action":"none","veto_status":"not_applicable","veto_reason":"","manual_audit_note":"Subset evidence cannot establish full-benchmark progress."}
```

### Example 4: Re-observe Due To Low Trust

Hypothetical future runtime row.

```json
{"decision_schema_version":"decision_log_schema_v0","decision_id":"dec_reobserve_001","timestamp":null,"benchmark_id":"recoverability_benchmark","benchmark_version":"v1","experiment_id":"future_runtime_run","rollout_id":"rollout_004","case_id":"case_future_low_trust","artifact_path":"analysis/future/decision_log.jsonl","source_script":"scripts/future_decision_logger.py","git_commit":null,"result_schema_version":"result_schema_v1","result_row_ref":"pending","terminal_label":"unknown","accepted_as_progress":false,"acceptance_reason":"runtime decision, not progress claim","mission_mode":"cautious","task_phase":"observe","event_detected":false,"event_type":"none","event_step":null,"selected_controller":"hold_orbit_state","controller_family":"future_runtime_controller","state_summary":"pose estimate inconsistent across observations","safety_level":"caution","recoverability_level":"marginal","trust_flags":["low_estimator_trust"],"failure_label":"unknown","regression_status":"not_applicable","subset_status":"not_applicable","known_phase34_recoverable_case":false,"decision_type":"re_observe","decision_reason":"low_trust","decision_scope":"runtime","decision_authority":"decision_manager","fallback_available":true,"fallback_action":"re_observe","veto_status":"not_evaluated","veto_reason":"no proposed risky action evaluated","manual_audit_note":"Hypothetical future trust example."}
```

### Example 5: Veto Due To Overspeed Risk

Hypothetical future runtime-assurance row.

```json
{"decision_schema_version":"decision_log_schema_v0","decision_id":"dec_veto_001","timestamp":null,"benchmark_id":"recoverability_benchmark","benchmark_version":"v1","experiment_id":"future_veto_run","rollout_id":"rollout_005","case_id":"case_future_overspeed","artifact_path":"analysis/future/decision_log.jsonl","source_script":"scripts/future_decision_logger.py","git_commit":null,"result_schema_version":"result_schema_v1","result_row_ref":"pending","terminal_label":"overspeed","accepted_as_progress":false,"acceptance_reason":"safety veto blocks clean progress","mission_mode":"recovery","task_phase":"approach","event_detected":true,"event_type":"overspeed_risk_prediction","event_step":900,"selected_controller":"aggressive_transfer","controller_family":"experimental_transfer","state_summary":"predicted speed exceeds threshold under proposed action","safety_level":"critical","recoverability_level":"marginal","trust_flags":["low_controller_trust"],"failure_label":"overspeed","regression_status":"safety_regression","subset_status":"full_benchmark","known_phase34_recoverable_case":true,"decision_type":"veto_action","decision_reason":"overspeed_risk","decision_scope":"veto","decision_authority":"runtime_assurance","fallback_available":true,"fallback_action":"safe_mode","veto_status":"safe_mode","veto_reason":"predicted overspeed risk exceeds hard threshold","manual_audit_note":"Hypothetical final-veto evidence; not a formal safety proof."}
```

## Rules For Missing Fields

- Required identity and link fields must not be missing in future decision logs.
- Optional fields should be empty in CSV and `null` in JSONL.
- Missing decision evidence should produce `decision_type=unknown`, `decision_reason=manual_audit_required`, or `decision_reason=missing_required_fields`, not optimistic acceptance.
- Missing `terminal_label` blocks an evaluator decision of `accept_progress`.
- Missing `veto_reason` is allowed only when `veto_status=not_applicable`.
- Missing `trust_flags` should be represented as `unknown`, not as `none`.
- Missing safety or recoverability evidence should set the corresponding enum to `unknown`.

## Rules For Accepted / Rejected / Diagnostic Decisions

- `accept_progress` requires the linked result row to satisfy Recoverability Regression Policy v0.
- `reject_progress` should explain the blocking reason, such as `closest_approach_only`, `known_success_regressed`, `safety_violation`, or `missing_required_fields`.
- `mark_diagnostic` should preserve useful evidence while preventing false progress claims.
- A result can be scientifically useful even if `decision_type=mark_diagnostic`.
- `accepted_as_progress=true` should not appear with `decision_type=reject_progress` or `decision_type=mark_diagnostic`.
- `terminal_label=unknown` should normally require `manual_audit_note` and should block accepted progress until resolved.
- A decision log must not hide failed retries, vetoes, or aborts inside aggregate metrics.
- A decision log must not claim formal safety.

## Runtime Decision Rules

- `continue` should require acceptable safety and recoverability evidence.
- `retry` should identify why retry is available and what retry-ready state exists.
- `retreat` should identify the risk or margin loss that motivated withdrawal.
- `re_observe` should identify uncertainty, low trust, or ambiguous event evidence.
- `safe_mode` should identify the safety condition or degraded capability.
- `abort` should identify why no acceptable continuation or retry remains.
- `switch_controller` should identify the old controller, selected controller, and reason.
- `veto_action` must include `veto_reason`, `veto_status`, and evidence summary.

## What Is Design-Only In Week 5

The following are design-only in this Week 5 artifact:

- Decision Manager implementation.
- Runtime Assurance / Final Veto implementation.
- Trust Manager implementation.
- Veto monitor.
- Controller switching logic.
- Decision-log validation script.
- Example analysis artifacts under `analysis/example_decision_log_v0/`.
- Any hardware, ROS, sensor, or robotics integration.
- Any formal-safety claim.

This document only defines the logging schema and interpretation rules.

## Week 6 Handoff Questions For ROAHM / Contact Recoverability Notes

Week 6 should answer:

- How can plug-insertion contact phases map to decision states?
- What does `event_detected` mean for first contact?
- What would justify retry versus retreat in plug insertion?
- What would justify `re_observe` when pose confidence is low?
- What contact states should become diagnostic labels?
- What must not be claimed when mapping ROAHM concepts into the spacecraft repo?
- How should contact force, contact mode, and pose confidence enter `state_summary` without implying hardware validation?
- Which contact events should trigger `EVENT_REVIEW` rather than automatic continuation?
- How should final-veto examples remain hypothetical until implemented?

## Week 5 Completion Rule

Week 5 is complete when this document exists, the protected regression guard still passes, and no historical evidence has been modified.
