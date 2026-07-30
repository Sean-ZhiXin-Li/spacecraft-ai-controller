# Staged Recovery Instrumentation Validation v0

## Status

Measured instrumentation-validation trace completed; staged recovery execution not authorized.

Executed: 2026-07-30

Implementation commit: `942b8e9108ace5f0c481347898a8ddfb86b92548`.

Staged recovery execution: `not_authorized`.

## Purpose

Stage 0C validates whether the Stage 0B observational logger can receive evidence from one existing bounded recovery path without changing that path. It freezes one logger-disabled baseline and one logger-enabled observed execution from independently reloaded copies of the same canonical branch state.

This is an instrumentation engineering validation. It is not a recovery experiment, controller experiment, phase-policy test, or extension of the published four-branch diagnostic.

## Source Contracts

The integration reuses these repository contracts without modification:

- `runtime_assurance/recovery_branch_executor.py` for branch action generation, Final Veto evaluation, prediction, and one-step transition execution;
- `scripts/run_recovery_branch_runner.py` for bounded counter and terminal semantics and the unchanged 32-step infrastructure cap;
- `runtime_assurance/staged_recovery_instrumentation.py` for status-bearing observations and pure orbital derivations;
- `runtime_assurance/staged_recovery_runtime_logger.py` for immutable event construction, event hashing, and deterministic JSONL;
- the frozen recovery manifest and canonical branch state for case, seed, hashes, branch geometry, and simulator constants.

## Frozen Validation Configuration

- Validation ID: `staged_recovery_instrumentation_validation_v0`
- Source: canonical recovery branch state
- Branch: `velocity_opposed_thrust_v0`
- Validation horizon: 8 realized physical transitions
- Pair: one logger-disabled baseline plus one logger-enabled observed run
- Published trace: observed run only
- Expected events: one initial snapshot, eight transitions, one terminal event

The eight-transition bound is an instrumentation validation allocation. It is neither the 32-step bounded-runner cap nor the frozen 10,000-transition recovery horizon, and it does not establish scientific sufficiency.

## Observer Integration

The adapter owns one optional callback boundary. With no observer, the same one-step executor loop runs and returns the same immutable semantic transcript. With an observer, each immutable snapshot is passed after the runtime has fixed the corresponding evidence.

The callback receives no mutable simulator, controller, monitor, or continuation object. Its return value must be `None`; other values are rejected and never consumed as runtime input. An observer exception is an infrastructure failure that stops validation and cannot become a scientific terminal outcome.

## Runtime Evidence Boundary

The initial snapshot preserves the canonical state and counters without an action or transition. Each transition preserves the measured pre-state, proposed action, Final Veto prediction and decision, predicted state, executed action, realized state, evaluator-status record, and post-transition counters. The terminal snapshot preserves final state and counters without adding a transition.

The underlying runtime terminal reason and the validation wrapper reason remain separate. `instrumentation_validation_complete` is a diagnostic completion label, not Recovery Success v0, simulator success, horizon sufficiency, or a scientific failure label.

## Paired Equivalence

Required equality covers source identity, initial state, recovery and total counters, proposed and executed actions, monitor predictions and decisions, action dispositions, predicted and realized state hashes, predicted and realized speed ratios, evaluator-status sequences, runtime terminal reason, and final state. Comparisons use exact deterministic equality; Stage 0C adds no tolerance.

Publication is forbidden if any required check is false or unavailable.

## Stage 0B Mapping

Stage 0C uses the frozen Stage 0B session to construct and hash events. Stage 0B's original synthetic trace-manifest contract remains unchanged. Stage 0C publishes a separate measured-instrumentation-validation trace manifest around those canonical events with:

- `trace_classification = measured_instrumentation_validation`;
- `runtime_source = bounded_recovery_validation_path`;
- `scientific_result = false`.

This preserves the Stage 0B milestone while making the new measured classification explicit at the Stage 0C boundary.

## Predicted and Realized Evidence

Predicted and realized states, state hashes, speed ratios, and overspeed headroom remain separate. Progress deltas use measured pre/post states only. State-hash equality denotes exact canonical identity and is not interpreted as physical distance.

## Field Completeness

Every Stage 0A field and Stage 0B event field receives a measured-run completeness row. The report distinguishes runtime-supplied fields, logger-derived fields, expected event availability, correctly unavailable fields, invalid fields, unexpectedly missing fields, and unsupported fields.

Staged phase, phase-transition, no-progress, handoff-readiness, and retreat fields are expected to remain `not_evaluated` because no staged phase runtime exists. Recovery Success v0, instability, unsafe-state, and simulator-success evidence are not invented by this validation wrapper. Available correction authority remains unsupported.

The eight-step validation allocation is not recorded as `recovery_horizon_remaining`; doing so would conflate an engineering bound with the frozen scientific horizon.

## Artifacts

After a successful clean-commit measured validation, the atomically published bundle contains exactly:

- `validation_manifest.json`
- `trace_manifest.json`
- `staged_recovery_trace.jsonl`
- `equivalence_report.json`
- `field_completeness.json`
- `summary.md`

The baseline trace is never published. Partial publication and overwrite are forbidden.

## Failure Policy

Baseline failure, observed-run failure, observer failure, equivalence failure, field-completeness failure, writer failure, or staged-validation failure publishes nothing. No automatic retry, comparison relaxation, branch substitution, or horizon adjustment is permitted.

## Protected Evidence

The output directory is separate from the frozen recovery experiment, mechanism diagnosis, staged architecture, Stage 0A, Stage 0B, Final Veto evidence, Phase34-37 evidence, runtime code, controller code, and simulator code. Publication rejects protected targets and existing user paths.

## Current Limitations

The integration covers one existing bounded branch path and does not establish logger completeness for every future runner. No staged phase runtime supplies phase fields. No handoff-readiness or correction-authority evaluator exists. No numerical phase guard, no-progress threshold, hysteresis parameter, or phase action law is frozen.

## Stage 1 Boundary

After Stage 0C, the next smallest milestone remains offline validation of transition guards, priority, missing-value behavior, no-progress structure, and hysteresis structure. No staged rollout is authorized by this trace.

## Claim Restrictions

This validation cannot establish recovery performance, task recovery, staged-policy effectiveness, controller optimality, formal safety, hardware validity, cross-domain effectiveness, or deployment readiness.

This trace validates observational instrumentation on one bounded runtime path. It is not a recovery-performance experiment, staged-controller execution, phase-policy validation, task-recovery result, formal safety result, hardware result, or deployment result.
