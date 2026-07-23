# Recovery Branch Runner v0

## Status

Infrastructure implemented; recovery experiment not run.

Completed: 2026-07-19

This document describes runner infrastructure only. No recovery comparison experiment has been performed.

## Purpose

Recovery Branch Runner v0 provides a bounded orchestration layer around the frozen branch state and Recovery Branch Executor v0. It can carry one selected branch through a small number of in-memory transitions, record deterministic infrastructure diagnostics, and stop on the subset of frozen conditions that current evidence can evaluate.

The runner does not select a branch, compare actions, compute recovery performance, publish experiment results, or claim recovery success.

## Relationship To Recovery Branch Executor v0

The dependency direction is:

```text
frozen branch_state.json
        |
branch-state integrity validation
        |
one selected branch ID
        |
Recovery Branch Executor v0, one transition at a time
        |
runner-owned continuation state
        |
stop-condition report and optional bounded JSONL diagnostics
```

The executor remains responsible for generating the frozen branch action, applying unchanged Final Veto evaluation, and delegating an allowed action to `step_phase34_35_transition`. The runner owns only iteration, transition counts, continuation state, stop-condition ordering, and bounded diagnostic records.

The original `branch_state.json` is validated on every executor call and is never mutated. Post-branch continuation state remains an in-memory runner value rather than a rewritten branch-state artifact.

## Runner Interface

The programmatic entry point is:

```python
run_recovery_branch(
    branch_id,
    *,
    branch_state_path,
    horizon_steps,
    output_dir,
)
```

Only one of the four frozen branch IDs may be selected for a call. The runner never expands a call into four jobs and contains no comparison logic.

Recovery Branch Runner v0 accepts horizons from 1 through 32 transitions. The cap is an infrastructure safeguard, not a scientific recovery horizon or a claim that 32 steps are sufficient. The frozen 10,000-step recovery horizon remains unavailable.

## Default And CLI Safety

Running the script without a mode prints help and performs no rollout.

Supported non-executing modes are:

- `--plan`: validates one branch plan and prints deterministic plan metadata;
- `--validate-only`: validates the frozen state, horizon, and optional output path.

Both modes avoid executor calls and create no output directory or artifact. An execution path exists only behind the explicit `--execute-nonformal` mode. This task did not invoke that mode.

## Runner State Tracking

Each immutable `RecoveryBranchRunnerRecord` contains:

| Field | Meaning |
| --- | --- |
| `branch_id` | the one selected frozen branch |
| `step` | one-based post-branch runner step |
| `state_hash` | canonical hash of the current state after the step boundary |
| `action` | generated action, or null for explicit abort |
| `transition_executed` | whether a physical transition occurred |
| `terminal_reason` | triggered stop label or `continue` |
| `valid` | validity of the bounded execution record |

This is an internal runner record. It is not Result Schema v1, a formal result row, a recovery-success record, or a publication artifact.

## Stop Condition Architecture

The framework declares the frozen labels:

- `recovery_success`;
- `overspeed`;
- `instability`;
- `unsafe_state`;
- `invalid_simulation`;
- `invalid_recovery_evaluation`;
- `action_rejected`;
- `explicit_abort`;
- `recovery_horizon_exhausted`;
- `total_horizon_exhausted`.

Each condition has one infrastructure status: `triggered`, `clear`, or `not_evaluated`.

Current evaluation is deliberately limited:

- overspeed uses the frozen strict realized `speed_ratio > 1.90` rule;
- invalid simulation checks finite realized state and speed ratio;
- action rejection and explicit abort use executor terminal evidence;
- recovery and total horizon exhaustion use realized transition counts;
- recovery success remains `not_evaluated` because this bounded runner does not
  collect the complete crossing/recoverability summary or use the frozen
  10,000-transition horizon;
- instability and unsafe-state remain `not_evaluated` because this runner does
  not supply explicit instrumentation for them.

The runner therefore cannot claim recovery success even when a bounded trajectory remains below the overspeed threshold.

Pure reusable evaluators and their evidence limits are documented in
`docs/experiments/recovery_evaluators_v0.md`. They are accepted by the shared
stop-condition layer but are not invoked optimistically by this runner when its
required evidence is unavailable.

## Logging Boundary

Diagnostics are disabled unless explicitly requested. When enabled, the runner writes only:

`recovery_branch_diagnostics.jsonl`

The JSONL stream is bounded by the 32-step infrastructure cap. Each line uses deterministic UTF-8 JSON with sorted keys and stable separators. Records contain state hashes and control-boundary evidence, not scientific acceptance fields or counterfactual comparisons.

The writer refuses:

- the frozen recovery experiment directory;
- Final Veto formal evidence paths;
- protected Phase34-37 analysis directories;
- overwrite of an existing diagnostic log.

It never creates `results.csv`, `paired_results.csv`, a formal decision log, a summary, or a comparison plot.

## Current Limitations

- No recovery comparison experiment has run.
- No branch ranking or winner exists.
- Recovery success is not evaluated by this bounded runner.
- Pure instability and unsafe-state evaluators exist, but this runner does not
  have the required explicit evidence to call them.
- The runner does not implement the frozen 10,000-step horizon.
- The runner does not write Result Schema v1 or Decision Log Schema v0 artifacts.
- No controller switching, adaptive policy, learning, optimization, or action tuning is present.
- No formal safety, hardware, deployment, or cross-domain claim is supported.

A separate no-execution preflight and artifact layer is now documented in
`docs/experiments/recovery_experiment_runner_v0.md`. It validates the future
full-horizon contract and synthetic publication behavior without changing this
runner's 32-step cap. A real execution entry point still requires separate
authorization before any four-branch comparison is executed.
