# Recovery Branch Boundary Registry v0

Status: Repository-backed case-specific branch boundaries implemented and four deterministic branch-state members frozen.

Completed: 2026-08-03

## Purpose

The boundary registry replaces the unsupported assumption that all Final Veto cases possess a common state after 27 nominal transitions. It assigns every source case one explicit boundary backed by already frozen repository evidence.

## Boundary types

`legacy_fixed_prefix` preserves the published canonical state after 27 realized transitions and before the step-28 action. `source_declared_fixed_prefix` is available only when frozen source metadata explicitly declares a count. `monitor_off_preterminal_state` identifies the last complete state before a frozen monitor-off terminal transition. `ineligible` preserves cases without an unambiguous reproducible boundary.

## Transition indexing

The Phase34 and Phase35 rollout loops call `PreTransitionActionContext(step=N)` before executing transition `N`. Its `current_state` is therefore the state after `N-1` realized transitions. The loop then executes transition `N`, invokes `PostTransitionObservation(step=N)`, evaluates terminal predicates on the realized next state, and stores `steps=N`.

For the frozen transition-22 overspeed case, the branch state is the pre-transition step-22 state, which is physically the state after 21 realized transitions. Transition 22 is executed only to verify the frozen terminal count and `overspeed` reason. A state after transition 22 is terminal and is never used as the branch state.

## Source evidence

The canonical boundary is bound to `analysis/recovery_action_branching_nonformal_v0/branch_state.json`. Terminal counts and reasons are bound to exact monitor-off rows in `analysis/final_veto_ablation_v0/results.csv`. File hashes, canonical row hashes, case configuration, simulator configuration, and controller-source hashes are validated before extraction.

## Phase34 boundaries

The eight preservation cases use their frozen pre-success states. Their boundary transition counts range from 477 through 4792 and are one less than their frozen successful terminal transitions.

## Phase35 boundaries

The canonical thrust-8000 case retains the legacy transition-27 boundary. The four noncanonical stress cases use pre-overspeed boundaries after 21, 24, 25, and 26 realized transitions, respectively.

## Multi-boundary interpretation

Selection metrics are evaluated at each case's own frozen boundary. These registry members are multi-boundary calibration inputs, not synchronized-time samples. Comparing their metrics does not imply equal elapsed physical progress or recovery performance.

## Determinism

For monitor-off preterminal boundaries, extraction runs through the terminal transition and requires its observed count and reason to match frozen evidence. The published branch state remains the prior valid state. Selected generated cases must exactly match fresh reproductions in boundary metadata, Cartesian state, derived values, prediction, action trace, state trace, and canonical payload hash.

## Authority boundary

Boundary extraction executes existing nominal source behavior only. It does not execute a recovery branch, generate a new action, alter Final Veto, implement a staged phase, or authorize staged recovery execution.
