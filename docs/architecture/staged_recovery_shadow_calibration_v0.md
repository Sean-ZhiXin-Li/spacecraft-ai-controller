# Staged Recovery Shadow Calibration v0

Status: Calibration infrastructure implemented; frozen trace capture and candidate analysis require separately committed execution phases.

Completed: 2026-08-11

## Purpose

Stage 1B-B expands the observational shadow boundary from four smoke traces to a fixed thirteen-trace engineering calibration set. It evaluates 216 declared anti-chatter and evidence-window configurations without granting any shadow output physical authority.

## Physical Trace Matrix

Each of the four frozen branch-state registry members is paired with the existing zero-action, velocity-opposed, and tangential-error-correction branches. A thirteenth trace records the existing explicit-abort branch from the legacy member. Every trace has one observer-disabled baseline and one fresh observer-enabled execution, bounded at 32 recovery transitions.

Baseline traces are retained only in memory for exact comparison. Only observed traces may be published. The trace-set command performs exactly 26 bounded executions and has no retry or scientific override option.

## Authority Boundary

`shadow_output_consumed_by_physical_runtime = false`

The callback receives immutable runtime snapshots and returns `None`. Any non-`None` return or observer exception is an infrastructure failure. Shadow phase, counters, candidate parameters, and guard recommendations are never inputs to branch action generation, simulator transitions, Final Veto, evaluator results, termination, or a real phase runtime.

## Evidence Preservation

Every observed trace record preserves the canonical Stage 0B event, the complete Stage 1A guard-evaluation inventory, and the default Stage 1B-A shadow record. `not_evaluated`, `unsupported`, `policy_unresolved`, and `invalid` remain distinct. None is converted to a favorable or unfavorable Boolean.

## Candidate Grid

The Cartesian product contains exactly 216 configurations:

- hazard-clear consecutive evidence: 1, 2, or 3 steps;
- minimum phase dwell: 1, 2, or 4 steps;
- no-progress window: 2, 4, or 8 transitions;
- required improving components: 2 or 3;
- consecutive no-progress windows: 1 or 2;
- cooldown: 0 or 2 steps;
- transition budget: 8.

Candidate IDs and ordering are canonical. No CLI parameter override exists.

## No-Progress Convention

The engineering-only component evidence is radius-gap reduction, absolute radial-component reduction, absolute tangential-error reduction, and overspeed-headroom increase. A component improves only when its cumulative signed change over the candidate window is strictly positive. Energy proxy evidence is excluded. No combined scalar progress score is created.

The deterministic stuck rule requires at least four consecutive blocked recommendations for the same phase while the current shadow phase differs. Low transition count by itself is not stuck evidence.

## Offline Replay

All 216 candidates replay the same 13 committed observed traces, producing exactly 2,808 pure shadow evaluations and zero physical executions. Ranking uses the declared lexicographic tuple. Hard disqualifiers are evaluated before ranking and are never relaxed after results are observed.

## Engineering Candidate

The highest-ranked nondisqualified configuration is frozen as `engineering_candidate_v0`. It is the best candidate only under this fixed engineering trace set and ranking contract. It remains `shadow_only = true`, `active_authority = false`, and `staged_recovery_execution = not_authorized`.

## Claim Restrictions

The result may establish deterministic shadow-policy selection with exact physical neutrality. It cannot establish controller improvement, recovery improvement, optimality, formal safety, validated active thresholds, handoff readiness, or deployment readiness.
