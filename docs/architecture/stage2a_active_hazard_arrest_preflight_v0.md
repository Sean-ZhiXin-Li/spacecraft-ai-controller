# Stage 2A Active Hazard Arrest Authority Boundary Preflight v0

## Status

Authority boundary preflight frozen; no staged recovery phase has active authority.

Completed: 2026-08-12

## Purpose

This read-only preflight defines the evidence, action, Final Veto, release,
abort, isolation, and claim boundaries that a separately reviewed Stage 2A
implementation would have to satisfy. It does not implement a gate, controller,
phase action, runtime switch, or experiment, and it authorizes no physical
intervention.

## Frozen Stage 1B Basis

The checked-in Stage 1B calibration validates 216 candidates over 2,808 offline
replays with zero physical executions during ranking. Its selected source is
`shadow_candidate_hc2_d4_w2_r3_n1_cd0_tb8`. The published candidate remains:

- `shadow_only = true`;
- `active_authority = false`;
- `scientific_threshold_validation = not_performed`;
- `staged_recovery_execution = not_authorized`.

Its dwell, cooldown, no-progress, consecutive-clear, and transition-budget
values are engineering shadow parameters. They are not active entry, release,
or safety thresholds. The selected replay recorded zero `hazard_arrest` phase
entries and 2,817 unavailable guard observations, so it cannot validate active
hazard-arrest behavior.

## Authority Model

```text
existing physical recovery branch
        |
        | normal authority
        v
physical action
        |
        +---- unchanged Final Veto remains active

valid hazard evidence
        |
        v
future Stage 2A hazard-arrest gate
        |
        | only after separate authorization
        v
one bounded provisional intervention proposal
        |
        +---- unchanged Final Veto evaluates the proposal
        v
release to the predeclared existing recovery branch or terminate
```

Today the gate and intervention path do not exist. The future gate must not
consume shadow phase recommendations as authority and must never bypass Final
Veto or the existing adverse-stop ordering.

## READY NOW

Repository-backed facts usable in a future implementation are:

1. `realized_speed_ratio` is derived from the measured Cartesian state and the
   frozen target circular speed semantics.
2. Realized overspeed is the strict comparison
   `realized_speed_ratio > 1.90`; exactly `1.90` is clear under this hazard
   definition.
3. `predicted_speed_ratio` is a separate one-step prediction. Predicted
   overspeed is `predicted_speed_ratio > 1.90`; predicted clear is not realized
   clear.
4. Signed overspeed headroom is derived relative to the same frozen threshold.
   It is diagnostic margin, not a separate trigger threshold.
5. Final Veto `one_step_overspeed_veto_v0` evaluates every non-abort proposal.
   It allows a nominal action at `<= 1.90`; at `> 1.90` it records
   `predicted_nominal_overspeed`, evaluates its zero-action fallback, and returns
   a veto decision. The recovery executor treats a veto as
   `recovery_action_rejected`, executes zero transitions, and does not silently
   execute the fallback or select another branch.
6. Measured, derived, predicted, externally supplied, not-evaluated, invalid,
   and unsupported evidence remain distinct through Stage 0A, Stage 0B, and
   Stage 1A.
7. Existing immutable branch-state loading, one-step prediction, exact
   prediction/realization equality, action disposition, counters, terminal
   evidence, and observational logging can support a bounded experiment.
8. The existing `velocity_opposed_thrust_v0` action is
   `-0.25 * v / ||v||` with the repository's zero-vector handling. It is the
   most direct existing action for reducing speed and already uses unchanged
   Final Veto. It is only a provisional future intervention candidate.
9. Existing stop priority places invalid simulation, invalid recovery
   evaluation, realized overspeed, instability, unsafe state, action rejection,
   and explicit abort ahead of recovery success and horizon exhaustion.
10. Explicit abort already has terminal semantics with no action and no
    physical transition.

## NOT READY

- No active hazard-arrest action law is frozen. Reusing
  `velocity_opposed_thrust_v0` requires an explicit mapping and bounded
  experiment; the architecture warns that velocity-opposed thrust can erase
  useful radial and tangential motion and remove excessive energy.
- No scientifically validated active entry threshold exists beyond the one
  frozen overspeed definition. Shadow calibration parameters cannot be used as
  scientific or active thresholds.
- No hazard-arrest completion guard, release hysteresis, active dwell limit,
  maximum intervention count, or active authority budget is frozen.
- Absence of overspeed alone does not establish stabilization or readiness to
  resume task recovery.
- `available_correction_authority` is unsupported. It cannot be read as false,
  true, zero, or adequate authority.
- Instability, unsafe-state, and handoff-readiness evidence may be unavailable
  unless a compatible external evaluator supplies valid values.
- No handoff-readiness evaluator or active nominal handoff exists.
- No retreat target, retreat action, retreat-success predicate, or autonomous
  abort policy exists.
- The Stage 1B engineering candidate observed no hazard-arrest entry and cannot
  establish intervention efficacy, release behavior, or hazard false-positive
  and false-negative rates.

## Evidence Semantics

| Evidence | Level | Current meaning | Active use today |
| --- | --- | --- | --- |
| Cartesian state | measured | Supplied runtime state | Observation only |
| Realized speed ratio | derived from measured state | Current strict hazard comparison | No active gate implemented |
| Predicted speed ratio/state | predicted/external then instrumented | One-step proposed-action consequence | Final Veto evidence only |
| Final Veto decision | externally supplied/measured | Allow or reject one proposal | Existing hard action boundary |
| Overspeed headroom | derived | Signed margin to `1.90` | Diagnostic only |
| State/instrumentation validity | measured/derived status | Whether required evidence is structurally usable | Must block active use when invalid |
| Instability/unsafe state | externally supplied | Adverse evidence when a valid evaluator supplies it | Missing values cannot be treated as clear |
| Correction authority | unsupported | No evaluator exists | Prohibited as trigger/release input |
| Handoff readiness | not evaluated | No evaluator exists | Cannot authorize handoff |
| Shadow phase and candidate parameters | shadow-derived | Engineering recommendations and anti-chatter state | No physical authority |

Unknown, invalid, unsupported, and policy-unresolved values must remain null or
status-bearing. They cannot be converted to favorable Boolean evidence.

## Provisional Future Action

The most defensible existing physical candidate is a one-step proposal produced
by `generate_velocity_opposed_action`, mapped explicitly from
`velocity_opposed_thrust_v0` to a provisional hazard-arrest experiment action.
The justification is narrow: it directly opposes measured velocity and has
existing deterministic generation, clipping, prediction, Final Veto, and
execution tests.

This choice is not a claim that the action is safe, optimal, stabilizing, or
recovery-producing. Historical one-case mechanism evidence shows that sustained
velocity-opposed thrust reduced speed strongly while degrading useful motion
and target geometry. A future Stage 2A experiment therefore must permit only a
predeclared bounded intervention, log energy and component degradation, and
return authority to the existing branch rather than continue indefinitely.

## Future Entry Contract

A future implementation may propose hazard-arrest authority only when all of
the following are present and valid:

- immutable case, branch-state, simulator, constants, and implementation
  provenance;
- finite measured Cartesian state;
- valid Stage 0A instrumentation and recovery-evaluation status;
- a frozen hazard trigger using existing semantics, separately reviewed before
  execution;
- complete proposed-action and one-step predicted-state evidence;
- an unchanged Final Veto decision for the provisional action;
- an explicit authority token scoped only to `hazard_arrest`, one predeclared
  experiment, and a bounded intervention count.

The minimal future experiment trigger is pre-action evidence that the
predeclared normal branch's proposed action has a valid one-step predicted speed
ratio strictly greater than `1.90`, while the measured current state is valid
and its realized speed ratio is at most `1.90`. This is the existing strict
hazard comparison applied at the already defined prediction boundary. It does
not authorize execution today. A current realized speed ratio greater than
`1.90` remains the existing adverse terminal condition and must not be converted
into authority to execute another action.

## Future Release Contract

Release must return authority only to the exact predeclared existing recovery
branch and must not route to another staged phase. For the minimal future
experiment, at most one provisional intervention proposal is permitted per
trace. After one allowed intervention transition, release requires a fresh
finite measured state with `realized_speed_ratio <= 1.90`, a fresh prediction
for the resumed existing branch with `predicted_speed_ratio <= 1.90`, valid
instrumentation and evaluator evidence, and no active adverse stop. These are
the clear complements of the existing strict hazard rule, not new thresholds.
Passing this release gate is not stabilization or task recovery. If the release
gate is incomplete or unfavorable, the experiment must not issue a second
intervention; it must stop under a separately declared infrastructure outcome
or existing adverse terminal reason. The shadow candidate's
`hazard_clear_consecutive_steps = 2` is an engineering replay value and is not
used by this one-intervention release rule.

Loss of evidence while intervention authority is held must never be interpreted
as release. It must block further intervention and follow the separately frozen
abort/failure path.

## Immediate Block And Abort Conditions

A future gate must block the intervention proposal before physics when:

- state, instrumentation, prediction, provenance, action, or evaluator evidence
  is missing, malformed, nonfinite, invalid, or inconsistent;
- the registry member or configuration hashes fail validation;
- the proposed action differs from the frozen provisional action mapping;
- Final Veto rejects the proposal or its prediction fails;
- action clipping or proposed-versus-executed equality violates the frozen
  action contract;
- active authority is requested for any phase other than `hazard_arrest`;
- the intervention bound or experiment bound is exhausted.

Existing adverse terminal evidence retains priority. Invalid simulation,
invalid recovery evaluation, realized overspeed after an executed transition,
instability, unsafe state, action rejection, and externally supplied explicit
abort must stop the active path according to the frozen runtime semantics.
Explicit abort remains a terminal decision, never a physical fallback action.

## Authority Remaining Prohibited

Active authority remains prohibited for `stabilization_assessment`,
`radial_recommitment`, `tangential_alignment`, `crossing_preparation`,
`recoverability_verification`, `nominal_handoff`, and `retreat`. No staged edge
is executable. Explicit abort may be externally supplied as terminal evidence
but is not an active recovery phase action. The shadow FSM and
`engineering_candidate_v0` remain observational only.

## FUTURE STAGE 2A EXPERIMENT CONTRACT

The smallest implementation task should create an authority-enforcing adapter
and tests before any measured execution. A later, separately reviewed experiment
should freeze:

- one registry member selected before outcomes are observed;
- one existing recovery branch as the baseline/normal-authority path;
- one provisional one-step `velocity_opposed_thrust_v0` mapping for
  `hazard_arrest` only;
- exactly one maximum hazard-arrest intervention proposal per active trace;
- unchanged Final Veto and stop-condition priority;
- a maximum of 32 physical transitions for infrastructure bounding, explicitly
  not a scientific recovery horizon;
- one baseline run and one fresh active-boundary run;
- the one-intervention release rule above frozen before execution;
- no automatic retry, no CLI threshold/action override, and atomic publication.

Required isolation checks are identical initial state and provenance, unchanged
nominal branch behavior outside intervention events, no authority leakage to
other phases, every provisional action passing through Final Veto, exact
prediction/realization consistency, bounded transition and intervention counts,
and terminal reasons produced only by existing runtime semantics.

Required engineering metrics include trigger count, proposed/allowed/rejected
intervention count, physical transition count, realized and predicted speed
ratios, headroom, action magnitude and geometry, useful radial/tangential motion,
energy-proxy change, recoverability components, release count and reason,
blocked/abort reason, and authority-isolation failures. Recovery Success must
not be the sole or required objective of this authority-boundary experiment.

Experiment failure includes any unauthorized phase/action, Final Veto bypass,
invalid evidence consumption, authority after release/termination, intervention
budget overrun, prediction/realization mismatch, unplanned fallback execution,
or protected-artifact change.

## Claim Restrictions

Even a successful future Stage 2A bounded experiment would establish only that
a narrowly scoped hazard-arrest authority adapter executed under its declared
isolation and evidence contract for the frozen case. It would not establish
recovery improvement, controller optimality, general threshold validity,
stability, false-positive/false-negative rates, nominal handoff readiness,
retreat capability, formal safety, hardware validity, deployment readiness, or
authority for any other staged phase.
