# Final Veto v0 Scientific Interpretation

## Status

Post-experiment analysis document.

Completed: 2026-07-16

This report interprets the frozen Final Veto Overspeed Ablation v0 evidence. It is not a replacement for `analysis/final_veto_ablation_v0/summary.md`, does not alter the formal artifacts, and does not add experimental evidence.

## Research Question

Can a lightweight one-step predictive veto mechanism reduce a declared unsafe simulated transition while preserving known recoverable behavior?

The experiment makes that question falsifiable through two separate sets:

- eight known Phase34 recoverable preservation cases;
- five Phase35-derived diagnostic overspeed stress cases.

Each case has a matched `monitor_off` and `monitor_on` arm. The monitor uses the exact injected one-step rollout predictor, a strict predicted `speed_ratio > 1.90` trigger, and one step of zero action as its declared fallback.

## Hypothesis

A bounded veto mechanism can:

- reject nominal actions predicted to cross a specific simulated hazard threshold;
- reduce realized instances of that declared hazard;
- preserve previously recoverable trajectories when the monitor does not need to intervene.

The same mechanism is not expected, by itself, to:

- generate a new recoverable trajectory;
- select among recovery maneuvers;
- restore task progress after intervention;
- prove that its fallback is safe;
- establish formal Runtime Assurance.

## Results

### Preservation Set

| Metric | Monitor-off | Monitor-on |
| --- | ---: | ---: |
| Target-radius crossing | `8 / 8` | `8 / 8` |
| Recoverable crossing | `8 / 8` | `8 / 8` |
| Simulator-defined success | `8 / 8` | `8 / 8` |
| Overspeed | `0 / 8` | `0 / 8` |
| Invalid simulation | `0 / 8` | `0 / 8` |
| Blocked success | n/a | `0 / 8` |
| Unnecessary veto | n/a | `0 / 8` |

The monitor-on preservation arms performed 11,327 evaluations, allowed every proposal, and applied zero vetoes.

#### Why `8 / 8` Matters

The preservation set is the minimum regression obligation inherited from the Phase34 evidence. A monitor that reduced stress-set overspeed while destroying these cases would show a safety-performance tradeoff, not clean progress.

Preserving all eight cases establishes that the tested monitor did not damage the known Phase34 recoverable behavior under the matched formal runs. It does not establish preservation outside those eight cases, under other thresholds, or under other controller and dynamics regimes.

### Diagnostic Stress Set

| Metric | Monitor-off | Monitor-on |
| --- | ---: | ---: |
| Overspeed | `5 / 5` | `0 / 5` |
| Target-radius crossing | `0 / 5` | `0 / 5` |
| Recoverable crossing | `0 / 5` | `0 / 5` |
| Simulator-defined success | `0 / 5` | `0 / 5` |
| Invalid simulation | `0 / 5` | `0 / 5` |
| Task recovered after hazard avoidance | n/a | `0 / 5` |

All five complete stress pairs satisfy the declared paired definition of an avoided failure:

1. the matched monitor-off arm reached overspeed;
2. the monitor-on arm did not reach overspeed;
3. the monitor-on arm was not invalid;
4. pair identity and configuration matched.

Hazard avoidance: **YES**.

Task recovery: **NO**.

### Failure-Mode Transition

Every diagnostic stress pair changed its raw termination outcome from:

```text
overspeed -> max_steps
```

The controlled terminal label changed from:

```text
overspeed -> no_crossing
```

The monitor prevented the declared overspeed outcome, but the zero-action fallback did not produce a target-radius crossing, recoverable crossing, or simulator success. The experiment therefore substituted a horizon-limited non-crossing outcome for an overspeed failure.

This is scientifically useful hazard evidence. It is not successful recovery.

## Intervention Burden

| Measure | Value |
| --- | ---: |
| Monitor evaluations | `511327` |
| Allows | `11450` |
| Vetoes | `499877` |
| Zero-action fallback executions | `499877` |
| Overall intervention rate | `0.9776072845752327` |
| False negatives | `0` |
| Fallback failures | `0` |

The stress arms accounted for 500,000 evaluations and 499,877 vetoes, an intervention rate of `0.999754`. The monitor was not a literal veto-everything rule because it allowed all preservation proposals and 123 stress proposals. Nevertheless, its near-total stress intervention is a major limitation.

Zero recorded false negatives means that no allowed monitor-on proposal in these runs produced a realized next-state overspeed under the declared rule. Zero recorded fallback failures means that no executed zero-action fallback step itself produced realized overspeed. Neither count proves those events are impossible outside the tested runs.

The large positive step-count changes on the five stress pairs show the performance cost directly: the monitor avoided early overspeed but kept each trajectory active until the 100,000-step horizon. Hazard reduction and useful autonomous behavior must therefore remain separate metrics.

## Key Insight

> A veto layer is not a recovery policy.

The implemented control path is:

```text
state
  -> exact one-step prediction
  -> strict overspeed comparison
  -> allow nominal action or veto
  -> one-step zero-action fallback
```

This path answers whether a proposed action should be refused under one declared threshold. It does not answer what action should replace it to recover task progress.

A recovery-aware architecture requires a larger decision path:

```text
state or belief
  -> prediction
  -> risk classification
  -> recoverability assessment
  -> decision
       -> continue nominal control
       -> adjust action
       -> select recovery controller
       -> retreat
       -> enter safe mode
       -> terminate
  -> observe outcome and update evidence
```

The shared architecture should select among domain-specific controllers; it should not attempt to impose one universal low-level controller on every embodiment.

## What The Experiment Supports

The frozen evidence supports the following scoped statements:

- A rule-based one-step monitor reduced the declared simulated overspeed hazard from `5 / 5` to `0 / 5` on the predeclared diagnostic stress set.
- All eight protected Phase34 recoverable cases retained crossing, recoverable crossing, and simulator-defined success under monitor-on.
- The paired design reports intervention burden, blocked success, unnecessary veto, false negatives, and fallback failures rather than treating veto count as proof of safety.
- The repository can produce and validate a complete simulator-level intervention evidence package.

## What The Experiment Does Not Support

- The monitor recovered any diagnostic stress task.
- The zero-action fallback is generally safe or useful.
- One-step prediction is sufficient for all hazards.
- The `1.90` threshold is a formally derived invariant.
- Zero false negatives will persist on untested cases.
- The monitor constitutes verified Runtime Assurance.
- The intervention architecture is ready for hardware or deployment.

## Next Scientific Question

The next question is:

> After a nominal action is rejected, what evidence should select continue, adjust, recover, retreat, safe mode, or termination while preserving recoverability and controlling resource cost?

Answering that question requires recovery actions, recovery-margin and recovery-cost measurements, explicit decision logic, and experiments that distinguish safe refusal from restored task progress.

## Avoiding Overclaiming

Final Veto v0 provides no:

- formal safety guarantee;
- verified Runtime Assurance claim;
- guaranteed overspeed avoidance claim;
- real-spacecraft or flight validation;
- hardware validation;
- deployment-readiness claim;
- sim-to-real evidence;
- drone, manipulation, legged, ground, marine, or rover validation;
- universal robotics or cross-embodiment validation.

The result is a bounded simulator experiment whose strongest contribution is the separation of hazard avoidance, recoverability preservation, task recovery, and intervention cost.
