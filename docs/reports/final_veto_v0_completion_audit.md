# Final Veto v0 Completion Audit

## Status

Post-experiment research-state audit.

Completed: 2026-07-16

The Final Veto Overspeed Ablation v0 experiment is complete and frozen. This document records the repository state after completion; it does not replace the formal artifacts or reinterpret them as evidence of formal safety.

## Audit Scope

This audit covers the implemented 2D spacecraft research path, the Week 1-8 contracts, the bounded Final Veto implementation, and the frozen formal evidence under `analysis/final_veto_ablation_v0/`.

The repository remains a cross-embodiment, recoverability-aware autonomous-control research framework with one primary implemented and evidence-supported testbed: the simplified 2D spacecraft simulator. Architectural relevance to other physical systems is not cross-domain validation.

## Current Milestone

The original scientific problem is:

> A target-radius crossing is not recoverable orbital insertion.

The repository reached the current milestone through four evidence stages:

| Stage | Question | Evidence contribution | Boundary |
| --- | --- | --- | --- |
| Phase34 | Can a crossing-producing trajectory be made recoverable after crossing? | The `radius_priority` post-cross controller preserved `8 / 24` crossings and converted all eight to recoverable crossings; the reduced Phase31-style reference had the same eight crossings and zero recoverable crossings. | It improved downstream recoverability but did not expand upstream crossing generation. |
| Phase36 | Do alternative transfer families expand crossing generation, and what geometry characterizes non-crossing cases? | Phase36B found the same `8 / 24` crossing and recoverable set in four families. Phase36C split 16 baseline non-crossing cases into eight `near_crossing` and eight `over_conservative_transfer` diagnostics. | Geometry evidence did not establish new crossing or recoverability progress. |
| Phase37 | Can radial commit timing or weak tangential shaping solve the diagnosed failures? | Phase37A created zero new crossings while selected delayed variants preserved the known `8 / 24`. Phase37B created `0 / 4` selected crossings and preserved only `4 / 8` regression recoverable cases. | These were negative or diagnostic results, not new controller progress. |
| Final Veto v0 | Can a one-step predictive veto reduce a declared simulated overspeed hazard without destroying known recoverability? | Eight preservation pairs retained crossing, recoverability, and simulator success. Five stress pairs changed monitor-off overspeed to monitor-on non-overspeed. | Hazard avoidance was demonstrated; stress-case task recovery was not. |

The current milestone is therefore not "solved insertion" or "completed Runtime Assurance." It is a completed, auditable simulator ablation showing that one bounded intervention layer can avoid one declared hazard while preserving the tested known-success cases.

## Completed Components

| Component | Current repository implementation | Completion meaning | Remaining boundary |
| --- | --- | --- | --- |
| Physics simulator | `envs/`, `simulator/`, and the exact Phase34/35 compatibility transition in `simulator/phase34_35_transition.py` | The repository can execute the protected 2D dynamics and reproduce the phase evidence. | The simulator is planar and simplified; multiple older dynamics paths still exist. |
| Explicit controllers | `controller/` and the Phase34-37 experiment scripts | Explicit controller behavior supports the protected phase trail and the Final Veto nominal arms. | No universal controller or solved upstream crossing generator exists. |
| Recoverability Benchmark v1 | `docs/benchmarks/recoverability_benchmark_v1.md` | Crossing, recoverability, simulator outcome, safety, subsets, and false progress have stable meanings. | It is a contract, not a general benchmark execution framework. |
| Failure Label Taxonomy v0 | `docs/benchmarks/failure_label_taxonomy_v0.md` | Controlled terminal labels, precursor labels, diagnostics, and ambiguity rules are defined. | Automatic taxonomy assignment is implemented only in bounded result paths, not every historical evaluator. |
| Result Schema v1 | `docs/benchmarks/result_schema_v1.md` plus Final Veto arm and pair records | The formal ablation emits schema-compatible identity, event, recoverability, safety, subset, regression, and monitor fields. | Historical CSVs remain intentionally unmigrated; no repository-wide writer exists. |
| Recoverability Regression Policy v0 | `docs/benchmarks/recoverability_regression_policy_v0.md`, `scripts/check_phase_results.py`, and Final Veto validation | Known Phase34 recoverable cases are explicitly protected before a positive monitor claim is accepted. | There is no universal regression service for arbitrary future experiments. |
| Decision Log Schema v0 | `docs/architecture/decision_log_schema_v0.md` | Runtime decisions and evaluator decisions have separate controlled evidence fields. | Most decision types remain unimplemented; Final Veto exercises only a bounded allow/veto profile. |
| Exact predictor boundary | `simulator/phase34_35_transition.py` | The monitor can receive a rollout-consistent one-step predictor without owning physics. | This boundary is specific to the protected Phase34/35 scalar transition semantics. |
| Final Veto monitor | `runtime_assurance/final_veto_monitor.py` | A one-step strict `speed_ratio > 1.90` rule returns allow or veto and substitutes the declared zero-action fallback. | It handles one hazard, one horizon, and one fallback; the fallback is not proven safe. |
| Paired runner | `scripts/run_final_veto_ablation.py` | Frozen cases expand into matched monitor-off/on jobs with stable identities and configuration hashes. | It is an experiment runner, not a Decision Manager or mission executive. |
| Artifact layer | `scripts/final_veto_artifacts.py` | Arm CSV, pair CSV, JSONL, summary, and plot publication use isolated, refusal-oriented paths. | It is specialized for the bounded ablation package. |
| Formal validator | `scripts/check_final_veto_results.py` | Structure, pair completeness, preservation, stress exercise, claim eligibility, and all five result artifacts are enforced separately. | Passing validation is not formal verification. |
| Compact decision logging | `scripts/final_veto_compact_log.py` and the formal `decision_log.jsonl` | Intervention segments and terminal events remain auditable without publishing 511,327 per-step records. | Compact logs do not provide a complete raw control trace. |
| Deterministic comparison rendering | `scripts/render_final_veto_comparison.py` | The formal comparison figure is derived from the frozen result CSVs without simulation. | The plot communicates recorded evidence; it does not add evidence. |

## Frozen Evidence

The complete formal evidence package is:

- `analysis/final_veto_ablation_v0/manifest.json`
- `analysis/final_veto_ablation_v0/results.csv`
- `analysis/final_veto_ablation_v0/paired_results.csv`
- `analysis/final_veto_ablation_v0/decision_log.jsonl`
- `analysis/final_veto_ablation_v0/summary.md`
- `analysis/final_veto_ablation_v0/comparison.png`

These artifacts are frozen. Interpretive documentation may cite them but must not rewrite them.

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `5E4387ED375855E0EB79D3B01599C421360EC33235000D4BE7FE076794CDA3A3` |
| `results.csv` | `1D41F5AF976D4C2408C6EB0D11540B78A5D4B971E749AAF04BB081B77A933A61` |
| `paired_results.csv` | `723A2E069D56CB762CA44FF25524414B7D044E80CCA5D2AB87B05ACAEF8FDD11` |
| `decision_log.jsonl` | `8926598EA30981076ADC5C851055B01480B55C425DBC310D6F1E45FE7019B72F` |
| `summary.md` | `84F5F0E4968DBE250FC6EB2CD23C7C63BBB2573496A6A20609C1D99F26A8F979` |
| `comparison.png` | `E73FED74C6250A194FB70A1D5BA37D1DD5CB86635CC8A2BA301467CEC4DF736B` |

## Frozen Result Summary

| Evidence group | Monitor-off | Monitor-on | Supported interpretation |
| --- | ---: | ---: | --- |
| Preservation crossing | `8 / 8` | `8 / 8` | Known crossing behavior was preserved. |
| Preservation recoverable crossing | `8 / 8` | `8 / 8` | All protected known-recoverable cases remained recoverable. |
| Preservation simulator success | `8 / 8` | `8 / 8` | Simulator-defined success was preserved on the protected set. |
| Stress overspeed | `5 / 5` | `0 / 5` | The declared simulated overspeed hazard was avoided in five complete pairs. |
| Stress task recovery | `0 / 5` | `0 / 5` | The intervention did not recover the task. |

The monitor was evaluated 511,327 times and vetoed 499,877 nominal proposals. There were zero recorded false negatives and zero recorded fallback failures. The aggregate intervention rate was approximately `0.9776072846`, so intervention burden is a central result rather than a footnote.

All five diagnostic stress pairs changed termination behavior from `overspeed` to `max_steps`. In controlled terminal-label terms, they changed from `overspeed` to `no_crossing`. This is hazard avoidance through failure-mode substitution, not task completion.

## Evidence Boundaries

### Implemented Evidence

- A simplified 2D spacecraft simulator and explicit controller trail.
- Protected Phase34, Phase36, and Phase37 historical artifacts and aggregate guard.
- An exact one-step predictor boundary for the Phase34/35 rollout semantics.
- A rule-based one-step overspeed monitor with an injected predictor.
- A paired monitor-off/on experiment over eight preservation and five diagnostic stress cases.
- Schema-compatible arm records, pair records, compact decision evidence, summary, and comparison plot.
- Formal structural, pairing, preservation, stress-exercise, and artifact-completeness validation.

### Documented Contracts With Bounded Implementation

- Recoverability Benchmark v1.
- Failure Label Taxonomy v0.
- Result Schema v1.
- Recoverability Regression Policy v0.
- Decision Log Schema v0.

The Final Veto experiment uses narrow portions of these contracts. Their existence does not mean every repository experiment has been migrated to them.

### Not Implemented

- Formal or verified Runtime Assurance.
- A general Decision Manager.
- A recovery manager or recovery-controller selector.
- Calibrated trust estimation.
- Recovery margin and recovery cost estimators.
- Multi-horizon or probabilistic risk prediction.
- Hardware validation, flight validation, or sim-to-real transfer.
- Contact-rich manipulation, drone, legged, ground, marine, or rover implementations.
- Universal cross-embodiment controller performance.

## Scientific State After Final Veto v0

The experiment answers one bounded question: a one-step predictive rule can prevent the tested next-state overspeed outcomes while preserving the eight tested known-recoverable cases.

It also exposes the next architecture gap. Once the monitor rejects the nominal action, the system has only a zero-action fallback and no mechanism for selecting a recovery strategy. The resulting stress trajectories avoid overspeed but continue until the horizon without crossing.

The next research problem is therefore not whether a veto can stop a declared bad transition. It is how an autonomous system should choose among continuing, adjusting, recovering, retreating, entering safe mode, or terminating after risk is detected.

## Non-Claims

Final Veto v0 does not establish:

- solved orbital insertion;
- solved upstream crossing generation;
- stress-case task recovery;
- a proven-safe fallback;
- formal safety or verified Runtime Assurance;
- flight, hardware, or deployment readiness;
- real-spacecraft validation;
- cross-domain experimental validation;
- a completed integrated autonomy architecture.

## Completion Statement

Final Veto Overspeed Ablation v0 is complete and frozen. The repository now has one auditable intervention experiment, but the result redirects the research question from veto-only hazard avoidance to recovery-aware decision and controller selection.
