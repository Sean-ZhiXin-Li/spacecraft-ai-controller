# Recoverability Platform Transition Report v1

## Status

Status: Week 8 platform-transition milestone

Completed: 2026-07-10

## 1. Executive Summary

Before Week 1, this repository already contained a scientifically valuable sequence of 2D spacecraft experiments. Phase34 established that post-cross synchronization can turn existing target-radius crossings into recoverable crossings. Phase36B, Phase36C, Phase37A, and Phase37B then showed that changing transfer families, radial commitment timing, or weak tangential shaping did not expand the protected crossing basin while preserving all known successes. The evidence was strong, but its fields, failure terms, acceptance rules, and decision records remained distributed across phase-specific artifacts.

Weeks 1 through 7 converted that evidence trail into a coherent research foundation. The work defined a benchmark, a mechanism-aware failure taxonomy, a normalized future result contract, a regression policy, a decision-evidence logging contract, a cross-embodiment architecture boundary, and a falsifiable Final Veto ablation design. Week 8 closes that documentation phase and records the implementation handoff.

The repository's long-term identity is:

> A cross-embodiment, recoverability-aware autonomous-control framework for physical systems operating under uncertainty, limited resources, and safety constraints.

The simplified 2D spacecraft simulator remains the primary implemented and evidence-supported testbed. Contact-rich robotic manipulation is the second major conceptual and practical research direction. Aerial robots, manipulators, legged or animal-like robots, ground robots, marine vehicles, planetary rovers, and other physical embodiments are future applications. Architectural relevance across embodiments is not experimental validation across domains. The framework does not propose one universal low-level controller; its shared object is the decision and assurance architecture around domain-specific controllers.

The platform-level principle is:

> Intermediate event success is not recoverable task completion.

This principle appears as follows:

- Spacecraft: target-radius crossing is not recoverable orbital insertion.
- Manipulation: first contact is not successful insertion.
- Perception: detection or pose estimation is not task completion.
- Learning: lower loss or higher reward is not rollout-level recoverability progress.
- Runtime assurance: issuing a veto is not evidence that a failure was avoided unless the paired counterfactual supports that conclusion.

The current state must be read in four distinct categories:

| Category | Current state |
| --- | --- |
| Implemented evidence | The 2D simulator, explicit controllers, Phase34/36/37 artifacts, and `scripts/check_phase_results.py` provide the protected scientific baseline. |
| Documented contract | The benchmark, taxonomy, result schema, regression policy, and decision-log schema define how future work must be measured and audited. |
| Experiment design | The minimal one-step overspeed Final Veto ablation is specified, paired, scoped, and ready to implement. |
| Future architecture | An integrated Decision Manager, trust assessment, runtime assurance implementation, normalized writers, and other embodiments remain unimplemented. |

The main implementation gap is therefore not another architectural document. It is the absence of a fresh, schema-compatible, paired experiment that exercises the first monitor while preserving known recoverability. The recommended next implementation milestone is exactly one minimal overspeed Final Veto ablation with one monitor, one hazard, one-step prediction, one fallback, eight protected preservation cases, and five diagnostic stress cases.

## 2. Starting Point Before Week 1

The repository entered Week 1 with a strong phase-driven experimental history and a clear central observation: first crossing is not insertion. Its historical artifacts preserved both positive and negative controller results, and the Phase34 through Phase37 sequence had narrowed the main scientific bottleneck to upstream crossing generation rather than post-cross stabilization.

That starting point was scientifically valuable, but insufficiently standardized for the next stage:

- Phase34/36/37 CSVs used real but phase-specific fields and labels.
- Crossing, recoverability, final simulator outcome, safety, and closest approach were not organized under one normalized result schema.
- There was no controlled failure-label taxonomy separating terminal mechanisms, precursors, diagnostics, and manual audit notes.
- There was no explicit policy defining accepted progress, diagnostic-only evidence, and regression.
- Known successful Phase34 recoverability behavior was protected by a historical guard, but not yet tied to general claim rules for future controllers or monitors.
- Runtime and evaluator decisions had no standardized evidence log.
- Final Veto and Runtime Assurance existed as architecture ideas, but no concrete, paired, falsifiable ablation had been declared.
- Long-term architecture ideas were not yet connected by a common cross-embodiment identity and explicit domain-validation boundaries.

This does not invalidate the earlier phase structure. The phase artifacts are the evidence from which the normalized contracts were derived. The documentation phase added comparability, auditability, and claim discipline without rewriting that history.

## 3. Protected Scientific Baseline

The protected scientific baseline is encoded by `scripts/check_phase_results.py` and supported by the historical artifact summaries. These facts are evidence constraints for future work, not values to regenerate or revise in this report.

| Phase or set | Protected evidence | Interpretation |
| --- | --- | --- |
| Phase34 `radius_priority` | 24 cases; 8 crossings; 8 recoverable crossings | Phase34 improved post-cross recoverability for crossing-producing cases. |
| Reduced Phase31-style baseline | 24 cases; 8 crossings; 0 recoverable crossings | Relative to the reduced baseline, Phase34 converted existing crossings into recoverable crossings. It did not expand crossing generation beyond 8/24. |
| Phase36B `baseline_phase34` | 24 cases; 8 crossings; 8 recoverable crossings; 0 overspeed; 0 instability | Baseline protected behavior was reproduced within the Phase36B comparison. |
| Phase36B `grazing_corridor` | 24 cases; 8 crossings; 8 recoverable crossings; 0 overspeed; 0 instability | The transfer-family change did not improve full-benchmark crossing or recoverability beyond Phase34. |
| Phase36B `redesigned_delayed_crossing` | 24 cases; 8 crossings; 8 recoverable crossings; 0 overspeed; 0 instability | The transfer-family change did not improve full-benchmark crossing or recoverability beyond Phase34. |
| Phase36B `spiral_approach` | 24 cases; 8 crossings; 8 recoverable crossings; 0 overspeed; 0 instability | The transfer-family change did not improve full-benchmark crossing or recoverability beyond Phase34. |
| Phase36C baseline non-crossing set | 16 cases; 8 `near_crossing`; 8 `over_conservative_transfer` | This is diagnostic geometry evidence. It does not establish new crossing or recoverability progress. |
| Phase37A | 144 rows; 0 new crossings on the baseline non-crossing cases; 0 overspeed; 0 instability | Radial timing variants did not solve upstream non-crossing cases. |
| Phase37A `delayed_commit_low` | 8/24 crossings; 8/24 recoverable crossings | This selected variant preserved known aggregate behavior but did not expand the crossing set. |
| Phase37A `delayed_commit_medium` | 8/24 crossings; 8/24 recoverable crossings | This selected variant preserved known aggregate behavior but did not expand the crossing set. |
| Phase37B selected weak cases | 4 cases; 0 crossings; 0 recoverable crossings | Weak tangential shaping did not solve the selected non-crossing cases. |
| Phase37B regression cases | 8 cases; 4 crossings; 4 recoverable crossings | Preservation of known successful behavior was incomplete. |
| Phase37B overall | 24 rows; 0 overspeed; 0 instability | The result is diagnostic subset evidence, not accepted progress. |

The protected interpretation is therefore narrow and stable. Phase34 improved post-cross recoverability but did not expand crossing generation. Phase36B did not improve the full-benchmark crossing or recoverability counts. Phase36C diagnosed non-crossing geometry. Phase37A preserved selected known behavior without creating new crossings. Phase37B produced a useful negative diagnostic while losing part of the regression subset.

## 4. Week 1-8 Milestone Summary

Week numbers identify roadmap milestones, not minimum calendar durations. Completion dates below are actual artifact completion dates.

| Milestone | Completion date | Main artifact | Question answered | New capability | Implementation status |
| --- | --- | --- | --- | --- | --- |
| Week 1: Recoverability Benchmark v1 | 2026-07-04 | `docs/benchmarks/recoverability_benchmark_v1.md` | What does the current evidence support, and what does it not support? | Benchmark definitions, protected evidence summary, scientific non-claims, and false-progress refusal rules | `documented_contract`, grounded in implemented evidence |
| Week 2: Failure Label Taxonomy v0 | 2026-07-05 | `docs/benchmarks/failure_label_taxonomy_v0.md` | What mechanism caused a rollout to fail? | Controlled terminal labels, precursor labels, diagnostic labels, manual-audit rules, and label priority | `documented_contract`; normalized label writer not implemented |
| Week 3: Result Schema v1 | 2026-07-08 | `docs/benchmarks/result_schema_v1.md` | What must a future rollout result record? | Event, recoverability, final outcome, safety, effort, subset, regression, and accepted-progress fields | `documented_contract`; no normalized writer or validator |
| Week 4: Recoverability Regression Policy v0 | 2026-07-08 | `docs/benchmarks/recoverability_regression_policy_v0.md` | What can count as progress, diagnostic evidence, or regression? | Known-success preservation, claim-specific requirements, safety and subset constraints, and a future gate design | `documented_contract`; only the historical guard is implemented |
| Week 5: Decision Log Schema v0 | 2026-07-09 | `docs/architecture/decision_log_schema_v0.md` | Why did the system or evaluator continue, retry, retreat, reject, accept, or veto? | Runtime and evaluator decision types, trust evidence, fallback representation, veto records, and CSV/JSONL concepts | `documented_contract`; no decision-log writer |
| Week 6: Cross-Embodiment Recoverability Framework | 2026-07-10 | `docs/research/roahm_contact_recoverability_notes.md` | Which decision and recoverability structures can be shared across physically embodied autonomous systems? | Cross-embodiment identity, shared autonomy loop, contact-rich manipulation direction, validation boundaries, and retry/retreat/re-observe/trust/veto abstractions | `design_only` architecture informed by spacecraft evidence |
| Week 7: Final Veto Ablation Plan v0 | 2026-07-10 | `docs/experiments/final_veto_ablation_plan_v0.md` | What is the smallest experiment that can test a pre-action veto without destroying known recoverability? | Simulator overspeed target, one-step prediction, zero-action fallback, separated preservation/stress sets, paired counterfactual design, and tradeoff metrics | `experiment_ready`; no monitor or runner implemented |
| Week 8: Platform Transition Report v1 | 2026-07-10 | `docs/reports/recoverability_platform_transition_report_v1.md` | What is now defined, measurable, auditable, protected, implementation-ready, or still conceptual? | Documentation-phase closure, readiness accounting, and one bounded implementation handoff | `documented_contract`; no implementation performed |

## 5. Documentation Dependency Graph

The logical dependency chain is:

```text
Recoverability Benchmark v1
  -> Failure Label Taxonomy v0
  -> Result Schema v1
  -> Recoverability Regression Policy v0
  -> Decision Log Schema v0
  -> Cross-Embodiment Recoverability Framework
  -> Final Veto Ablation Plan v0
```

Each layer constrains the next one:

| Layer skipped | Consequence |
| --- | --- |
| Recoverability Benchmark v1 | Progress has no stable benchmark definition, and crossing can be confused with recoverable completion. |
| Failure Label Taxonomy v0 | Failures collapse into outcome-only labels, obscuring mechanism, ambiguity, and audit needs. |
| Result Schema v1 | Event, recoverability, safety, final outcome, subset, and regression fields cannot be compared consistently. |
| Recoverability Regression Policy v0 | Proxy or subset gains can be presented as progress even when known Phase34 successes are destroyed. |
| Decision Log Schema v0 | Vetoes, fallbacks, runtime decisions, and evaluator accept/reject decisions lack reconstructable evidence. |
| Cross-Embodiment Recoverability Framework | General architectural relevance can be confused with empirical validation in drones, manipulation, legged systems, marine systems, or other domains. |
| Final Veto Ablation Plan v0 | Final Veto remains a philosophical architecture concept rather than a bounded, falsifiable simulator experiment. |

The dependency is conceptual rather than a claim that every contract is already implemented. In particular, the Final Veto plan depends on schema and decision-log concepts that still need minimal writers and validation code.

## 6. What Is Now Measurable

"Defined" and "measured" are not interchangeable. Historical fields demonstrate that some quantities are already emitted, while other quantities exist only as future schema contracts or paired-ablation definitions.

| Concept | Before this phase | Current measurement or contract | Current limitation |
| --- | --- | --- | --- |
| Target-radius crossing | Present in phase-specific artifacts | Historical CSVs and the protected guard measure crossing counts; Result Schema v1 standardizes `crossed_target_radius` and crossing timing | Future scripts do not yet use a common writer |
| Recoverable crossing | Present in Phase34/36/37 evidence | Historical `recoverable_crossing` values and protected counts are available; benchmark and schema define its separation from crossing | A normalized recovery-margin metric is not implemented |
| Final simulator success | Present under simulator-specific legacy fields | Result Schema v1 preserves `final_simulator_success` as simulator-defined | It is not mission success, hardware success, or necessarily equivalent to recoverable crossing |
| Closest approach | Present in diagnostic artifacts | Explicitly classified as a diagnostic metric | Improvement alone cannot establish crossing or recoverability progress |
| Overspeed | Present in current phase artifacts and checks | Historical boolean/count fields exist; taxonomy and Week 7 identify simulator-defined overspeed as a mechanism and ablation target | No pre-action monitor or fresh paired monitor results exist |
| Instability | Present in current phase artifacts and checks | Historical boolean/count fields exist and are required by the result contract | Mechanism details remain phase-specific in older scripts |
| Terminal failure mechanism | Inconsistently represented by phase-specific labels | Failure Label Taxonomy v0 defines a controlled `terminal_label` and priority rules | Historical rows were not migrated, and no normalized label writer exists |
| Diagnostic subset status | Expressed in phase-specific analyses | Result Schema v1 defines `is_full_benchmark`, `subset_id`, and representative-subset metadata | Existing artifacts do not uniformly populate these fields |
| Regression-set membership | Known operationally through selected historical cases | Result Schema v1 and the regression policy define explicit membership and known-success flags | No automated Result Schema regression gate exists |
| Accepted-as-progress status | Inferred manually from phase conclusions | Result Schema v1 defines `accepted_as_progress` and `acceptance_reason`; the policy defines eligibility | No current experiment writer or validator enforces the fields |
| Decision type | Not standardized | Decision Log Schema v0 defines runtime and evaluator decision enums | No runtime or evaluator decision logger exists |
| Decision reason | Not standardized | Decision Log Schema v0 defines controlled reason enums and evidence links | Reasons are not emitted by current controllers |
| Veto status | Architecture concept only | Decision Log Schema v0 defines allow, modify, switch, retreat, safe-mode, abort, blocked, and related states | No Final Veto implementation produces these records |
| Avoided failure | Not measured as a paired counterfactual | Week 7 defines it as monitor-off hazard with no corresponding monitor-on hazard in a complete pair | Requires fresh matched off/on runs; a veto alone is insufficient |
| Blocked success | Not measured | Week 7 defines loss of an off-arm recoverable crossing or simulator success after monitor intervention | Requires paired runner and monitor-specific extension fields |
| Unnecessary veto | Not measured | Week 7 defines veto activity in hazard-negative counterfactual conditions and requires event and rollout reporting | Requires decision logs and paired outcomes |
| False negative | Not measured | Week 7 defines a monitor-on overspeed with no preceding veto for the threshold-crossing action | Requires step-level monitor evidence |
| Fallback failure | Not measured | Week 7 defines a veto followed by realized overspeed | Requires fallback execution and outcome linking |
| Recoverability preservation | Historically visible in protected aggregate counts | Regression policy requires every known Phase34 recoverable case to retain crossing and recoverability; Week 7 fixes an eight-case preservation set | No fresh monitor-on preservation run exists yet |

## 7. Platform Readiness Matrix

The allowed readiness values are `implemented_and_evidence_supported`, `partially_implemented`, `documented_contract`, `experiment_ready`, `design_only`, and `deferred`.

| Component | Readiness | Basis | Key boundary |
| --- | --- | --- | --- |
| Spacecraft dynamics simulator | `implemented_and_evidence_supported` | Existing 2D rollouts and historical artifacts | Simplified planar simulation only |
| Phase34 post-cross controller | `implemented_and_evidence_supported` | 8/24 crossings and 8/24 recoverable crossings under protected evidence | Does not solve upstream crossing generation |
| Protected historical regression guard | `implemented_and_evidence_supported` | `scripts/check_phase_results.py` checks Phase34/36/37 facts | It does not validate Result Schema v1 or future monitor outputs |
| Recoverability Benchmark v1 | `documented_contract` | Week 1 benchmark grounded in protected evidence | No general benchmark runner was added |
| Failure Label Taxonomy v0 | `documented_contract` | Week 2 controlled labels and ambiguity rules | No normalized label assignment engine |
| Result Schema v1 | `documented_contract` | Week 3 field and representation definitions | No writer or automated validator |
| Recoverability regression policy | `documented_contract` | Week 4 claim and preservation rules | Proposed future gate is not implemented |
| Automated Result Schema validator | `deferred` | Required behavior is documented | No script exists yet |
| Decision logging | `design_only` | Decision Log Schema v0 defines the contract | No JSONL/CSV writer is connected to rollouts |
| Decision Manager | `design_only` | Architecture and decision enums exist | No runtime selection or authority component |
| Trust manager | `design_only` | Trust categories and flags are conceptual | No estimator, calibration, or authority policy |
| Final Veto ablation design | `experiment_ready` | Week 7 freezes one hazard, threshold, fallback, pair design, and acceptance logic | No fresh runs or monitor implementation |
| Final Veto implementation | `deferred` | Potential module boundary is specified | No `runtime_assurance` module exists |
| Paired monitor runner | `deferred` | Inputs, arms, and outputs are specified | No runner exists |
| Cross-embodiment architecture | `design_only` | Shared decision structure and domain boundaries are documented | Only spacecraft has current repository evidence |
| Contact-rich manipulation implementation | `deferred` | Contact concepts provide a second research direction | No manipulation code, benchmark, or data was added |
| Sensor-noise experiments | `deferred` | Identified as a small future test | Not designed or run in this phase |
| Delayed-observation experiments | `deferred` | Identified as a small future test | Not designed or run in this phase |
| Formal safety verification | `deferred` | Explicitly outside current evidence | No proof, invariant set, or verified monitor |
| Hardware validation | `deferred` | Explicitly outside current evidence | No hardware, flight, or sim-to-real evidence |

## 8. Current Architecture

The shared high-level architecture is:

```text
Observation or belief estimation
  -> Event detection
  -> Trust assessment
  -> Recoverability assessment
  -> Decision Manager
  -> Controller or planner selection
  -> Runtime Assurance or Final Veto
  -> Physical action
  -> Result, failure, and decision logging
```

The shared architecture does not imply shared dynamics, sensors, actuators, thresholds, recovery actions, or low-level controllers. Each physical embodiment requires its own models, evidence, benchmarks, and validation.

| Architecture layer | Current implemented status | Current documented interface | Evidence available | Missing implementation |
| --- | --- | --- | --- | --- |
| Observation or belief estimation | `partially_implemented`: simulator state is directly available | Week 5/6 list state, belief, uncertainty, and trust evidence concepts | Deterministic simulator states used by current controllers | Belief estimator, uncertainty propagation, sensor abstraction, and observation trust |
| Event detection | `partially_implemented`: crossing and simulator events are detected in phase scripts | Benchmark and Result Schema define crossing event fields | Historical crossing, CAPTURE, LOCK, and related phase outputs | Common event interface and normalized event writer |
| Trust assessment | `design_only` | Decision Log Schema and Week 6 define trust flags/categories | No calibrated trust evidence | Trust estimator, calibration method, and authority effects |
| Recoverability assessment | `partially_implemented`: historical scripts classify recoverable crossings | Benchmark and Result Schema define recoverability outcomes and optional margin/cost concepts | Phase34/36/37 recoverability counts | Stable online estimate, margin, cost, and horizon-aware assessment |
| Decision Manager | `design_only` | Decision types, reasons, scope, authority, fallback, and audit fields are documented | Evaluator decisions can be reconstructed manually from summaries | Integrated runtime decision component and policy |
| Controller or planner selection | `partially_implemented`: experiments select explicit controller families offline | Decision Log Schema can record selected controller and switch decisions | Historical controller-family comparisons | Runtime regime-based selection and switching authority |
| Runtime Assurance or Final Veto | `experiment_ready` design only | Week 7 defines one-step overspeed evaluation, allow/veto evidence, and one fallback | Existing overspeed fields and archived stress-case selection evidence | Monitor module, pre-action integration, fallback handling, and validator |
| Physical action | `implemented_and_evidence_supported` inside the 2D simulator | Week 7 proposes a monitor boundary around nominal action execution | Existing simulator rollouts | No real actuator, hardware, or cross-domain action interface |
| Result, failure, and decision logging | `partially_implemented`: phase-specific result CSVs exist | Result Schema v1, Failure Label Taxonomy v0, and Decision Log Schema v0 define normalized contracts | Historical CSVs and summaries | Common result writer, decision writer, schema extension, pair linker, and validation scripts |

The current spacecraft simulator does not implement this sequence as one integrated runtime autonomy system. The diagram is a target architecture constrained by existing evidence, not a description of a completed platform.

## 9. Evidence, Contract, And Implementation Boundaries

| Category | Meaning | Current examples |
| --- | --- | --- |
| Implemented Evidence | Code and artifacts that have actually run and support scoped claims | 2D simulator; explicit controllers; Phase34/36/37 CSV artifacts and summaries; protected counts; `scripts/check_phase_results.py` |
| Documented Contract | Stable definitions and required representations for future work | Recoverability Benchmark v1; Failure Label Taxonomy v0; Result Schema v1; Recoverability Regression Policy v0; Decision Log Schema v0 |
| Experiment Design | A predeclared, falsifiable test that has not yet produced fresh evidence | Final Veto overspeed ablation with paired monitor-off/on arms, preservation set, diagnostic stress set, fallback, and acceptance criteria |
| Conceptual Architecture | General structures whose interfaces or hypotheses are described but not integrated | Cross-embodiment framework; trust manager; Decision Manager; contact-domain extensions; future controller-selection authority |
| Prohibited Interpretation | Claims not supported by current implementation or evidence | Formal safety; verified Runtime Assurance; flight or hardware readiness; real spacecraft validation; sim-to-real transfer; cross-domain experimental validation |

Documentation can constrain future experiments, but it cannot substitute for those experiments. A schema field is not an implemented metric until a writer emits it correctly. A planned monitor is not Runtime Assurance evidence until paired runs exercise it. Cross-embodiment relevance is not validation in a second embodiment.

## 10. Scientific Advances Of The Documentation Phase

Weeks 1 through 8 produced research infrastructure and scientific-governance progress, not a new controller-performance breakthrough.

The repository can now:

- Separate event success from recoverable task completion as an explicit benchmark rule.
- Preserve negative and diagnostic results without relabeling proxy movement as accepted progress.
- Protect known Phase34 recoverable cases before broader claims are considered.
- Describe future rollout outputs in one language spanning events, recoverability, simulator outcome, safety, effort, labels, subsets, and regressions.
- Separate failure mechanisms from the evaluator or runtime decision taken in response.
- Record evaluator choices such as accept, reject, or mark diagnostic separately from runtime choices such as continue, retry, retreat, re-observe, safe mode, abort, switch, or veto.
- Test Final Veto through a falsifiable paired design that counts avoided failures and performance costs.
- State cross-embodiment architectural relevance while retaining domain-specific evidence boundaries.

These changes reduce ambiguity in future claims. They do not by themselves improve a controller, avoid a hazard, expand the crossing basin, or validate an integrated autonomy platform.

## 11. Remaining Scientific Questions

| Scientific question | Why it remains open | Evidence needed |
| --- | --- | --- |
| How can upstream crossing generation improve beyond 8/24 without losing the Phase34 recoverable cases? | Phase36B/37A/37B did not expand the protected crossing basin | Fresh full-benchmark and preservation-set rollouts with separate crossing/recoverability reporting |
| What state features best predict recoverability at crossing? | Current evidence reports outcomes but does not establish a stable predictor | Predeclared feature study with held-out validation and calibrated errors |
| Can a stable recovery margin be defined? | `minimum_recovery_margin` is a schema concept, not an accepted metric | Mathematical definition, units, controller/horizon assumptions, and empirical calibration |
| How should recovery cost and resource use be measured? | Historical effort fields are not normalized | Declared control-effort, fuel-proxy, time, and saturation definitions |
| Does the one-step overspeed monitor intervene early enough? | Week 7 fixes a one-step design but has no fresh runs | Paired stress results including false negatives and fallback failures |
| Is zero-action coast an adequate fallback? | It is a minimal design choice, not a proven safe action | Realized post-veto outcomes and comparison with monitor-off pairs |
| How many blocked successes and unnecessary vetoes occur? | Neither metric is currently logged | Pair-complete result and decision artifacts |
| Can decision logs be generated without excessive overhead? | The schema exists without a writer or runtime measurements | Logging implementation with size, latency, and completeness checks |
| How should trust be calibrated rather than assigned heuristically? | Trust categories are conceptual | Domain-specific uncertainty models and calibration experiments |
| How can contact-rich manipulation receive its own benchmark and schema without contaminating spacecraft evidence? | The analogy is structural, while physics and evidence differ | Separate domain extension, manipulation artifacts, and domain-specific validation |
| Which abstractions survive in drones, legged robots, marine systems, and other embodiments after domain-specific validation? | No such implementations or experiments exist here | Independent models, thresholds, benchmarks, and evidence for each embodiment |

## 12. Remaining Engineering Gaps

| Engineering gap | Current consequence | Minimal future resolution |
| --- | --- | --- |
| No automated Result Schema v1 validator | Schema compliance is manual | Add validation to the bounded ablation result checker |
| No normalized writer used by new experiments | Future outputs could drift into phase-specific formats | Emit Result Schema-compatible rows from the paired runner |
| No Decision Log Schema writer | Veto and allow evidence cannot yet be reconstructed mechanically | Add JSONL decision-event emission for the ablation |
| No `runtime_assurance` module | No pre-action monitor boundary exists | Add one independent overspeed monitor module |
| No paired monitor runner | Counterfactual completeness cannot be guaranteed | Add one off/on runner with stable pair IDs |
| No monitor-specific schema extension | Veto metrics lack declared machine-readable fields | Freeze a narrow extension in the experiment manifest |
| No fresh preservation reruns | Monitor preservation is untested | Run eight monitor-off/on Phase34 known-success cases |
| No fresh diagnostic stress reruns | Archived overspeed rows select cases but cannot prove a new effect | Run five declared stress pairs into a new directory |
| No veto decision logs | Intervention evidence is absent | Log allow/veto, threshold, prediction, fallback, and outcome links |
| No Final Veto result checker | Pairing, preservation, and tradeoff criteria are not enforced | Implement `scripts/check_final_veto_results.py` |
| No trust estimator | Trust cannot affect authority using calibrated evidence | Postpone until the first monitor ablation is complete |
| No Decision Manager | Runtime retry/retreat/re-observe/switch choices are not integrated | Postpone behind evidence from the minimal monitor |
| No cross-domain implementation | Architectural relevance is untested outside spacecraft | Require separate future domain projects, benchmarks, and validation |

## 13. First Implementation Recommendation

The single next primary implementation milestone should be:

> Implement the minimal overspeed Final Veto ablation defined in `docs/experiments/final_veto_ablation_plan_v0.md`.

Keep the milestone bounded to:

- one rule-based monitor;
- one simulator-defined hazard, executed-state `speed_ratio > 1.90`;
- one-step nominal-action prediction;
- one fallback, zero action `(0.0, 0.0)` for one step;
- eight known Phase34 recoverable preservation cases;
- five predeclared diagnostic overspeed stress cases;
- paired monitor-off and monitor-on runs under identical conditions;
- Result Schema-compatible outputs;
- Decision Log-compatible allow and veto records;
- one result validator;
- fresh artifacts in a new, isolated analysis directory.

This should precede a full Decision Manager, trust estimation, synthetic sensor expansion, new robotics domains, a major learning expansion, 3D spacecraft dynamics, or hardware integration. The minimal ablation tests whether the documented contracts can support one real intervention study while preserving known scientific behavior. Larger architecture work would multiply unresolved interfaces before the first assurance mechanism has produced measurable evidence.

## 14. Proposed Implementation Issue Breakdown

This is a local implementation backlog, not a set of created GitHub issues.

| Issue | Deliverable | Responsibilities and acceptance boundary |
| --- | --- | --- |
| A: Freeze Final Veto Experiment Manifest | A machine-readable manifest in a new artifact directory | Freeze hazard, `1.90` threshold, one-step horizon, zero-action fallback, eight preservation cases, five stress cases, acceptance criteria, artifact paths, and monitor ID before evaluation |
| B: Add Minimal Overspeed Monitor | Potential path: `runtime_assurance/final_veto_monitor.py` | Accept state and nominal action; predict one step with the same simulator dynamics; return allow or veto plus numerical evidence; remain independent of controller internals |
| C: Add Paired Ablation Runner | Potential path: `scripts/run_final_veto_ablation.py` | Run off/on pairs with identical conditions and stable IDs; write only fresh outputs; never overwrite Phase34/36/37 artifacts |
| D: Add Result And Decision Logging | `results.csv`, `paired_results.csv`, `decision_log.jsonl`, and manifest | Emit Result Schema-compatible rows, declared monitor extension fields, and Decision Log-compatible allow/veto events |
| E: Add Final Veto Result Validator | Potential path: `scripts/check_final_veto_results.py` | Run the historical guard; verify pair completeness, thresholds, and subset metadata; verify 8/8 crossing and recoverability preservation; report avoided failures, blocked successes, unnecessary vetoes, false negatives, and fallback failures |
| F: Run And Interpret The Ablation | Fresh artifacts, summary, comparison plot, and scoped conclusion | Preserve negative results; separate preservation and stress claims; do not infer formal safety or broader Runtime Assurance |

## 15. Work That Should Be Postponed

| Postponed work | Reason for postponement |
| --- | --- |
| Full Decision Manager | It should be informed by actual intervention and fallback evidence, not only enums and diagrams. |
| Multi-hazard Runtime Assurance | One hazard must first establish the monitor, pairing, logging, and validation path. |
| Formal safety proof | The first monitor threshold and fallback are experiment-design choices, not verified invariants. |
| Universal trust manager | Trust must be calibrated within a declared domain and observation model. |
| Large controller refactor | It risks changing the protected baseline before the monitor boundary is tested. |
| Large learning-model expansion | Learning metrics should not outrun rollout-level recoverability measurement. |
| ROS2 and hardware integration | Current evidence is simulation-only and the required software assurance path does not yet exist. |
| Drone implementation | Cross-embodiment relevance does not establish aerial-domain models or validation. |
| Legged-robot implementation | Terrain, contact, support, and actuator evidence require a separate domain effort. |
| Marine-vehicle implementation | Currents, communication, localization, and return-energy models are domain-specific. |
| Rover implementation | Mobility, terrain, and resource benchmarks require separate evidence. |
| Full plug-insertion integration into this repository | Manipulation is a second research direction, but should receive its own domain contract and evidence rather than displacing the spacecraft testbed. |
| 3D spacecraft conversion | It would expand dynamics and validation scope before the minimal assurance loop is tested. |
| Multi-agent systems | The single-agent recoverability and intervention contracts must first be exercised end to end. |

Postponement protects scientific focus. It keeps the next result attributable to one monitor, one hazard, one fallback, and declared case sets.

## 16. Scope Guardrails

Future implementation must retain the following guardrails:

- Preserve historical phase evidence and continue running `scripts/check_phase_results.py`.
- Create new artifacts rather than overwrite Phase34/36/37 CSVs, summaries, or plots.
- Separate the protected preservation set from the diagnostic stress set.
- Report target-radius crossing separately from recoverable crossing.
- Report simulator success separately from broader mission success.
- Separate architectural relevance across embodiments from experimental validation in any domain.
- Separate monitor intervention from a paired, evidenced avoided failure.
- Report blocked successes, unnecessary vetoes, false negatives, fallback failures, and performance cost.
- Keep subset claims scoped to their declared subsets.
- Do not allow documentation language to outrun implementation evidence.

## 17. Non-Claims

Weeks 1 through 8 do not establish:

- solved orbital insertion;
- solved upstream crossing generation;
- formal Runtime Assurance;
- formal safety;
- guaranteed overspeed prevention;
- flight readiness;
- real spacecraft validation;
- hardware readiness;
- sim-to-real transfer;
- drone-control validation;
- manipulator validation;
- legged-robot validation;
- ground-robot validation;
- marine autonomy validation;
- rover validation;
- universal cross-embodiment control performance;
- a completed integrated autonomy architecture.

Documentation alone does not prove controller or platform performance. The Week 7 plan is not an implemented monitor, and a future veto event will not be an avoided failure unless paired evidence supports that interpretation.

## 18. Documentation Phase Closure

Week 1 through Week 8 documentation deliverables are complete. The scientific baseline, failure taxonomy, result schema, regression policy, decision logging contract, cross-embodiment identity, and first veto ablation are now documented.

Additional documentation should be created only when required by implementation evidence. The next phase should prioritize code, fresh experiments, validation, and scoped interpretation. Future documents should accompany actual implementation milestones rather than extend the architecture indefinitely.

The closure principle is:

> Stop expanding the framework on paper until the first implementation-level ablation produces new evidence.

## 19. Final Platform Status Table

| Layer | Current status | Strongest evidence | Next action |
| --- | --- | --- | --- |
| Scientific baseline | `implemented_and_evidence_supported` | Protected Phase34/36/37 counts and preserved artifacts | Keep the historical guard mandatory |
| Benchmark | `documented_contract` | Recoverability Benchmark v1 tied to current evidence | Use it in fresh ablation manifests and outputs |
| Failure labels | `documented_contract` | Controlled taxonomy and ambiguity rules | Add normalized label assignment to the first new writer |
| Result schema | `documented_contract` | Result Schema v1 field/type and missing-value rules | Implement a narrow writer and validator for the ablation |
| Regression protection | `partially_implemented` | Historical guard plus documented known-success policy | Add future-artifact preservation checks |
| Decision evidence | `documented_contract` | Decision Log Schema v0 | Emit allow/veto JSONL events in the paired runner |
| Cross-embodiment architecture | `design_only` | Shared decision structure with explicit validation boundaries | Keep spacecraft primary; require separate domain evidence later |
| Final Veto | `experiment_ready` design | Predeclared overspeed ablation with paired sets and metrics | Implement the one-monitor ablation only |
| Trust | `design_only` | Defined trust concepts and log fields | Postpone calibration and authority logic |
| Decision Manager | `design_only` | Decision types, reasons, fallbacks, and architecture placement | Postpone until intervention evidence exists |
| Hardware and other embodiments | `deferred` | Architectural hypotheses only | Require domain-specific models, benchmarks, and validation in future work |

## 20. Final Conclusion

The repository has not completed the full autonomy architecture. It has completed the documentation and experiment-design foundation required to begin a disciplined implementation phase. Compared with the phase-driven starting point, future work now has stable definitions for recoverability, mechanism-aware failure labels, normalized result fields, known-success regression protection, decision evidence, cross-embodiment scope boundaries, and a falsifiable first veto experiment.

The strongest next step is the minimal overspeed Final Veto ablation. Any implementation must preserve all known Phase34 recoverable cases and report both simulated hazard benefits and performance costs, including blocked successes and unnecessary vetoes. Negative or unexercised monitor results must remain visible.

The project remains centered on recoverability-aware autonomous control across physical embodiments, with the simplified 2D spacecraft simulator as the current primary implemented and evidence-supported testbed. Contact-rich manipulation is the second major conceptual and practical direction. Every additional embodiment will require its own physics, thresholds, benchmarks, and validation before domain-specific claims are permitted.
