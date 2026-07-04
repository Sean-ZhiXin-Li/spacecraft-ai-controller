# Concept -> Metric -> Experiment Matrix

Status: working research design document.

This document converts recurring project principles into experimentally testable research constructs. The purpose is to prevent important ideas from remaining only slogans. Each concept below is treated as a candidate scientific variable: it must have an operational definition, observable variables, metrics, experimental protocol, failure cases, alternative explanations, and falsification criteria.

The concepts covered here are:

- Survival before optimization
- Trust decay
- Failure labeling
- Structured divergence
- Final veto
- Recoverability
- Refusal of false progress

The current repository is strongest on recoverability and false-progress separation because the orbital-control experiments already separate target-radius crossing from simulator-defined recoverable crossing. The other concepts are present as research philosophy and architecture direction, but they need explicit instrumentation before they can support strong empirical claims.

## Evaluation Assumptions

All experiments in this document should report:

- Benchmark version and scenario set.
- Controller or controller family.
- Random seed policy.
- Initial-condition distribution.
- Success, intermediate event, and failure definitions.
- Irreversible failure definition.
- Horizon and termination rules.
- Resource metrics such as fuel proxy, effort, time, or peak action.
- Whether evaluation uses full benchmark, held-out set, subset, or diagnostic cases.

No subset result should be reported as a full-benchmark result. No learning baseline should be described as successful unless it outperforms the relevant explicit-controller and diagnostic baselines under the same protocol.

## 1. Survival Before Optimization

Core idea: the system should preserve the possibility of safe continuation before maximizing local reward, speed, precision, or nominal objective value.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Physical autonomy often fails when optimization consumes safety or recovery margin. A controller that reaches an objective faster but leaves no abort, stabilization, or retry option may be worse than a slower controller that preserves mission continuation. |
| Operational definition | A policy satisfies survival-before-optimization if, when objective improvement conflicts with safety or recoverability margin, it selects actions that preserve survival constraints and recoverability over actions that improve nominal score but increase irreversible-failure risk. |
| Variables | State `x`; belief `b`; action `u`; nominal objective score; safety margin; recovery margin; irreversible failure risk; time-to-goal; fuel or effort budget; controller mode; disturbance level; threshold settings. |
| Metrics | Irreversible failure rate; safe-termination rate; recoverability-preservation rate; minimum recovery margin along trajectory; goal success conditioned on survival; reward achieved before failure; safety-margin violations; fuel remaining after recovery; number of cases where high-reward action was vetoed or avoided. |
| Experimental protocol | Construct paired controllers or action-selection modes: one nominal optimizer and one survival-prioritized controller. Evaluate both on identical initial conditions and disturbances. Include cases where direct objective pursuit causes overspeed, instability, collision, excessive force, or unrecoverable state. Compare final success, intermediate events, irreversible failure, recovery margin, and cost. |
| Expected observations | Survival-prioritized control may be slower, less aggressive, or less reward-maximizing in easy cases. It should reduce irreversible failure and preserve recoverability in hard cases. It should not simply stop forever unless safe abort or degraded mission is allowed by the task definition. |
| Failure cases | Controller becomes too conservative and never completes achievable tasks; survival rule blocks necessary transient maneuvers; survival metric is too loose and permits false safety; margin estimator is wrong; controller preserves safety but destroys mission recoverability through resource depletion. |
| Alternative explanations | Improved performance may come from lower control effort rather than survival logic; nominal optimizer may be poorly tuned; benchmark may favor conservative behavior; survival thresholds may encode hidden task knowledge; random seeds may under-sample aggressive-success cases. |
| Falsification criteria | If survival-prioritized control does not reduce irreversible failure or improve recoverability preservation under conflict scenarios, or if it only improves by refusing task progress in cases where progress is required, the concept is not supported by that experiment. |

### Minimal Experiment

Create a benchmark slice with known tension between aggressive target approach and post-event recoverability. Compare:

- Existing reference controller.
- Phase34-style recoverability-preserving controller.
- A deliberately aggressive crossing-seeking controller.

Report crossing count, recoverable crossing count, overspeed, instability, closest approach, control effort, and minimum recovery margin.

## 2. Trust Decay

Core idea: confidence in a controller, estimator, planner, or learned policy should decrease when evidence accumulates that its assumptions are failing.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Autonomous systems should not keep trusting a module that is producing inconsistent predictions, repeated near-failures, unstable actions, or uncalibrated confidence. Trust decay formalizes when the stack should reduce authority, request fallback, switch controller, increase sensing, or abort. |
| Operational definition | Trust decay is a stateful process `T_{t+1} = update(T_t, evidence_t)` where trust decreases after prediction error, safety-margin loss, recovery-margin loss, repeated failed attempts, estimator inconsistency, actuator saturation, or controller disagreement. Trust must affect behavior through mode switching, action limits, veto thresholds, or diagnostic flags. |
| Variables | Trust score per module; prediction error; estimator innovation; controller disagreement; action saturation; margin trend; failure labels; number of retries; sensor dropout; latency; confidence estimate; mode-switch threshold. |
| Metrics | Calibration error between trust and empirical success probability; time from first anomaly to trust reduction; false alarm rate; missed degradation rate; mode-switch precision and recall; success rate after trust-triggered fallback; irreversible failures after trust should have decayed; area under trust-risk curve. |
| Experimental protocol | Inject controlled degradations into controller, estimator, or sensor stream. Examples: biased pose estimate, delayed observation, noisy velocity, weakened tangential control, contact-force ambiguity, or learned-policy out-of-distribution input. Evaluate whether trust decays before failure and whether fallback improves outcomes. |
| Expected observations | Trust should remain high in nominal cases, decay under genuine degradation, and trigger appropriate fallback before irreversible failure. Trust decay should be gradual or evidence-based, not random oscillation. |
| Failure cases | Trust remains high until failure; trust decays in nominal cases and causes unnecessary abort; trust score is not connected to action; trust decay happens only after terminal failure; thresholds are hand-tuned to one benchmark and do not transfer; trust recovers too quickly after repeated failures. |
| Alternative explanations | Fallback controller may be better regardless of trust logic; degradation injection may be unrealistic; trust score may merely track an obvious variable such as speed; threshold tuning may overfit; performance may improve because action magnitudes are clipped, not because trust estimation is meaningful. |
| Falsification criteria | If trust score is not predictive of future failure or recoverability loss on held-out degradations, or if trust-triggered fallback does not improve safety/recoverability compared with fixed fallback schedules, the trust-decay mechanism is not empirically supported. |

### Current Measurability

Trust decay is not yet strongly measurable unless the repository logs module-level confidence, prediction errors, margin trends, and mode-switch reasons. It should be treated as an instrumentation target before being treated as an empirical result.

## 3. Failure Labeling

Core idea: failures should be classified by mechanism, not only counted.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Aggregate failure rate hides the reason a controller failed. Upstream crossing failure, post-cross unrecoverability, overspeed, instability, excessive force, sensor loss, and resource depletion require different fixes. Failure labeling turns negative results into actionable scientific evidence. |
| Operational definition | Each rollout terminates with a structured label, and optionally a time-indexed sequence of precursor labels. Labels must distinguish event failure, recoverability failure, safety failure, resource failure, controller instability, perception failure, and benchmark timeout. |
| Variables | Termination reason; first failure time; first precursor event; target crossing time; closest approach; speed at crossing; maximum speed; instability flag; controller mode; action saturation; sensor state; resource level; contact state for robotics tasks. |
| Metrics | Label distribution; label entropy; first-failure-time distribution; confusion rate between labels; percentage of failures with unknown label; agreement between automatic and manual labels; label-conditioned success rate; label-conditioned recovery margin; label transition matrix. |
| Experimental protocol | Define a failure taxonomy and implement automatic labeling in evaluator output. Re-run representative benchmark phases and compare label distributions across controllers. Add manual audit for a sample of rollouts to estimate label correctness. |
| Expected observations | Controller changes should shift the failure distribution in interpretable ways. For example, a post-cross synchronization controller should reduce post-cross unrecoverability without necessarily improving upstream crossing generation. |
| Failure cases | Labels are too broad to guide controller design; labels are mutually inconsistent; multiple failures occur but only the last one is recorded; label priority hides root cause; unknown label rate remains high; automatic labels disagree with trajectory inspection. |
| Alternative explanations | Label shifts may be caused by changed termination order rather than true mechanism change; thresholds may encode the desired result; manual audit may be biased; label names may imply causality when they only describe symptoms. |
| Falsification criteria | If labels do not predict which controller modifications help, do not match manual trajectory review, or cannot distinguish known mechanisms such as crossing failure versus post-cross unrecoverability, the taxonomy is not scientifically useful. |

### Minimal Failure Taxonomy

Recommended initial labels:

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

For future robotics:

- `no_contact`
- `bad_contact`
- `jam`
- `excessive_force`
- `lost_pose`
- `sensor_dropout`
- `withdraw_retry_success`
- `withdraw_retry_failed`

## 4. Structured Divergence

Core idea: multiple agents, controllers, planners, or hypotheses should differ in meaningful ways rather than sharing the same blind spots.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Redundant systems can fail together if they share the same model, objective, data distribution, or controller assumptions. Structured divergence aims to reduce correlated failure by maintaining diverse hypotheses, controllers, risk tolerances, or degradation strategies. |
| Operational definition | A controller set or decision ensemble exhibits structured divergence if its members differ along specified dimensions such as objective weighting, dynamics assumptions, recovery strategy, risk threshold, controller architecture, or failure response, and those differences reduce correlated failure without unacceptable cost. |
| Variables | Controller identity; objective weights; risk thresholds; model parameters; planning horizon; action sequence; predicted failure probability; recovery target; fallback mode; disagreement score; shared failure labels; ensemble selection rule. |
| Metrics | Pairwise action divergence; pairwise outcome divergence; correlated failure rate; ensemble oracle success rate; selected-controller success rate; disagreement-before-failure rate; diversity-cost tradeoff; improvement over homogeneous ensemble; coverage of failure labels across controller variants. |
| Experimental protocol | Build small controller families that vary one structured dimension at a time. Evaluate each controller and the ensemble on the same benchmark. Compare homogeneous variants, random perturbation variants, and structured variants. Measure whether at least one variant succeeds in cases where others fail, and whether a selection rule can identify it before failure. |
| Expected observations | Structured divergence should reduce correlated failures in some scenario families. It may increase computational cost and may not improve performance unless paired with a selector, monitor, or planner that can choose among divergent options. |
| Failure cases | Variants are nominally different but fail on the same cases; divergence increases unsafe actions; ensemble cannot choose the successful variant; diversity is random and not interpretable; all variants optimize the same false progress metric; computational cost is too high. |
| Alternative explanations | Improvement may come from simply testing more controllers; one controller may dominate all others; benchmark may be too small; variation may accidentally tune a parameter rather than provide meaningful divergence; oracle ensemble results may not be achievable online. |
| Falsification criteria | If structured variants do not reduce correlated failure compared with homogeneous or randomly perturbed variants under equal compute, or if no online selector can exploit the divergence, the concept is not supported as an engineering mechanism. |

### Current Measurability

This concept is partially measurable using existing transfer-family and controller-sweep experiments, but it needs explicit logging of controller differences, pairwise failure correlation, and ensemble-selection assumptions.

## 5. Final Veto

Core idea: a runtime assurance layer should be able to stop, override, switch, abort, or degrade a controller when continued execution threatens safety or recoverability.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Optimizers and learned policies can continue pursuing a local objective even after the task has become unsafe or unrecoverable. A final veto mechanism gives the system an explicit authority boundary: some actions should not be executed even if the nominal controller prefers them. |
| Operational definition | A final veto is a monitor `M(x, b, u, context)` that evaluates proposed actions and either allows, modifies, blocks, switches mode, commands abort, or triggers safe hold based on safety, recoverability, trust, or resource criteria. |
| Variables | Proposed action; veto decision; veto reason; safety margin; recovery margin; trust score; predicted next state; predicted failure probability; controller mode; fallback action; threshold parameters; false positive and false negative cases. |
| Metrics | Veto precision; veto recall for actions leading to failure; avoided irreversible failures; unnecessary veto rate; success rate after veto; time-to-veto before failure; recovery-margin preserved by veto; degradation-goal success; performance cost in nominal cases. |
| Experimental protocol | Run a nominal controller with and without a veto monitor. The monitor can use simple threshold rules first: overspeed risk, instability trend, margin below threshold, action saturation, or predicted unrecoverable state. Evaluate on nominal, disturbed, and adversarial initial conditions. |
| Expected observations | Veto should be rare in easy cases, active in dangerous cases, and reduce irreversible failures. It may reduce nominal success if thresholds are too conservative. The best result is not maximum veto count but improved survival and recoverability with controlled performance loss. |
| Failure cases | Veto triggers after failure is already irreversible; veto blocks all meaningful progress; fallback action is undefined or worse than nominal action; monitor relies on perfect state not available in realistic settings; thresholds overfit one benchmark; veto causes oscillatory mode switching. |
| Alternative explanations | The fallback controller may be responsible for improvement; action clipping may explain results; test cases may be biased toward veto-friendly failures; monitor may be using privileged simulator state unavailable in real systems. |
| Falsification criteria | If veto does not reduce irreversible failures or recoverability loss compared with a no-veto baseline, or if it achieves safety only by preventing legitimate task completion, the final-veto implementation fails the concept. |

### Minimal Experiment

Implement a non-learning veto monitor for existing orbital evaluation:

- Veto if predicted one-step speed exceeds threshold.
- Veto if recovery margin estimate falls below threshold.
- Veto if controller requests saturated action for too many consecutive steps.
- Switch to safe fallback or recovery controller.

Report both avoided failures and blocked successes.

## 6. Recoverability

Core idea: an intermediate event is useful only if the system can still reach an acceptable continuation, completion, retry, abort, or degraded mission state before irreversible failure.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | The repository's central scientific lesson is that target-radius crossing is not insertion. More generally, event success is not task success. Recoverability measures whether the system has entered a state from which useful physical continuation remains possible. |
| Operational definition | A state, belief, event, or rollout is recoverable relative to a task, controller, horizon, resource budget, and failure definition if the available controller can reach an acceptable recovery target before irreversible failure. |
| Variables | Event occurrence; event time; state at event; controller mode; recovery target; recovery horizon; recovery cost; recovery margin; failure set; irreversible failure set; safety constraints; belief uncertainty; resource remaining. |
| Metrics | Recoverable event count; recoverable-event rate; event-to-success conversion rate; event-to-failure rate; recovery time; recovery cost; recovery margin; robust recoverability under perturbation; probabilistic recoverability; controller-relative recoverability set size. |
| Experimental protocol | For each rollout, record intermediate event occurrence separately from recoverability and final success. At event time, evaluate whether the downstream controller reaches recovery target within horizon without irreversible failure. Run threshold sensitivity and randomized initial conditions. Preserve known recoverable cases while testing candidate changes on previous failures. |
| Expected observations | Some controllers may increase event count without increasing recoverable events. Post-event synchronization should improve event-to-success conversion for existing event-producing cases. Upstream controller changes should be judged by new recoverable events, not by closest approach alone. |
| Failure cases | Recovery target is poorly defined; recoverability is inferred from final success without event-time analysis; controller-relative claims are described as absolute; subset results are reported as full benchmark results; recovery horizon is chosen after seeing outcomes; cost and margin are ignored. |
| Alternative explanations | Apparent recoverability improvement may come from easier initial cases; threshold changes may reclassify outcomes; simulator termination order may bias labels; controller may exploit benchmark-specific definitions; final success may occur without meaningful post-event recovery. |
| Falsification criteria | If recoverability labels do not distinguish event-only cases from true continuation cases, or if a claimed recoverability improvement disappears under held-out initial conditions, threshold sensitivity, or regression checks on known successes, the recoverability claim is unsupported. |

### Current Measurability

Recoverability is currently the most mature concept in the repository. The next step is to make it more formal in evaluator outputs:

- Store event state.
- Store event-to-recovery trajectory segment.
- Store recovery target.
- Store recovery horizon.
- Store recovery cost.
- Store recovery margin.
- Store controller-relative label.

## 7. Refusal of False Progress

Core idea: the evaluation system should refuse to count intermediate metrics as scientific progress when they do not improve task-relevant recoverability, safety, robustness, or completion.

| Field | Experimental specification |
| --- | --- |
| Scientific motivation | Research can be misled by metrics that improve while the real task remains unsolved. Closest approach, target detection, contact, crossing, imitation loss, or reward may look positive while the system remains unrecoverable. Refusing false progress protects scientific claims. |
| Operational definition | A result is false progress if it improves a proxy metric without improving a pre-declared task-relevant metric such as recoverable event count, final success, safety, recovery margin, held-out performance, or resource-bounded completion. |
| Variables | Proxy metric; primary task metric; recoverability metric; safety metric; benchmark subset; held-out set; regression cases; thresholds; confidence interval; random seed; failure labels; artifact path. |
| Metrics | Proxy-primary divergence; number of proxy improvements rejected; regression count on known successes; held-out degradation; false-positive research claim rate; percentage of reports with predeclared primary metric; reproducibility of claimed improvement; benchmark coverage. |
| Experimental protocol | For every candidate controller or model, predeclare primary and secondary metrics. Mark a result as false progress if proxy metrics improve but primary metrics do not. Apply this to closest approach, crossing count, imitation loss, reward, and perception accuracy. Include regression checks on known successes. |
| Expected observations | Some experiments will show useful diagnostics but no accepted progress. This is not a failure of the protocol; it is the purpose. A diagnostic result can guide future work without changing the scientific claim. |
| Failure cases | Criteria are too strict and reject useful intermediate research; criteria are too vague and allow cherry-picking; primary metric changes after results; regression checks are incomplete; diagnostic artifacts are discarded instead of recorded. |
| Alternative explanations | A proxy may be necessary for future progress even if it does not immediately improve the primary metric; primary metric may be too coarse; benchmark may be too small to detect real improvement; result may improve a different legitimate objective not captured in the protocol. |
| Falsification criteria | If accepted "progress" repeatedly fails to improve held-out task outcomes, recoverability, safety, or robustness, the refusal protocol is too weak. If rejected proxy improvements later consistently predict true progress, the protocol is too strict or missing an intermediate validation metric. |

### Immediate Application

The current project should continue treating the following as diagnostic unless they improve the predeclared benchmark metrics:

- Lower imitation loss without better rollout performance.
- Higher reward without recoverable task success.
- Better closest approach without crossing or recoverability.
- More crossings without recoverable crossings.
- Better subset performance with regression on known full-benchmark successes.
- Perception accuracy without improved task recoverability.

## Concepts Not Yet Fully Measurable

| Concept | Current measurability | Missing instrumentation |
| --- | --- | --- |
| Survival before optimization | Partially measurable through failure rate, recoverability margin, overspeed, instability, and cost. | Explicit conflict cases where survival and objective optimization disagree; logged survival decision reasons; resource budgets. |
| Trust decay | Weakly measurable. Philosophy exists, but module-level trust is not yet a standard logged variable. | Trust scores; prediction errors; estimator innovations; confidence calibration; mode-switch triggers; fallback outcomes. |
| Failure labeling | Partially measurable if termination reasons exist, but taxonomy likely needs standardization. | Structured failure schema; precursor labels; first-failure time; manual audit protocol; label priority rules. |
| Structured divergence | Partially measurable through controller sweeps and transfer families. | Pairwise failure correlation; explicit diversity dimensions; ensemble selection protocol; equal-compute comparisons. |
| Final veto | Not fully measurable unless a monitor exists. | Proposed action logs; veto decisions; veto reasons; fallback actions; counterfactual no-veto rollouts. |
| Recoverability | Strongest current measurability. Existing evidence separates crossing from recoverable crossing. | Event-state logging; recovery margins; recovery cost; threshold sensitivity; randomized held-out benchmark. |
| Refusal of false progress | Conceptually measurable from reports and benchmark outcomes. | Predeclared primary metrics; result acceptance schema; explicit false-progress labels in experiment summaries. |

## Cross-Concept Experimental Design

The concepts should not be tested only in isolation. A useful experiment matrix should combine them:

| Experiment family | Concepts tested | Required comparison |
| --- | --- | --- |
| Aggressive versus conservative transfer | Survival before optimization, recoverability, false progress | More crossings versus more recoverable crossings and lower irreversible failure. |
| Post-event stabilization | Recoverability, failure labeling | Crossing-producing cases with and without post-cross synchronization. |
| Degraded sensor or state estimate | Trust decay, final veto, belief recoverability | Nominal controller versus trust-aware fallback under biased or delayed observations. |
| Controller family sweep | Structured divergence, false progress | Homogeneous variants versus structured variants with pairwise failure correlation. |
| Veto monitor ablation | Final veto, survival before optimization | Same controller with monitor on and off, including blocked successes and avoided failures. |
| Held-out randomized benchmark | Recoverability, refusal of false progress | Known successes preserved while testing new initial-condition families. |

## Reporting Template

Every experiment claiming progress on these concepts should include:

```text
Experiment ID:
Benchmark version:
Scenario set:
Controller(s):
Primary concept:
Primary metric:
Secondary metrics:
Proxy metrics:
Regression set:
Held-out set:
Failure labels:
Recoverability definition:
Irreversible failure definition:
Resource metrics:
Random seeds:
Artifact paths:
Accepted as progress: yes/no
Reason:
```

This reporting format is intentionally strict. It should make it difficult to accidentally convert diagnostics into claims.

## Ranking Matrix

Scores use a 1-5 scale.

- Maturity: how well the concept is already defined and connected to current repository evidence.
- Evidence: how much current data supports empirical statements about the concept.
- Implementation difficulty: 1 is easy, 5 is hard.
- Publication potential: likelihood of supporting a credible research contribution if implemented rigorously.

| Rank | Concept | Maturity | Evidence | Implementation difficulty | Publication potential | Rationale |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Recoverability | 5 | 4 | 3 | 5 | Central to current evidence. Already separates crossing from recoverable crossing. Needs stronger logging, held-out sets, margins, and threshold sensitivity. |
| 2 | Refusal of false progress | 4 | 4 | 2 | 4 | Strongly supported by experiment history: diagnostic learning and subset results did not become main claims. Easy to formalize through reporting rules. |
| 3 | Failure labeling | 4 | 3 | 2 | 4 | Natural next engineering step. Converts negative results into mechanisms and supports every other concept. |
| 4 | Survival before optimization | 3 | 2 | 3 | 4 | Philosophically central and experimentally testable, but needs explicit conflict scenarios and survival-margin metrics. |
| 5 | Final veto | 3 | 1 | 3 | 4 | Important for runtime assurance. Needs monitor implementation and counterfactual no-veto evaluation before evidence exists. |
| 6 | Structured divergence | 3 | 2 | 4 | 3 | Transfer-family experiments provide a starting point, but meaningful diversity and online selection are hard. |
| 7 | Trust decay | 2 | 1 | 4 | 4 | Potentially strong research direction, especially for perception and hardware-aware autonomy, but currently lacks instrumentation and calibrated trust variables. |

## Recommended Order of Work

1. Standardize failure labeling.
2. Add event-state, recovery-cost, and recovery-margin logging.
3. Add false-progress acceptance criteria to experiment summaries.
4. Create survival-versus-optimization conflict cases.
5. Implement a simple final-veto monitor.
6. Add structured controller-divergence analysis.
7. Add trust-decay instrumentation after prediction errors and confidence signals exist.

This order keeps the work grounded. Recoverability and false-progress refusal are already close to the current benchmark. Trust decay and structured divergence should not be overclaimed until the repository can measure them directly.
