# July-August 2026 Research and Engineering Execution Plan

Status: two-month execution plan.

Time window: July 1, 2026 through August 31, 2026.

This plan covers the transition from a paper-driven repository to a long-term research platform. It is intentionally narrow. The goal is not to launch every future direction at once. The goal is to make the next layer of research measurable, reproducible, and aligned with the repository's core lesson:

```text
crossing is not insertion
```

For July and August, the strongest direction is:

```text
Recoverability Benchmark v1 + failure labeling + decision evidence logging
```

This gives the platform a better foundation before adding hardware, vision, robotics, or learning-heavy work.

## Guiding Constraints

The schedule assumes limited time and competing obligations:

- ROAHM weekly meetings.
- Internship tasks.
- Robotics learning.
- English improvement.
- Preparing for Grade 11.
- Continuing the spacecraft research platform without random feature additions.

The plan should feel sustainable. A good week should produce one clear artifact, not five half-finished directions.

## Two-Month Objective

By the end of August 2026, the repository should have:

- A cleaner Recoverability Benchmark v1 definition.
- Standard failure labels for spacecraft experiments.
- Better evaluator outputs that separate crossing, recoverability, closest approach, overspeed, instability, timeout, and resource metrics.
- A minimal decision-log schema, even if no full runtime decision system exists yet.
- One small runtime-assurance prototype or design stub that can support future final-veto experiments.
- A lightweight robotics/contact abstraction note influenced by the ROAHM internship, without copying lab code or turning the repository into a robotics repo.
- Documentation that clearly separates current evidence from future architecture.

The two-month goal is not to solve upstream crossing generation. It is to make the next experiments scientifically cleaner.

## Workstream Priorities

### Priority 1: Recoverability Benchmark v1

This is the central repository work.

Why first:

- It directly follows from the IAI2O paper.
- It protects the known Phase34 evidence.
- It prevents future experiments from confusing proxy progress with real progress.
- It gives learning, runtime assurance, and robotics-transfer ideas a common evaluation language.

Expected artifacts:

- `docs/benchmarks/recoverability_benchmark_v1.md`
- Updated evaluator output schema.
- Example benchmark result summary.
- Regression checklist for known `8/24` recoverable cases.

### Priority 2: Failure Labeling and Metrics

The repository needs structured failure reasons before more controller search.

Why second:

- Current failures must become diagnosable.
- Future controller changes should be judged by mechanism, not only aggregate success.
- Failure labels connect directly to recoverability, final veto, trust decay, and false-progress refusal.

Expected artifacts:

- Failure taxonomy document or section.
- CSV/log fields for termination reason and precursor reason.
- Small validation run showing labels are populated.

### Priority 3: Decision Evidence Logging

Do not implement a full autonomy executive yet. Start with logs that future runtime decision systems will need.

Why third:

- Decision architecture is not useful unless decisions are observable.
- Logging is lower risk than full runtime autonomy.
- It prepares for final veto and trust decay without overbuilding.

Expected artifacts:

- `docs/architecture/decision_log_schema_v0.md` or a section in the benchmark docs.
- Optional JSON/CSV field additions.
- One example rollout record with event state, failure label, and decision evidence fields.

### Priority 4: ROAHM Concept Transfer

The internship should influence concepts and evaluation, not inject unrelated code.

Why fourth:

- Plug insertion and spacecraft recoverability share approach, alignment, event/contact, stabilization, retry, and failure-mode structure.
- Robotics insight can improve abstractions without causing scope creep.

Expected artifacts:

- `docs/research/roahm_contact_recoverability_notes.md`
- A contact-state taxonomy draft.
- A short mapping between plug insertion and spacecraft event recoverability.

### Priority 5: Minimal Runtime Assurance Prototype

Only after metrics and labels exist, add a very small prototype.

Why fifth:

- A final-veto monitor without clean metrics cannot be evaluated.
- The prototype should be rule-based and transparent.

Expected artifacts:

- A simple monitor design or implementation stub.
- A small ablation plan: controller with monitor versus without monitor.
- No claim of safety guarantee.

## Weekly Plan

## Week 1: July 1-July 7

### Primary Research Goal

Freeze the post-paper scientific baseline.

The goal is to make sure the repository's next stage starts from the correct claim:

- Target-radius crossing is not recoverability.
- Phase34 solved post-cross synchronization for existing crossing-producing cases.
- Upstream crossing generation remains unresolved.
- Phase36B and Phase37 results are diagnostic, not new full-benchmark wins.

### Engineering Goal

Create a benchmark inventory and artifact map.

Work:

- List central scripts, controllers, analysis outputs, result CSVs, and figures.
- Identify which files define benchmark behavior and which are historical artifacts.
- Mark files that should not be modified casually.

### Learning Goal

Review only the control concepts needed for recoverability evaluation:

- State.
- Controller.
- Terminal set.
- Stability versus recoverability.
- Reachability versus recoverability.

Do not start a large textbook sequence.

### Documentation Goal

Draft `docs/benchmarks/recoverability_benchmark_v1.md`.

Minimum contents:

- Benchmark scope.
- Known baseline numbers.
- What counts as crossing.
- What counts as recoverable crossing.
- What counts as failure.
- What must be reported for every experiment.

### Deliverables

- Benchmark inventory section.
- First draft of Recoverability Benchmark v1.
- List of central files and fragile files.
- GitHub milestone created: `Recoverability Benchmark v1`.

### Expected Git Commits

- `docs: draft recoverability benchmark v1`
- `docs: add post-paper artifact inventory`

### Expected GitHub Issues

- `Define Recoverability Benchmark v1`
- `Inventory benchmark scripts and result artifacts`
- `Identify protected historical evidence files`

### Milestone Criteria

Week 1 is complete when a new contributor could read one document and understand the current evidence base without rerunning every phase.

## Week 2: July 8-July 14

### Primary Research Goal

Define the failure taxonomy.

The research question is:

```text
Why did a rollout fail?
```

not only:

```text
Did it fail?
```

### Engineering Goal

Add or standardize failure labels in evaluator outputs.

Start with a simple taxonomy:

- `success`
- `no_crossing`
- `crossing_unrecoverable`
- `recoverable_crossing_failed_late`
- `overspeed`
- `instability`
- `timeout`
- `resource_depletion`
- `invalid_simulation`
- `unknown`

The first version can be conservative. It is better to label uncertain cases as `unknown` than to invent false precision.

### Learning Goal

Study failure analysis in robotics and control at a practical level:

- Termination reasons.
- Precursor events.
- Confusion between symptom and root cause.
- Why failure labels must be auditable.

### Documentation Goal

Add a failure-labeling section to the benchmark document or create:

- `docs/benchmarks/failure_label_taxonomy_v0.md`

### Deliverables

- Failure taxonomy v0.
- Updated result schema proposal.
- One small run or manual audit showing how labels apply to known successes and failures.

### Expected Git Commits

- `docs: define failure label taxonomy v0`
- `eval: add failure label fields to benchmark outputs`

If code changes are too large for the week, split the second commit into a schema-only documentation commit and leave implementation for Week 3.

### Expected GitHub Issues

- `Add structured failure labels to evaluator output`
- `Validate failure labels on known Phase34 and Phase37 cases`

### Milestone Criteria

Week 2 is complete when every rollout can have at least one termination label and the label definitions are documented.

## Week 3: July 15-July 21

### Primary Research Goal

Separate event metrics from recovery metrics in a repeatable evaluator output.

The research question is:

```text
What happened before, during, and after the intermediate event?
```

### Engineering Goal

Add minimal recoverability metric fields.

Recommended fields:

- `crossed_target_radius`
- `crossing_time`
- `state_at_crossing`
- `recoverable_crossing`
- `recovery_time`
- `closest_approach`
- `max_speed`
- `overspeed`
- `instability`
- `termination_label`
- `control_effort`
- `fuel_proxy`
- `minimum_recovery_margin` if available

Do not overbuild a full metrics engine yet. Add fields that current scripts can actually populate.

### Learning Goal

Review basic state-estimation language:

- True state.
- Observation.
- Belief.
- Measurement noise.
- Confidence.

This prepares for future perception and robotics work without adding sensors yet.

### Documentation Goal

Create a short result-schema document:

- `docs/benchmarks/result_schema_v1.md`

### Deliverables

- Result schema v1.
- Updated evaluator or post-processing script.
- Example CSV or JSON output from one representative run.
- Short note explaining any fields not yet implemented.

### Expected Git Commits

- `docs: define recoverability result schema v1`
- `eval: record event and recovery metrics`
- `analysis: add sample benchmark output with new schema`

### Expected GitHub Issues

- `Separate crossing metrics from recoverability metrics`
- `Add control effort and fuel proxy fields`
- `Create sample result artifact for schema v1`

### Milestone Criteria

Week 3 is complete when a result file can show the difference between no crossing, crossing without recoverability, and recoverable crossing.

## Week 4: July 22-July 28

### Primary Research Goal

Protect known successes before searching for new ones.

The research question is:

```text
Can new experiments preserve the known 8/24 recoverable cases?
```

### Engineering Goal

Create a regression-safe benchmark command or checklist.

The regression set should include:

- Known Phase34 recoverable cases.
- Known non-crossing cases.
- At least one diagnostic subset from Phase37.

The goal is not yet to run huge sweeps. The goal is a reliable small gate.

### Learning Goal

Review basic experiment design:

- Train/test or tune/held-out separation.
- Regression sets.
- Threshold sensitivity.
- Why subset results cannot be reported as full-benchmark results.

### Documentation Goal

Add a regression policy:

- What must pass before a controller result can be considered progress.
- What counts as diagnostic only.
- How to report regression on known successes.

### Deliverables

- Regression checklist.
- Script or documented command for small benchmark gate.
- One baseline run using the gate.
- Updated GitHub milestone status.

### Expected Git Commits

- `docs: add regression policy for controller experiments`
- `scripts: add recoverability regression gate`
- `analysis: record baseline regression gate output`

### Expected GitHub Issues

- `Define known-success regression set`
- `Add lightweight benchmark regression gate`
- `Document diagnostic versus accepted progress criteria`

### Milestone Criteria

Week 4 is complete when any future controller change can be checked against known recoverable successes before being taken seriously.

## Week 5: July 29-August 4

### Primary Research Goal

Introduce decision evidence without building a full autonomy executive.

The research question is:

```text
What information would justify continue, retry, retreat, re-observe, safe mode, or abort?
```

### Engineering Goal

Define and optionally log a minimal decision-evidence schema.

Recommended fields:

- `mission_mode`
- `task_phase`
- `event_detected`
- `safety_level`
- `recoverability_level`
- `trust_flags`
- `selected_controller`
- `decision`
- `decision_reason`
- `fallback_available`
- `veto_status`

Do not build the full Decision Manager yet. Start by making decisions observable.

### Learning Goal

Review runtime assurance basics:

- Safety monitor.
- Veto.
- Fallback controller.
- Assurance case.
- Difference between empirical monitor and formal guarantee.

### Documentation Goal

Create:

- `docs/architecture/decision_log_schema_v0.md`

or append a concrete schema section to `docs/architecture/decision_and_runtime_assurance.md`.

### Deliverables

- Decision log schema v0.
- Example decision trace for a known rollout.
- Clear statement of what is not implemented yet.

### Expected Git Commits

- `docs: define decision log schema v0`
- `eval: add optional decision evidence fields`

### Expected GitHub Issues

- `Define decision evidence fields`
- `Add example decision trace for recoverability benchmark`

### Milestone Criteria

Week 5 is complete when the repository can describe why a rollout continued, failed, or would have triggered fallback, even if the runtime decision layer is not fully active.

## Week 6: August 5-August 11

### Primary Research Goal

Connect ROAHM internship concepts to the platform without mixing claims.

The research question is:

```text
What can plug insertion teach recoverability-aware autonomy?
```

The answer should be concepts, not copied code.

### Engineering Goal

No major code work this week unless internship workload is light.

Use this week to protect quality and avoid overload. If time allows, add a small contact-state taxonomy document.

### Learning Goal

Focus on the internship-relevant basics:

- End-effector pose.
- Contact state.
- Force direction.
- Pose estimation error.
- Imitation learning as a baseline, not proof of robustness.

Avoid deep ROS, advanced tactile sensing, or large vision models.

### Documentation Goal

Create:

- `docs/research/roahm_contact_recoverability_notes.md`

Suggested sections:

- Approach, alignment, contact, insertion, stabilization.
- Contact is not insertion.
- Pose estimate is not task success.
- Failure labels for plug insertion.
- What can transfer to spacecraft evaluation.
- What must not be claimed.

### Deliverables

- Contact recoverability note.
- Draft plug-insertion failure taxonomy.
- Mapping table between spacecraft and plug insertion phases.

### Expected Git Commits

- `docs: map plug insertion concepts to recoverability`
- `docs: draft contact failure taxonomy`

### Expected GitHub Issues

- `Draft contact-state taxonomy from ROAHM learning`
- `Map plug insertion evaluation concepts to spacecraft benchmark`

### Milestone Criteria

Week 6 is complete when the robotics internship has influenced the research vocabulary without changing the repository's scientific claims or adding unrelated dependencies.

## Week 7: August 12-August 18

### Primary Research Goal

Prototype a minimal final-veto experiment design.

The research question is:

```text
Can a simple monitor prevent known bad continuations without blocking known recoverable successes?
```

This week should be design-first. Implementation is optional if time is limited.

### Engineering Goal

Add a small rule-based runtime-assurance prototype or ablation plan.

Candidate veto rules:

- Veto if predicted speed exceeds overspeed threshold.
- Veto if instability trend persists.
- Veto if recovery margin falls below threshold.
- Veto if controller saturation persists for too many steps.

The fallback can be simple:

- Switch to recovery controller.
- Reduce action magnitude.
- Retreat or safe-hold in simulation.
- Terminate as safe abort if no fallback exists.

Do not claim formal safety.

### Learning Goal

Study only the runtime assurance concepts needed for the prototype:

- Monitor.
- Safety envelope.
- Fallback.
- False positive.
- False negative.
- Counterfactual comparison.

### Documentation Goal

Write a design note:

- `docs/experiments/final_veto_ablation_plan_v0.md`

### Deliverables

- Final-veto ablation plan.
- Optional monitor stub.
- Defined metrics: avoided failure, blocked success, recoverability preserved, cost added.
- Regression requirement against known Phase34 successes.

### Expected Git Commits

- `docs: add final veto ablation plan`
- `runtime: add simple veto monitor stub`

If implementation is too much, only make the documentation commit.

### Expected GitHub Issues

- `Design final-veto ablation experiment`
- `Add simple rule-based veto monitor`
- `Evaluate blocked successes versus avoided failures`

### Milestone Criteria

Week 7 is complete when final veto has a testable experiment design and cannot be confused with a vague safety slogan.

## Week 8: August 19-August 31

### Primary Research Goal

Consolidate the transition from paper artifact to research platform.

The research question is:

```text
What is now measurable that was only philosophical before?
```

### Engineering Goal

Run the small benchmark gate and produce an end-of-August snapshot.

The snapshot should include:

- Current baseline metrics.
- Failure-label distribution.
- Known-success regression result.
- Any schema changes.
- Any final-veto design or prototype status.
- Open issues for September.

### Learning Goal

Review and summarize what was actually useful from July-August:

- Recoverability evaluation.
- Failure labeling.
- State estimation basics.
- Runtime assurance basics.
- Plug insertion/contact concepts.

Do not start a new technical area in the final week.

### Documentation Goal

Create:

- `docs/reports/august_2026_platform_transition_report.md`

This should be short and factual:

- What changed.
- What evidence exists.
- What did not change.
- What remains unresolved.
- What should happen next.

### Deliverables

- End-of-August transition report.
- Updated benchmark documentation.
- Closed or updated GitHub issues.
- September issue backlog.
- Clean repository status or intentional uncommitted work list.

### Expected Git Commits

- `analysis: add august benchmark snapshot`
- `docs: add august 2026 platform transition report`
- `docs: update september research backlog`

### Expected GitHub Issues

- `Run August benchmark snapshot`
- `Write August platform transition report`
- `Prepare September recoverability research backlog`

### Milestone Criteria

Week 8 is complete when the repository has a clear August snapshot and the next stage can begin without rediscovering what happened during the transition period.

## Repository Work: What Should Happen First

Recommended order:

1. Recoverability Benchmark v1.
2. Failure label taxonomy.
3. Result schema and metrics logging.
4. Regression-safe benchmark gate.
5. Decision evidence logging.
6. ROAHM contact-recoverability concept note.
7. Final-veto ablation design.
8. End-of-August platform transition report.

This order is deliberate. The repository should not jump directly to runtime assurance, robotics, or hardware before the benchmark can measure the difference between event success and recoverable task progress.

## ROAHM Integration Rules

The internship should influence the project through abstractions and evaluation ideas.

Allowed transfer:

- Contact-state taxonomy.
- Failure-label design.
- Pose-estimation uncertainty vocabulary.
- Retry and withdrawal concepts.
- Force/contact ambiguity concepts.
- Imitation-learning evaluation caution.
- Phase structure: approach, alignment, contact, stabilization, completion.
- Recoverability after contact.

Not allowed without explicit permission:

- Copying lab code.
- Copying private datasets.
- Publishing lab-specific mechanisms.
- Mixing confidential internship results into public repository claims.
- Claiming spacecraft results based on robotics experiments.

The scientific bridge should be:

```text
spacecraft crossing : post-cross recoverability
plug contact        : post-contact recoverability
```

Both domains share the problem that an event can look successful while the physical system is not yet in a recoverable completion basin.

## Minimal Learning Plan

The learning plan should be small.

### July

Focus:

- Recoverability versus reachability.
- Failure labeling.
- Basic state estimation vocabulary.
- Experiment design and regression testing.

Output:

- Notes integrated into benchmark and schema docs.

### August

Focus:

- Runtime assurance basics.
- Contact-rich manipulation vocabulary.
- Pose uncertainty.
- Force/contact failure modes.

Output:

- Contact recoverability note.
- Final-veto ablation plan.

### Do Not Add a Giant Reading List

The learning goal is to support repository decisions, not to consume the summer. Each learning topic should end in a small artifact:

- A glossary.
- A taxonomy.
- A schema.
- A benchmark rule.
- A short experiment design.

## Things Not To Build Yet

### ROS2

Wait because the repository does not yet have stable autonomy interfaces, sensor models, or decision logs. Adding ROS2 now would create integration complexity before the core research abstractions are stable.

### Real Hardware

Wait because the current benchmark is still software-only and recoverability metrics need to mature. Hardware should enter after logging, failure labels, and safety decisions are explicit.

### FPGA or Chip-Level Work

Wait because edge constraints can be simulated first through latency, rate limits, memory budgets, and compute budgets. Hardware acceleration is premature.

### Multi-Agent Autonomy

Wait because single-agent recoverability and decision authority are not yet formalized in code. Multi-agent work would multiply ambiguity.

### 3D Orbital Dynamics

Wait because the current 2D benchmark still has unresolved upstream crossing generation. Moving to 3D now risks escaping the known scientific bottleneck.

### Foundation Models or Large Vision Models

Wait because perception is not yet part of the benchmark. The project first needs observation, belief, uncertainty, and task-success interfaces.

### Full Experiment Manager

Wait unless the existing scripts become impossible to manage. A small benchmark gate and result schema are enough for July-August.

### Large Controller Refactor

Wait unless needed for failure labels or metrics. Refactoring controllers before the benchmark schema is stable risks churn without scientific benefit.

### Learning Baseline Expansion

Wait because current PPO and imitation baselines were diagnostic. Cleaner learning baselines should come after Recoverability Benchmark v1 and regression gates are stable.

## Risks and Avoidance

| Risk | Why it matters | Avoidance strategy |
| --- | --- | --- |
| Scope creep | Hardware, robotics, runtime assurance, and learning can easily overload two months. | Limit active engineering to benchmark, labels, schema, and one small veto design. |
| Abandoning the benchmark | New ideas may distract from the central evidence base. | Make Recoverability Benchmark v1 the first milestone and gate later work through it. |
| Mixing robotics and spacecraft too early | Plug insertion and spacecraft are related but not the same system. | Transfer concepts and evaluation structures only; keep claims separate. |
| Over-engineering | A full architecture implementation would consume time without improving evidence. | Prefer schemas, logs, and small rule-based prototypes. |
| Documentation without implementation | Strategy docs can grow while code remains unchanged. | Each documentation week should produce at least one schema, checklist, command, or issue. |
| Implementation without scientific clarity | Code changes may add fields that do not answer research questions. | Tie every new field to crossing, recoverability, failure label, risk, or decision evidence. |
| Regression on known successes | Controller changes may lose the known `8/24` recoverable cases. | Add known-success regression gate before major controller search. |
| Overclaiming internship connection | Robotics experience may tempt broad claims. | Use phrases like "conceptual transfer" and "evaluation analogy"; do not claim validated transfer. |
| Learning overload | Too many topics can weaken both internship and repository progress. | Restrict learning to what produces July-August artifacts. |

## Final Deliverables by End of August

### Repository Changes

- `docs/benchmarks/recoverability_benchmark_v1.md`
- `docs/benchmarks/failure_label_taxonomy_v0.md` or equivalent section.
- `docs/benchmarks/result_schema_v1.md`
- `docs/architecture/decision_log_schema_v0.md` or equivalent section.
- `docs/research/roahm_contact_recoverability_notes.md`
- `docs/experiments/final_veto_ablation_plan_v0.md`
- `docs/reports/august_2026_platform_transition_report.md`
- Optional evaluator/script updates for failure labels and metric fields.

### Engineering Improvements

- Standardized failure labels.
- Event and recovery fields in benchmark output.
- Regression-safe benchmark gate for known successes.
- Control-effort or fuel-proxy logging if easy to add.
- Decision-evidence fields defined and optionally logged.
- Simple final-veto prototype or ablation design.

### Experiments Completed

Minimum:

- One baseline run or artifact using the new schema.
- One failure-label audit on representative cases.
- One regression-gate result on known successes.

Optional:

- One final-veto dry run or counterfactual analysis.
- One threshold-sensitivity mini-run.

### Research Questions Clarified

By August 31, the project should be able to answer:

- Which cases cross but are not recoverable?
- Which cases fail before crossing?
- Which cases preserve known recoverability?
- Which failure labels dominate the current benchmark?
- Which metrics are still missing?
- What would a final-veto monitor need to observe?
- How do plug-insertion contact phases map conceptually to spacecraft event phases?

### Internship Skills Acquired

The expected internship-related learning by end of August:

- Basic plug-insertion phase vocabulary.
- Contact-state and force-feedback intuition.
- Pose-estimation uncertainty vocabulary.
- Failure-case analysis mindset.
- Imitation-learning evaluation caution.
- Understanding of why contact or perception success is not task success.

These skills should influence documents and future benchmark design, not create premature robotics code.

## End-of-August Definition of Success

August is successful if the repository is more measurable, not if it is larger.

Success means:

- The core scientific claim remains accurate and conservative.
- Recoverability Benchmark v1 exists.
- Failure labels exist.
- Result schema separates event, recoverability, safety, and resource metrics.
- Known successes have a regression check.
- Decision evidence is defined.
- ROAHM learning has been translated into clean abstractions.
- Future hardware, vision, and runtime assurance work has a path but has not been prematurely implemented.

Failure means:

- Many new ideas are added but crossing/recoverability evaluation is not improved.
- Robotics concepts are mixed into spacecraft claims without abstraction.
- The repository gains architecture documents but no benchmark or logging progress.
- Known successes are not protected.
- Learning baselines or hardware directions are expanded before the benchmark is ready.

## September Handoff

If July-August succeeds, September can choose one of three focused tracks:

1. Upstream crossing-generation research using the improved benchmark gate.
2. Runtime assurance prototype with final-veto ablation.
3. Sensor/noise/latency benchmark preparation for future hardware-aware autonomy.

Only one should become the primary September engineering track. The others should remain documented backlog items.
