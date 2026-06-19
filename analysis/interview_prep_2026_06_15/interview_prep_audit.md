# Interview Prep Audit - June 15, 2026

Scope: repository evidence plus live GitHub issue search for `Sean-ZhiXin-Li/spacecraft-ai-controller`. This is not a paper draft. It is a technical defense guide.

## Evidence Base Read

Primary evidence inspected:

- `README.md`
- `README_reproduce.md`
- `docs/benchmark_contract.md`
- `docs/research_direction.md`
- `analysis/artifact_manifest.md`
- `analysis/full_goal_mode_audit/*`
- `docs/research_workspace/outputs/*`
- `docs/research_workspace/outputs/paper_assets/*`
- `analysis/phase31_global_transfer_solver/summary.md`
- `analysis/phase32_direct_optimal_control/summary.md`
- `analysis/phase33_optimal_structure_extraction/phase33_summary.md`
- `analysis/phase33_optimal_structure_extraction/structure_decomposition.md`
- `analysis/phase34_post_cross_sync/summary.md`
- `analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md`
- `analysis/phase35_crossing_basin_expansion/summary.md`
- `analysis/phase36b_transfer_family_benchmark/summary.md`
- `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`
- `analysis/phase37a_radial_commit_timing/phase37a_summary.md`
- `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`
- PPO/IL result files: `analysis/ppo_transfer_results.md`, `analysis/phase_conditioned_il_result.md`, `analysis/soft_phase_conditioned_il_result.md`, `analysis/learned_phase_stabilized_result.md`, `analysis/minimal_il/minimal_il_summary.json`
- Historical context: `analysis/phase76_soft_hybrid/phase76_summary.md`, `analysis/phase8_multiregime/phase8_summary.md`
- Code: `simulator/physics.py`, `envs/orbit_env.py`, `scripts/explicit_controller_phase34_post_cross_sync.py`, `scripts/explicit_controller_phase37a_radial_commit_timing.py`, `scripts/explicit_controller_phase37b_weak_tangential_subset.py`
- GitHub Issues: open issues #3, #4, #5, #6, #7 from connector search.

Verification run:

- `python scripts/check_phase_results.py` passed for Phase34, Phase36B, Phase36C, Phase37A, Phase37B.
- `python -m pytest -q Tests/test_env_smoke.py Tests/test_quickrun_smoke.py` could not run because default Python, `orbittools`, and `spacecraft` environments all lacked `pytest`.

## Task 1 - Repository Deep Audit

### A. Strong Research Contributions

1. Separating target-radius crossing from recoverable insertion is the strongest contribution.
   Evidence: `README.md`, `docs/benchmark_contract.md`, `analysis/phase31_global_transfer_solver/summary.md`, `analysis/phase34_post_cross_sync/summary.md`.
   Why: Phase31-style reduced reference had `8 / 24` crossings but `0 / 24` recoverable crossings, while Phase34 preserved `8 / 24` crossings and reached `8 / 24` recoverable crossings.

2. Phase33 to Phase34 is a coherent scientific chain.
   Evidence: `analysis/phase33_optimal_structure_extraction/structure_decomposition.md` says first crossing step `81`, best recoverability step `512`, crossing-state distance `2.313443`, best distance `0.000470`; `analysis/phase34_post_cross_sync/summary.md` implements post-cross synchronization.
   Why: This is not random tuning. It is observation, mechanism extraction, controller modification, benchmark test.

3. Negative results are preserved and interpreted.
   Evidence: Phase35, Phase36B/C, Phase37A/B summaries; `analysis/artifact_manifest.md`.
   Why: The project does not hide failure. It uses failure to narrow the next research question.

4. Learning claims are honest.
   Evidence: `analysis/ppo_transfer_results.md`, `analysis/phase_conditioned_il_result.md`, `analysis/soft_phase_conditioned_il_result.md`, `analysis/minimal_il/minimal_il_summary.json`.
   Why: PPO/BC/IL are reported as failing key rollout behavior, not as solved AI autonomy.

### B. Strong Engineering Work

1. Artifact trail is unusually organized for a high-school independent project.
   Evidence: `analysis/artifact_manifest.md`, `docs/research_workspace/outputs/paper_assets/unified_main_results_table.md`, `metric_definition_table.md`.
   Why: Claims are mapped to source files rather than only narrative.

2. Regression guard exists and passed.
   Evidence: `scripts/check_phase_results.py` and the successful local run.
   Why: It protects exact aggregate claims for Phase34, Phase36B/C, Phase37A/B.

3. Metrics are more specific than a single success flag.
   Evidence: `docs/benchmark_contract.md`, Phase34/36/37 CSV schemas.
   Why: Crossing, recoverable crossing, overspeed, instability, sync error, and closest approach are separated.

4. GitHub Issues show engineering maturity.
   Evidence: live issues #3 Phase37B regression guard, #4 benchmark manifest, #5 CI smoke verification, #6 Phase37B postmortem, #7 Phase38 search space.
   Why: The open issues match real weaknesses rather than vanity tasks.

### C. Strong Scientific Thinking

1. The project moved from "make it succeed" to "define what success means."
   Evidence: Phase8 broad failure, Phase31 crossing/recoverability gap, `docs/benchmark_contract.md`.
   Why: This is research maturity.

2. You distinguish proxy metrics from real events.
   Evidence: Phase36C and Phase37B: closest approach and crossing potential can improve without crossings.
   Why: A PhD student will respect that you do not count proxy improvement as success.

3. You use regression preservation as a scientific constraint.
   Evidence: Phase37B preserved only `4 / 8` regression crossings under weak tangential shaping.
   Why: A new method must not damage known good cases.

### D. Weak Assumptions

1. The recoverability basin is hand-defined.
   Evidence: `RECOVERABLE_R_RATIO`, `RECOVERABLE_VR_RATIO`, `RECOVERABLE_VT_RATIO` used by Phase34 scripts via Phase21 imports.
   Risk: Yizhou may ask why those thresholds are physically meaningful. You should answer: "They are simulator-defined diagnostic thresholds, not physical proof."

2. The 24-case benchmark is structured and small.
   Evidence: `docs/benchmark_contract.md`.
   Risk: It supports controlled comparison, not broad generalization.

3. Phase32 direct shooting is not a rigorous optimal-control proof.
   Evidence: `analysis/phase32_direct_optimal_control/summary.md` says CasADi unavailable and SciPy direct shooting fallback used.
   Risk: Do not say "optimal control solved it."

4. 2D point-mass dynamics omit many spacecraft realities.
   Evidence: `simulator/physics.py`, `envs/orbit_env.py`, README limitations.
   Risk: No attitude, mass depletion, actuator delay, sensor uncertainty, J2, drag, 3D dynamics.

### E. Weak Engineering

1. Rollout logic is duplicated across phase scripts.
   Evidence: Phase34/37A/37B scripts each reimplement environment stepping, post-cross sync, success checks, and failure labels.
   Why it matters: drift risk.

2. Benchmark cases are encoded independently in scripts.
   Evidence: GitHub issue #4 asks for mechanical 24-case manifest.
   Why it matters: future scripts may silently diverge.

3. CI/reproducibility is not fully verified locally.
   Evidence: GitHub issue #5 and missing local `pytest`.
   Why it matters: regression guard passes, but smoke tests did not run here.

4. Generated markdown can contain wording drift.
   Evidence: `analysis/full_goal_mode_audit/scientific_honesty_audit.md`.
   Why it matters: old scripts can regenerate "Success" wording without "simulator-defined."

### F. Weak Scientific Reasoning

1. The term "recoverability" is operational, not formally derived.
   Evidence: metric definition table and code thresholds.
   What to say: "I use recoverability as a simulator-defined viability proxy, not a theorem."

2. Phase34's causal interpretation depends on unchanged upstream behavior.
   Evidence: Phase34 preserves Phase22/31-style transfer and changes post-cross mode.
   Risk: If duplicated code changes more than intended, causal attribution weakens.

3. The project lacks formal statistics.
   Evidence: 24-case structured grids, no confidence intervals in main summaries.
   What to say: "This is a controlled diagnostic benchmark, not a randomized statistical generalization study."

### G. Potential Overclaims

1. "AI-Controlled Spacecraft Orbital Simulator" in `README.md`.
   Risk: Sounds like AI solved spacecraft control. Safer: "2D orbital-control simulator with explicit and learned controller experiments."

2. "Successful insertion" language.
   Risk: Can imply real insertion. Safer: "simulator-defined recoverable insertion-like behavior."

3. "Optimal control outperform heuristic architectures" in `analysis/phase32_direct_optimal_control/summary.md`.
   Risk: SciPy direct shooting on 4 solves is not general optimal-control proof. Safer: "direct-shooting probe reached lower recoverability distance in selected cases."

4. "Recoverability physically reachable" in Phase32 wording.
   Risk: "physically" can sound real-world. Safer: "reachable under the simplified 2D dynamics in selected solves."

### H. Likely To Impress A Robotics PhD

1. Metric clarity: crossing vs recoverability vs simulator success.
2. Regression guard and artifact manifest.
3. Negative-result discipline, especially Phase37B.
4. Phase33 mechanism extraction before Phase34 implementation.
5. Awareness that learned policies failed because long-horizon phase structure was not recovered.

### I. Likely Difficult Follow-Ups

1. Why is your recoverability basin valid?
2. Why not formulate this as viability/reachability formally?
3. Why direct shooting rather than collocation, MPC, or trajectory optimization?
4. How sensitive are Phase34 results to thresholds and gains?
5. Are you sure post-cross sync, not accidental code drift, caused the improvement?
6. Why should a 24-case benchmark be trusted?
7. Could PPO failure be just bad reward design or insufficient training?
8. What does "generalization" mean here?
9. What would be the next rigorous experiment?

## Task 2 - Knowledge Dependency Graph

Critical chain:

```text
Newtonian gravity
-> 2D point-mass orbital dynamics
-> circular orbit speed v=sqrt(mu/r)
-> radius / radial velocity / tangential velocity decomposition
-> target-radius crossing
-> why crossing does not define orbital state
-> recoverability basin
-> post-cross synchronization
-> Phase33 structure extraction
-> Phase34 post-cross result
-> Phase36/37 upstream crossing-basin failure
-> scientific contribution and limitations
```

Learning/control chain:

```text
closed-loop control
-> phase-structured explicit controller
-> behavior cloning / covariate shift
-> PPO fine-tuning
-> long-horizon phase transition failure
-> hybrid explicit/learned future direction
```

Reproducibility chain:

```text
benchmark contract
-> artifact manifest
-> CSV result schemas
-> regression guard
-> GitHub CI issue
-> claim safety
```

Weak foundations to repair first:

1. Recoverability threshold meaning: Critical.
2. Orbital mechanics: radial/tangential decomposition, circular velocity, energy/angular momentum: Critical.
3. Control vocabulary: closed loop, feedback, stability, basin, viability: Critical.
4. RL/IL failure modes: Important.
5. Optimal control: direct shooting vs collocation vs MPC: Important.
6. ROS/MuJoCo/CasADi/MPC tradeoffs: Optional unless asked.

## Task 3 - Research Story Audit

Problem: In a simplified 2D orbital-control simulator, reaching target radius can look successful but not be dynamically recoverable.

Motivation: Single success labels and visual crossings can mislead controller development.

Gap: Earlier PPO, IL, heuristic, and transfer-family controllers did not distinguish crossing from recoverable post-cross state well enough.

Method: Define separate metrics; test explicit, learning, transfer, and direct-shooting approaches; isolate post-cross synchronization.

Experiments: Phase31 baseline, Phase32 direct-shooting probe, Phase33 structure extraction, Phase34 post-cross sync, Phase35/36/37 upstream tests, PPO/IL negative checks.

Results: Phase34 improves recoverable crossings on crossing-producing cases; upstream crossing-basin expansion remains unresolved; learned policies did not provide main positive result.

Contribution: Recoverability-aware evaluation and architecture diagnosis in simplified 2D orbital control.

Limitations: 2D, hand thresholds, reduced benchmark, no real spacecraft validation, no full optimality proof, learned policies negative.

Future work: Evidence-backed upstream search with regression guards; not broad Phase38 hype.

Logical jumps:

- Phase32 to "optimal control" is too strong because CasADi/IPOPT was unavailable.
- Phase34 to "solved insertion" is false; non-crossing cases remain.
- PPO/IL failure to "AI cannot work" is false; only tested setups failed.
- Closest approach improvement to crossing is false; Phase37B proves why.

## Task 4/5 - 100 Interview Questions With Follow-Up Tree

Format: Q = primary question. F1/F2 = follow-ups. Total: 300 questions.

### Project

1. Q: What is the one-sentence research question of your project?
   F1: Why is "crossing is not insertion" scientifically meaningful?
   F2: Which files support that claim?
2. Q: What is your strongest result?
   F1: Why is Phase34 stronger than Phase7.6?
   F2: What exact numbers prove it?
3. Q: What is not solved yet?
   F1: What evidence shows non-crossing cases remain unresolved?
   F2: What would count as solving them?
4. Q: Why is this a research project rather than only a software project?
   F1: What hypotheses were tested?
   F2: Which negative result changed your direction most?
5. Q: What would you say if someone calls this "AI spacecraft control"?
   F1: Why is that wording risky?
   F2: What role did AI actually play?

### Physics

6. Q: What dynamics are simulated?
   F1: Where is this implemented?
   F2: What physics are missing?
7. Q: Why is circular speed `sqrt(mu/r)` relevant?
   F1: How does tangential velocity error use this?
   F2: What happens if velocity magnitude is correct but direction is wrong?
8. Q: What are radial and tangential velocity?
   F1: Why does radial velocity matter at crossing?
   F2: How does Phase34 use it?
9. Q: Why does target radius alone not define an orbit?
   F1: What other state components matter?
   F2: Which experiment shows this?
10. Q: What numerical integration method is used?
    F1: What limitations does that create?
    F2: Why not claim high-fidelity astrodynamics?

### Orbital Mechanics

11. Q: What is a target-radius crossing?
    F1: Why is it geometric?
    F2: What metric records it?
12. Q: What is a recoverable crossing?
    F1: Does it mean the first crossing state is recoverable?
    F2: Which Phase34 note clarifies this?
13. Q: How do energy and angular momentum enter your diagnosis?
    F1: Are they primary success metrics?
    F2: Why are they diagnostic only?
14. Q: What makes a crossing state bad?
    F1: Is radial velocity or tangential velocity more important?
    F2: What evidence can you cite?
15. Q: What does "basin" mean here?
    F1: Is your basin formally proven?
    F2: How should you word it honestly?

### Simulation

16. Q: What are the default physical constants and assumptions?
    F1: Which code files define them?
    F2: Are they realistic spacecraft parameters?
17. Q: Why use a simplified simulator?
    F1: What does simplification allow scientifically?
    F2: What conclusions become invalid?
18. Q: What does `simulator-defined success label` mean?
    F1: Why not just say success?
    F2: How do CAPTURE and LOCK differ from real capture?
19. Q: How are overspeed and instability used?
    F1: Why are they not real safety proofs?
    F2: What phases report zero overspeed?
20. Q: What would improve simulation fidelity?
    F1: Which improvements are realistic near-term?
    F2: Which are outside this interview's evidence?

### Controllers

21. Q: What is an explicit phase controller?
    F1: Why use phases?
    F2: What failure mode happens without phase structure?
22. Q: What did Phase7.6 accomplish?
    F1: What was the grid size and result?
    F2: Why did Phase8 weaken the claim?
23. Q: What did Phase31 show?
    F1: Why were crossings not enough?
    F2: What were the full and reduced scopes?
24. Q: What did Phase34 change?
    F1: What did it keep fixed?
    F2: Why does that matter for causal interpretation?
25. Q: What is `radius_priority`?
    F1: Why was it chosen?
    F2: What are its benchmark counts?
26. Q: What are CAPTURE and LOCK?
    F1: Why are they useful internally?
    F2: Why are they dangerous in public wording?
27. Q: What is post-cross synchronization?
    F1: Which variables are synchronized?
    F2: Why after crossing, not before?
28. Q: Why did radial timing not work in Phase37A?
    F1: Which variants preserved baseline?
    F2: What did early/mid commitment do?
29. Q: Why did weak tangential shaping fail in Phase37B?
    F1: What improved despite failure?
    F2: Why was it unsafe globally?
30. Q: What is the next controller idea?
    F1: Why not implement it immediately?
    F2: What regression guards must it satisfy?

### Behavior Cloning

31. Q: What was behavior cloning supposed to do?
    F1: What dataset size appears in `analysis/ppo_transfer_results.md`?
    F2: Why did low validation loss not imply success?
32. Q: What is covariate shift?
    F1: How might it explain BC failure?
    F2: What evidence shows rollout failure?
33. Q: What is phase-conditioned IL?
    F1: Did it use oracle phase online?
    F2: What exact result did it get?
34. Q: What did soft phase conditioning change?
    F1: Did it recover crossing?
    F2: What remained missing?
35. Q: What did minimal IL show?
    F1: What were sample count and train loss?
    F2: Why did it still fail?

### PPO

36. Q: Why did you try PPO?
    F1: What is PPO good at in principle?
    F2: Why was it hard here?
37. Q: Did PPO solve the task?
    F1: What were radius crossings for PPO fine-tuned from BC?
    F2: What wording should you avoid?
38. Q: Could PPO failure be due to reward design?
    F1: How should you answer honestly?
    F2: What experiment would be needed to know?
39. Q: Why not end-to-end RL from scratch?
    F1: What does sparse long-horizon structure imply?
    F2: How did explicit structure help?
40. Q: What is the future role of learning?
    F1: Learning terminal controller or upstream planner?
    F2: What must learning preserve?

### Recoverability

41. Q: Define recoverability in your project.
    F1: Is it simulator-defined?
    F2: What are the three state components?
42. Q: Why is recoverability not just success?
    F1: How do thresholds enter?
    F2: What metric is more diagnostic?
43. Q: What did Phase33 prove or not prove?
    F1: What was representative vs general?
    F2: How did it motivate Phase34?
44. Q: How do you know Phase34 entered the basin?
    F1: Which CSV/summary fields show it?
    F2: What regression check protects it?
45. Q: What is the weakest part of the recoverability concept?
    F1: What math would strengthen it?
    F2: How would viability theory relate?

### Experiments

46. Q: What is the 24-case grid?
    F1: Which values define it?
    F2: Which issue asks to make it mechanical?
47. Q: What is the Phase31 full benchmark?
    F1: How many cases?
    F2: Why not merge it with Phase34 reduced comparison?
48. Q: What did Phase32 test?
    F1: Why was it a fallback?
    F2: What were its horizon and intervals?
49. Q: What did Phase35 test?
    F1: Which variants failed badly?
    F2: What did predictive bias preserve?
50. Q: What did Phase36B test?
    F1: Which families?
    F2: Did any improve crossing count?
51. Q: What did Phase36C do?
    F1: Was it a new controller run?
    F2: What were the failure labels?
52. Q: What did Phase37A test?
    F1: How many rollouts?
    F2: How many new crossings?
53. Q: What did Phase37B test?
    F1: Why only subset?
    F2: Why not expand it?
54. Q: What would a clean Phase38 experiment require?
    F1: What variables are candidates?
    F2: What would be the stopping rule?
55. Q: What is your most important negative result?
    F1: Why?
    F2: How did it change the project?

### Evaluation

56. Q: Why evaluate crossing and recoverability separately?
    F1: What false conclusion does this prevent?
    F2: Which table shows it best?
57. Q: What is a regression case?
    F1: Why protect it?
    F2: How did Phase37B fail it?
58. Q: What is closest approach?
    F1: Why is it not a crossing?
    F2: What did Phase37B show?
59. Q: What is crossing potential?
    F1: Why is it diagnostic?
    F2: What would make it more useful?
60. Q: How do you compare algorithms fairly?
    F1: What must remain fixed?
    F2: What still threatens fairness?

### Engineering

61. Q: Why Python?
    F1: What did it enable?
    F2: What are its limits?
62. Q: Why NumPy?
    F1: Why not a full physics engine?
    F2: What errors can NumPy/Euler introduce?
63. Q: Why explicit controllers?
    F1: Why are they interpretable?
    F2: Why are they brittle?
64. Q: Why not MPC yet?
    F1: What evidence says planner search comes first?
    F2: What would MPC optimize?
65. Q: Why not CasADi?
    F1: What happened in Phase32?
    F2: What would CasADi add?
66. Q: Why not MuJoCo?
    F1: What is MuJoCo optimized for?
    F2: What orbital-specific needs remain?
67. Q: Why not ROS?
    F1: Is this a deployed robot system?
    F2: When would ROS matter?
68. Q: What is your biggest codebase weakness?
    F1: Why does duplicated rollout logic matter?
    F2: How would you refactor?
69. Q: What does the regression guard check?
    F1: What does it not check?
    F2: How did it run locally?
70. Q: Why did pytest not run here?
    F1: What does that imply?
    F2: How should you state it?

### Reproducibility

71. Q: Can I reproduce your main result?
    F1: Which command verifies stored outputs?
    F2: Which command reruns Phase34?
72. Q: What is an artifact manifest?
    F1: Why is it useful?
    F2: What artifacts are public evidence?
73. Q: What is the benchmark contract?
    F1: What terms does it define?
    F2: What drift does it prevent?
74. Q: Are all files tracked?
    F1: What did git status show?
    F2: Why does untracked evidence matter?
75. Q: What would you improve before sharing with a professor?
    F1: Benchmark manifest?
    F2: CI verification?

### Scientific Thinking

76. Q: What hypothesis did Phase34 test?
    F1: What outcome would have falsified it?
    F2: Did it solve the whole problem?
77. Q: How do you distinguish diagnosis from result?
    F1: Is Phase36C a result?
    F2: Is closest approach a result?
78. Q: What is your uncertainty?
    F1: Which claim has high confidence?
    F2: Which claim has low confidence?
79. Q: How do you avoid hindsight bias?
    F1: What logs show hypothesis evolution?
    F2: What should future experiments pre-register?
80. Q: What makes a negative result valuable?
    F1: Give a repo example.
    F2: How did it constrain future work?

### Failure Analysis

81. Q: Why did local Phase7.6 success fail broader Phase8 testing?
    F1: What was the Phase8 success rate?
    F2: What failure mode dominated?
82. Q: Why did learned policies fail?
    F1: Phase consistency or reward?
    F2: What evidence supports each possibility?
83. Q: Why did transfer families fail to expand crossings?
    F1: What did all Phase36B families achieve?
    F2: What did Phase36C reveal?
84. Q: Why did radial timing fail?
    F1: Was it unsafe?
    F2: Did it preserve known crossings?
85. Q: Why did weak tangential shaping damage regression cases?
    F1: What does this say about local shaping?
    F2: How would you redesign the diagnostic?

### Future Work

86. Q: What is the highest-value next experiment?
    F1: Why small?
    F2: What metric must it improve?
87. Q: What would you not do next?
    F1: Why not broad PPO?
    F2: Why not broad Phase38 implementation?
88. Q: How would you add formal control theory?
    F1: Lyapunov? viability? reachability?
    F2: Which would help most?
89. Q: How would you add higher fidelity?
    F1: 3D? J2? mass depletion?
    F2: Which comes first and why?
90. Q: How would you make learning useful?
    F1: Structure-aware policy?
    F2: Hybrid residual controller?

### PhD-Level Discussion

91. Q: Is your recoverability basin invariant?
    F1: Can you prove it?
    F2: What evidence short of proof do you have?
92. Q: Is Phase34 overfit to the reduced benchmark?
    F1: What would held-out testing look like?
    F2: What evidence cannot answer this?
93. Q: How would you formulate this as optimal control?
    F1: State, control, cost, constraints?
    F2: Terminal set or running recoverability cost?
94. Q: How would you formulate this as reachability?
    F1: What is the target set?
    F2: What are unsafe sets?
95. Q: What is the relationship between post-cross sync and orbital energy?
    F1: Is sync just energy matching?
    F2: Why angular momentum also matters?
96. Q: If I changed thrust limits, what would happen?
    F1: Which file has thrust-scale benchmark values?
    F2: Would Phase34 still work?
97. Q: What if sensor noise is added?
    F1: Which controller parts are brittle?
    F2: What robustness test would you design?
98. Q: What if actuation delay is added?
    F1: Which phase would suffer most?
    F2: How would MPC help?
99. Q: What is your claim of novelty?
    F1: What repository evidence supports it?
    F2: What literature evidence is still needed?
100. Q: What would make you change your conclusion?
     F1: What result would disprove Phase34's interpretation?
     F2: What result would strengthen it most?

## Task 6 - Weak Knowledge Detection

Critical:

- Viability / recoverability / invariant sets.
  Why it matters: Your central word is "recoverability."
  Math: sets, state space, basins, invariance, reachable sets.
  Control: Lyapunov stability, terminal sets, feedback basins.
  AI: safe RL, constrained policy learning.
  Papers to know: viability theory basics; reachability analysis; Hamilton-Jacobi reachability; ROAHM papers on reachability/RRT-style safety.

- Orbital mechanics beyond circular velocity.
  Why it matters: a robotics PhD may test whether you understand radius vs full orbit state.
  Math: vectors, energy, angular momentum, polar coordinates.
  Control: state feedback in radial/tangential coordinates.
  AI: state representation matters.
  Papers/books: Vallado or Curtis orbital mechanics basics.

- Optimal control formulation.
  Why it matters: Phase32 uses "optimal-control" language.
  Math: objective, constraints, gradients, collocation.
  Control: direct shooting vs collocation vs MPC.
  AI: differentiable simulation and policy optimization.
  Papers/books: Betts practical methods for optimal control; CasADi docs if used.

Important:

- PPO and imitation-learning failure modes.
- Benchmark design and regression testing.
- Numerical integration error.
- Control saturation and thrust constraints.

Optional:

- ROS, MuJoCo, transformers, diffusion policies.
- High-fidelity astrodynamics perturbations.
- Distributed spacecraft autonomy vision.

## Task 7 - Scientific Honesty Audit

Overclaim candidates and safer versions:

1. Quote: "AI-Controlled Spacecraft Orbital Simulator" (`README.md` title).
   Risk: implies AI success and spacecraft readiness.
   Reviewer reaction: "Where is the successful AI controller?"
   Safer: "Simplified 2D spacecraft orbital-control simulator with explicit and learned controller experiments."

2. Quote: "Successful insertion in this sandbox requires..." (`README.md`).
   Risk: "successful insertion" can sound real.
   Safer: "Simulator-defined insertion-like success in this sandbox requires..."

3. Quote: "Can optimal control outperform heuristic architectures? `yes`." (`analysis/phase32_direct_optimal_control/summary.md`).
   Risk: too strong for 4 solves and SciPy fallback.
   Safer: "In selected direct-shooting probes, optimized trajectories reached lower recoverability distance than tested heuristic variants."

4. Quote: "Is recoverability theoretically reachable? `yes as a state in this coarse solve`." (`analysis/phase32_direct_optimal_control/summary.md`).
   Risk: "theoretically" sounds like proof.
   Safer: "The coarse direct-shooting solve found recoverable states in selected simplified-dynamics cases."

5. Quote: "Phase34 solved post-cross recoverability." (`README.md`).
   Risk: sounds complete.
   Safer: "Phase34 solved the tested post-cross recoverability gap for crossing-producing cases in the reduced benchmark."

6. Quote: "The explicit controller contains the correct control structure." (`analysis/ppo_transfer_results.md`).
   Risk: "correct" sounds universal.
   Safer: "The explicit controller contains the control structure that succeeds in this documented fixed-baseline comparison."

7. Quote: "The current project establishes the first layer of control-science primitives..." (`README.md`).
   Risk: broad importance claim.
   Safer: "The current project defines and tests several control-architecture primitives in a simplified 2D benchmark."

8. Quote: "true distributed autonomy requires cognitive diversity" (`README.md` long-term vision).
   Risk: far beyond repo evidence.
   Safer: Keep as personal long-term vision, not current result.

## Task 8 - Closed-Book Memory Test

Must memorize:

- Phase7.6: `soft_linear_3e4`, `217 / 270` success, `217 / 270` CAPTURE, `8` near-miss.
- Phase8: `220 / 1296` success, `265 / 1296` crossings/CAPTURE, dominant `no_capture_access`.
- Phase31 full: reduced 48-case grid; best crossings `12`; recoverable `0` for all listed families.
- Phase32: SciPy direct shooting fallback; CasADi unavailable; horizon `512` physics steps, `64` control intervals; 4 solves per objective.
- Phase33 best: `recoverability_target / baseline_crossing_high_angle`; first crossing step `81`; best recoverability step `512`; best sync `0.000464`; best distance `0.000470`; crossing-state distance `2.313443`.
- Phase34: 24 cases; Phase31-style reduced reference `8 / 24` crossings, `0 / 24` recoverable; `radius_priority` `8 / 24` crossings, `8 / 24` recoverable; crossing-case best distance `3.9923 -> 0.9855`; overspeed `0`.
- Phase35: baseline and predictive bias `8 / 24`; radial energy push and tangential corridor `0 / 24`; radial energy push overspeed `5`.
- Phase36B: four families; each `8 / 24` crossings, `8 / 24` recoverable; overspeed `0`; instability `0`.
- Phase36C: baseline non-crossing `16 / 24`; `8` near-crossing, `8` over-conservative transfer.
- Phase37A: 6 variants x 24 = `144` rollouts; `0 / 16` new crossings; delayed low/medium preserved `8 / 24`; overspeed `0`; instability `0`.
- Phase37B: `24` subset rollouts; weak tangential selected crossings `0 / 4`; selected recoverable `0 / 4`; regression preservation `4 / 8`; closest approach improved `3 / 4`; overspeed `0`; instability `0`.
- PPO transfer: BC balanced samples `3063`; explicit crossing `1`; BC `0`; PPO fine-tuned `0`; explicit final radius error `31433.54`; BC `375039922.79`; PPO `374964010.17`.
- Minimal IL: `48,458` samples; `60` epochs; train loss `0.00030948646601108544`; no crossing; no success.
- Demo: success true; crossings `1`; first crossing step `48,269`; final radius error `27,657.63 m`.

Flashcards:

1. What is the Phase34 headline? Answer: `8 / 24` crossings preserved, recoverable crossings `0 / 24 -> 8 / 24`.
2. What is the Phase37A total rollout count? Answer: `144`.
3. What is Phase37B's regression failure? Answer: weak tangential preserved only `4 / 8` crossing and recoverable regression cases.
4. What does Phase36C prove? Answer: diagnostic only; `16 / 24` baseline non-crossing cases split into `8` near-crossing and `8` over-conservative.
5. Did PPO solve the task? Answer: No; BC and PPO fine-tuned had `0` crossings in the documented comparison.

Oral exact-recall questions:

- Give the three Phase34 numbers that matter most.
- Give the Phase37B selected-case and regression-case counts.
- State the Phase32 solver limitation.
- State the Phase8 generalization result.
- State why Phase31 and Phase34 denominators must not be mixed.

## Task 9 - Engineering Defense

Why Python: fast experimentation, NumPy plotting stack, easy script-based phase sweeps. Limitation: speed, duplicated scripts, weaker deployment story.

Why NumPy: transparent vector math for 2D dynamics and diagnostics. Limitation: no high-fidelity physics engine or automatic differentiation by default.

Why explicit controllers: interpretable, easy to diagnose phase structure, strong for hypothesis testing. Limitation: hand-tuned and brittle.

Why PPO: natural RL baseline for continuous control. Honest result: tested PPO did not recover crossing behavior.

Why BC: try to transfer explicit controller structure. Honest result: low loss did not imply rollout success.

Why not MPC yet: Phase36/37 show upstream geometry variable is still unclear; MPC would add complexity before knowing what to optimize.

Why not trajectory optimization as main controller: Phase32 was a probe; CasADi unavailable; direct shooting is not production control.

Why not diffusion / transformer policies: no evidence that data volume, state representation, or long-horizon supervision are ready.

Why not end-to-end RL: sparse long horizon and phase transitions already defeated simpler learning transfer.

Why not CasADi: intended but unavailable in Phase32 runtime.

Why not MuJoCo: orbital mechanics and long-horizon target-radius recoverability are not MuJoCo's core advantage.

Why not ROS: no hardware robot or distributed runtime yet.

## Task 10 - Scientific Defense

Why this problem: It exposes a real control issue: reaching a geometric milestone does not guarantee dynamic recoverability.

Why this simulator: It isolates the control-architecture question before high-fidelity complications.

Why this metric: Crossing alone was misleading; recoverable crossing adds radius, radial velocity, tangential velocity alignment.

Why this evaluation: Phase34/36/37 use comparable 24-case scopes and fixed terminal controller.

Why these baselines: PPO/BC/IL, explicit heuristics, transfer families, direct-shooting probe cover learning, hand control, planning, and optimization.

Why these limitations: They are forced by evidence; no 3D, no flight validation, no broad generalization.

What negative results taught: proxy improvements and local shaping are insufficient unless they create actual crossings and preserve regression cases.

Realistic future work: small Phase38 upstream search with protected `8 / 24` crossing cases, then formal MPC/trajectory optimization only if the search finds a useful variable.

## Task 11 - Research Communication

30 seconds:

> My project studies a simplified 2D spacecraft orbital-control benchmark. The main finding is that crossing the target radius is not the same as reaching a recoverable orbit-like state. In the Phase34 reduced benchmark, post-cross synchronization preserved `8 / 24` crossings and improved recoverable crossings from `0 / 24` to `8 / 24`, but later Phase36 and Phase37 tests showed that creating new upstream crossings remains unresolved.

60 seconds:

> I built a 2D orbital-control sandbox to compare learned policies, explicit controllers, and transfer-style controllers. The important shift in the project was realizing that target-radius crossing is only a geometric event. Phase31-style behavior could cross the target radius but still had `0 / 24` recoverable crossings in the reduced comparison. Phase32 and Phase33 suggested that recoverability can happen after crossing through smooth synchronization of radius, radial velocity, and tangential velocity. Phase34 implemented that idea explicitly and converted the crossing-producing cases into recoverable crossings. I do not claim this is real spacecraft control or that AI solved it. The current unresolved problem is upstream crossing generation.

3 minutes:

> Start with simulator, metrics, Phase31 gap, Phase32/33 mechanism, Phase34 result, Phase36/37 negative evidence, PPO/IL negative evidence, limitations.

10 minutes:

1. Problem and simulator assumptions.
2. Why crossing is not insertion.
3. Early learning and local explicit-controller background.
4. Phase31 crossing without recoverability.
5. Phase32/33 post-cross mechanism.
6. Phase34 main result.
7. Phase35/36/37 upstream failures.
8. Engineering evidence: benchmark contract, artifact manifest, regression guard.
9. Limitations.
10. Future work.

## Task 13 - Three-Day Intensive Preparation Plan

Current date/time: June 15 afternoon. Interview: June 18, 10 PM.

June 15 afternoon/evening:

- 15:00-16:00: Read `README.md`, `docs/benchmark_contract.md`, `analysis/artifact_manifest.md`. Memorize definitions. Expected improvement: high.
- 16:00-17:00: Redraw pipeline: transfer -> crossing -> post-cross sync -> recoverability. Do not read long-term vision. Expected improvement: high.
- 17:00-18:00: Read Phase31, Phase33, Phase34 summaries. Explain aloud why first crossing is not recoverability. Expected improvement: very high.
- 19:00-20:00: Memorize exact Phase34 numbers. Expected improvement: very high.
- 20:00-21:00: Read PPO/IL negative files. Practice "AI did not solve it" answer. Expected improvement: high.
- 21:00-22:00: Oral drill: questions 1-30.

June 16:

- 09:00-10:00: Physics basics: circular speed, radial/tangential decomposition, energy/angular momentum. Expected improvement: very high.
- 10:00-11:00: Code revisit: `simulator/physics.py`, `envs/orbit_env.py`. Do not deep-dive all plotting. Expected improvement: high.
- 11:00-12:00: Phase34 script structure. Focus on what changed and what stayed fixed. Expected improvement: high.
- 13:00-14:00: Phase35/36B/36C. Memorize negative results. Expected improvement: high.
- 14:00-15:00: Phase37A/B. Memorize `144`, `0 / 16`, `0 / 4`, `4 / 8`, `3 / 4`. Expected improvement: very high.
- 15:00-16:00: Reproducibility: benchmark contract, artifact manifest, regression guard. Expected improvement: medium-high.
- 19:00-21:00: Oral drill: questions 31-70.
- 21:00-22:00: Write one-page personal cheat sheet from memory.

June 17:

- 09:00-10:30: Weak foundations: viability, basin, invariance, MPC, direct shooting vs collocation. Expected improvement: very high.
- 10:30-12:00: Engineering defense: Python, NumPy, explicit controllers, why not MPC/CasADi/MuJoCo/ROS. Expected improvement: high.
- 13:00-14:00: Overclaim audit. Practice safe replacements. Expected improvement: high.
- 14:00-15:00: ROAHM prep. Vocabulary: reachability, optimization, differentiable simulation, safety, planning. Expected improvement: medium-high.
- 15:00-16:00: Draw three diagrams: state decomposition, Phase33 timeline, Phase34 benchmark table. Expected improvement: high.
- 19:00-21:30: Mock interview. Use questions 71-100.
- 21:30-22:00: Fix weak answers only.

June 18:

- 09:00-10:00: Exact numbers flashcards.
- 10:00-11:00: Explain full story in 3 minutes and 10 minutes.
- 11:00-12:00: Review limitations. Do not add new claims.
- 13:00-14:00: Read Phase34, Phase37B, benchmark contract one last time.
- 14:00-15:00: Practice hard questions: recoverability validity, benchmark size, PPO fairness, why no MPC.
- 15:00-16:00: Rest from new material.
- 19:00-20:00: Final oral run: 30s, 60s, 3min intro.
- 20:00-21:00: Exact recall only.
- 21:00-21:45: Stop studying new material. Review cheat sheet.
- 21:45-22:00: Quiet setup.

Do not spend time on:

- Broad long-term vision.
- New experiments.
- Paper wording.
- Pretty figures.
- Deep PPO implementation unless asked.
- Full print library duplicates.

## Task 14 - ROAHM Lab Preparation

Sources checked online:

- DEFORM project/paper: `https://roahmlab.github.io/DEFORM/`, `https://arxiv.org/abs/2406.05931`. This includes Yizhou Chen and differentiable physics for deformable linear objects.
- ARMOUR project/paper: `https://roahmlab.github.io/armour/`, `https://arxiv.org/abs/2301.13308`. This reflects ROAHM-style robust planning, optimization, uncertainty, and reachability.
- Public search results associating Yizhou Chen with University of Michigan / ROAHM / differentiable physics / robotics.

Likely ROAHM research philosophy:

- Safety-aware planning and control.
- Optimization-based robotics.
- Reachability / verification / constraints.
- Learning integrated with models, not hype-only ML.
- Simulation and differentiable physics when useful.
- Careful evaluation under constraints and failure cases.

Alignment with your project:

- Strong alignment: reachability/recoverability framing, failure modes, regression guards, optimization/control vocabulary.
- Partial alignment: simplified simulator and explicit controllers are compatible with early-stage research, but not yet ROAHM-level formal safety.
- Weak alignment: learned policies are negative and not algorithmically novel.

Be careful:

- Do not force a connection to differentiable physics. Your Phase32 is SciPy direct shooting, not differentiable physics research.
- Do not claim formal reachability. Say your recoverability metric is an operational proxy.
- Do not oversell autonomy. Say simplified closed-loop simulator.

Vocabulary to use correctly:

- state space, terminal set, basin, feedback policy, rollout, closed loop, constraint, objective, proxy metric, regression guard, failure mode, ablation, direct shooting, MPC, reachability.

## Task 15 - Final Readiness Report

Scores:

- Research Readiness: 78/100
- Engineering Readiness: 68/100
- Scientific Thinking: 82/100
- Interview Readiness: 62/100 today, can reach 78/100 by June 18
- Communication: 70/100
- Mathematics Readiness: 55/100
- Control Theory Readiness: 58/100
- Machine Learning Readiness: 62/100
- Robotics Readiness: 55/100
- Confidence Level: moderate if you stay honest; low if you overclaim.

Weakness table:

| Weakness | Time to fix | Difficulty | Priority | Interview impact |
|---|---:|---|---|---|
| Recoverability/invariant-set theory | 6-8 h | Hard | Critical | Very high |
| Exact result recall | 3 h | Medium | Critical | Very high |
| Orbital mechanics basics | 4-6 h | Medium | Critical | High |
| PPO/BC failure explanation | 2 h | Medium | Important | High |
| Direct shooting vs collocation/MPC | 3 h | Medium | Important | High |
| Reproducibility story | 1.5 h | Easy | Important | Medium |
| ROAHM alignment | 1 h | Easy | Important | Medium |
| Long-term vision restraint | 30 min | Easy | Important | High |

The Top 20 Things Sean Must Master Before June 18:

1. Phase34 headline: `8 / 24` crossings, recoverable `0 / 24 -> 8 / 24`.
2. Difference between crossing and recoverable crossing.
3. Why first crossing can be dynamically bad.
4. Phase33 timing: crossing step `81`, best recoverability step `512`.
5. Phase32 limitation: SciPy direct shooting fallback, not CasADi/IPOPT proof.
6. Phase37A: `144` rollouts, `0 / 16` new crossings.
7. Phase37B: `0 / 4` selected crossings, `4 / 8` regression preservation.
8. Phase36C: `16 / 24` non-crossing split into `8` near-crossing and `8` over-conservative.
9. PPO/BC did not solve: learned policies `0` crossings in fixed-baseline comparison.
10. Why 24-case benchmark supports controlled comparison, not broad generalization.
11. Why simulator success is not mission success.
12. The 2D dynamics assumption and missing physics.
13. Why explicit controllers were scientifically useful.
14. Why not MPC/CasADi yet.
15. Why closest approach is only diagnostic.
16. Regression guard purpose and what passed.
17. Local pytest gap: missing `pytest` in checked interpreters.
18. GitHub issues #3-#7 as current engineering/research cleanup tasks.
19. Safe novelty claim: recoverability-aware evaluation and post-cross architecture diagnosis.
20. Safest one-sentence limitation: "I cannot conclude real spacecraft readiness or broad generalization from repository evidence."

## Task 12 - Oral Examination Mode

Start with one question:

What exactly is the difference between a target-radius crossing and a recoverable crossing in your repository, and what Phase34 evidence proves that this distinction matters?
