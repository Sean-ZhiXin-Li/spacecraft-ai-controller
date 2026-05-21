# Ben Kraske-Oriented Project Questions

## 150-Word Project Summary

I am building a 2D physics-based orbital-control sandbox that compares PPO reinforcement learning, behavior cloning, probe baselines, and explicit rule-based controllers under thrust-limited Newtonian dynamics. The original goal was simple orbit insertion, but the main finding has become more structural: reaching or crossing the target radius is not the same as achieving closed-loop orbit lock. In the current evidence trail, PPO can reduce local radial motion but does not reliably create the phase-changing control needed for insertion: descent, post-crossing capture, and lock. The best explicit controllers are not flight-ready, and their success region is narrow, but they make the failure modes visible. They show that recoverability depends on simultaneous radius, radial-velocity, and tangential-velocity synchronization after crossing. I want to reframe the project as a study of explainable autonomous decision-making under uncertainty, where learned policies are evaluated by interpretable failure structure rather than reward alone. This keeps the claims technically bounded.

## Relevant Repository Map

- Orbital environment and state definition: `envs/orbit_env.py`; alternate multi-task/noise hooks: `envs/orbit_env_mt.py`, `envs/multi_orbit_env.py`.
- Action space: `envs/orbit_env.py` uses a 2D normalized action in `[-1, 1]^2`, scaled by `thrust_scale`.
- Observation: `envs/orbit_env.py` returns `[x, y, vx, vy, v_r]`. PPO inference paths often use 4D `[x, y, vx, vy]` or 5D variants depending on checkpoint.
- Reward function: `ppo_orbit/rewards_utils.py`; includes radius error, speed error, radial velocity suppression, tangential alignment, energy/angular-momentum proxies, smoothness, and lock-like bonuses.
- PPO controller/checkpoints: `controller/ppo_controller.py`, `ppo_orbit/ppo.py`, `ppo_orbit/ppo_infer_model.py`, `scripts/test_ppo_controller.py`.
- Explicit/rule-based controllers: `controller/orbit_lock_controller.py`, `controller/stable_orbit_controller.py`, `controller/expert_controller.py`, plus phase scripts under `scripts/explicit_controller_phase*.py`.
- Trajectory logging and replay: `tools/replay_recorder.py`, `scripts/record_replay.py`, `utils/replay_io.py`, `analysis/behavior_metrics.py`.
- Diagnostic plots: `scripts/day20_policy_surface.py`, `scripts/day23_orbit_lock_eval.py`, `tools/diagnostics/diag_orbit.py`, `tools/plots/plot_radius_vs_time.py`.
- Existing summaries: `README.md`, `analysis/ppo_failure_mechanism.md`, `analysis/orbit_lock_benchmark.md`, `analysis/orbit_lock_generalization.md`, `analysis/phase34_post_cross_sync/summary.md`, `analysis/final_project_summary.md`.

## Why This Connects To Your Work

This project is currently closer to an MDP than a true POMDP: the main environment exposes position, velocity, and derived radial velocity, and the physics is deterministic except for initialization and configuration choices. It becomes POMDP-like only if target parameters, sensor readings, actuator effectiveness, delays, or fault modes are hidden from the controller.

That said, the project connects naturally to decision-making under uncertainty because the hard part is not minimizing a scalar reward. The hard part is choosing when to switch control regimes under long-horizon dynamics, where local progress can be misleading. It connects to explainable AI because explicit controllers expose named phases and failure modes, while PPO behavior must be probed through policy surfaces and state-region disagreements. It connects to aerospace autonomy for safety because the key distinction is between apparent progress, such as radius crossing, and recoverable, survivable control behavior.

## Current Technical Finding

The supported finding is: PPO has not reliably learned true closed-loop orbital insertion in the current benchmarks. Existing diagnostics show low-action or shutdown-like behavior, no target-radius crossing in representative PPO benchmark cases, and missing sign-changing radial-velocity correction. Explicit phase controllers are not universal, but in narrow regimes they create the missing structure: descent, crossing, capture, and lock.

Concrete evidence to mention:

- `analysis/orbit_lock_benchmark.md`: explicit controller succeeds in `2 / 3` representative setups; PPO succeeds in `0 / 3`.
- `analysis/figs/day20_policy_surface/collapse_states.json`: PPO checkpoints show low action norms near unresolved states.
- `analysis/phase34_post_cross_sync/summary.md`: post-cross synchronization improves crossing-producing cases from `0 / 8` recoverable crossings to `8 / 8`, while non-crossing families remain unsolved.

## Possible POMDP/XAI Framing

The current project should be presented as an MDP-style orbital control sandbox with a clear path toward POMDP experiments. A principled POMDP version would keep the same 2D physics and controllers but hide or corrupt selected variables, then test whether policies or controllers can infer latent state from history.

Good POMDP/XAI framing:

- Treat orbit insertion as phase inference: descent, capture, lock, fault/recovery.
- Treat PPO failure as a question of explainable decision structure: where does the policy stop acting, ignore sign changes, or fail to switch modes?
- Treat explicit controllers as interpretable reference policies, not proof of flight readiness.
- Evaluate safety through recoverability, crossing quality, sign-changing correction, and failure classification, not just reward.

## Hidden-State And Uncertainty Opportunities

| Variable | Why it matters | POMDP-like? | Minimal test | Metric |
|---|---|---:|---|---|
| Observation noise in `r`, `v_r`, or velocity | Navigation sensors are imperfect; radial damping may be fragile. | Yes, if noise is unobserved. | Add noise wrapper around observations only. | Success rate, tail `|v_r|`, false CAPTURE/LOCK transitions. |
| Delayed observations | Guidance uses stale state estimates. | Yes. | Queue observations by 1-20 steps. | Crossing count, recoverable crossings, lock loss events. |
| Unknown thrust efficiency | Same commanded action produces different acceleration. | Yes, if efficiency is hidden. | Multiply executed thrust by fixed or sampled factor per episode. | Radius crossing rate, final radius error, fuel/action norm. |
| Hidden thrust degradation | Long-duration actuator wear affects control authority. | Yes. | Let thrust efficiency decay over time. | Time-to-crossing, success under degradation, recovery after drift. |
| Sensor dropout | Missing observations force history-based control. | Yes. | Randomly hold last observation or mask velocity components. | Dropout tolerance curve, tail stability, unsafe termination rate. |
| Uncertain target orbit | Mission target may be estimated, moving, or partially known. | Yes if target is not in observation. | Randomize target radius and hide or noisy-encode it. | Generalization success, target-radius error normalized by target. |
| Unmodeled perturbations | Solar pressure, third-body effects, or numerical mismatch. | Partly. | Add small unobserved acceleration bias. | Robust success rate, correction latency, drift after lock. |
| Hidden mode/fault state | Fault-aware autonomy is central to safety. | Strongly. | Add discrete actuator/sensor fault mode per episode. | Fault detection latency, safe abort/fallback rate, policy disagreement regions. |

## Explainable Policy Diagnostics Without Retraining

| Diagnostic | File/script to inspect or extend | Output | Question answered |
|---|---|---|---|
| Action distribution vs radius error | Extend `scripts/day20_policy_surface.py` | CSV plus heatmap of action norm/components over `r_error` | Does PPO command different actions inside vs outside target radius? |
| Action distribution vs radial velocity | Extend `scripts/day20_policy_surface.py` | `action_norm_vs_vr` and `action_radial_vs_vr` plots | Does PPO actively damp both signs of `v_r`? |
| Behavior near target radius | Use `scripts/day20_policy_surface.py` and `scripts/day23_orbit_lock_eval.py` | local grid around `r_error=0`, `v_r=0`, `v_t_error=0` | Does PPO lock, coast, or drift near target? |
| Sign-change response | Extend `scripts/day23_orbit_lock_eval.py` | tail sign-change counts and action-after-crossing CSV | Does PPO change behavior when `r_error` or `v_r` changes sign? |
| Coast/freeze bias | Existing Day20 collapse detector | shutdown-step CSV and action-norm timeline | Is PPO treating low motion as success before orbit lock? |
| Explicit rule comparison | Compare `LoadedPolicy.act` with `OrbitLockController.act_with_info` | disagreement heatmap over local state grid | Where does PPO disagree with the interpretable controller? |
| Phase abstraction probe | Use phase labels from `analysis/phase_controller_dataset` | phase-conditioned action summary | Can PPO behavior be approximated by descent/capture/lock regions? |

## Questions To Ask Ben

1. In a deterministic 2D orbital simulator with observation `[x, y, vx, vy, v_r]`, should I describe the current control problem as an MDP and reserve the POMDP framing for planned uncertainty wrappers?
2. If I introduce partial observability, which latent variable would be most principled for aerospace autonomy: delayed state, noisy navigation, hidden thrust efficiency, target-orbit uncertainty, or fault mode?
3. Is it better to make the target orbit hidden/noisy, or to keep the target known and hide actuator/sensor health, if the research goal is safety-oriented decision-making under uncertainty?
4. PPO reduces radial motion but fails to create sign-changing radial-velocity correction after crossing. What XAI method would best distinguish "stabilizing" from "freezing" behavior?
5. Would policy-surface plots over `(radius error, radial velocity, tangential velocity error)` be a credible explanation tool, or should I build a more formal state abstraction or decision graph?
6. Can an explicit descent/capture/lock controller be treated as an interpretable policy abstraction for comparing learned policies, even if it is hand-built and only works in a narrow regime?
7. How would you evaluate whether a learned policy has discovered the same decision structure as an explicit controller: action agreement, phase classification, transition timing, or recoverability outcomes?
8. For a POMDP version, should I start with memoryless PPO under observation corruption as a failure baseline, then compare it to a history-aware policy or belief-state estimator?
9. What safety metric would be most defensible here: probability of recoverable crossing, time inside an unsafe radial-velocity region, fault recovery rate, or verified invariant violations?
10. If I build one research-quality next module as a high school student, would a fault/uncertainty wrapper plus explainable failure taxonomy be more valuable than another controller-tuning phase?
11. Could decision trees, finite-state machines, or option-like policy graphs be used to extract a human-readable policy from PPO rollouts in this setting?
12. What would make this project scientifically stronger: broader uncertainty sweeps, formal POMDP modeling, better policy explanations, or a smaller but cleaner benchmark with rigorous metrics?

## Three Possible Next Project Modules

1. Easy: PPO/expert disagreement atlas. Extend `scripts/day20_policy_surface.py` to produce CSV/plots showing where PPO and `OrbitLockController` choose different radial/tangential actions. This is mostly analysis-only and does not require retraining.
2. Medium: Observation-noise and delay wrapper. Add a wrapper around observations, not physics, to test navigation uncertainty. Compare PPO, explicit controller, and probe baselines on success, recoverable crossing, tail `|v_r|`, and false lock transitions.
3. Hard: Hidden fault-state POMDP module. Introduce per-episode latent actuator degradation or sensor dropout, then test memoryless policies/controllers against a history-aware estimator or explicit fault-detection logic.

## Three Strongest Figures/Results To Mention In Email

1. `analysis/demo/orbit_demo_trajectory.png` and `analysis/demo/orbit_demo_zoom.gif`: visual example of explicit-controller crossing, CAPTURE, and LOCK behavior.
2. `analysis/figs/day20_policy_surface/action_norm_vs_vr_compare.svg` plus `collapse_states.json`: PPO policy-surface evidence for low-action behavior near unresolved states.
3. `analysis/phase34_post_cross_sync/post_cross_sync_examples.png` and `summary.md`: post-cross synchronization converts crossing-producing cases from `0 / 8` to `8 / 8` recoverable crossings, without claiming universal insertion.

## Missing Evidence That Would Strengthen The Summary

- A clean, current PPO-vs-explicit benchmark table using the same environment, horizon, thrust scale, and target radius for every controller.
- A policy-surface CSV that includes radial and tangential action components, not only action norm.
- A formal definition of recoverability distance and exactly how it maps to CAPTURE/LOCK/success labels.
- A small uncertainty benchmark with fixed seeds, confidence intervals, and failure categories.
- A short note resolving PPO observation dimensionality differences between 4D inference checkpoints and the 5D environment observation.
