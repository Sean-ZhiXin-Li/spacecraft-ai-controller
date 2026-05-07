# Phase 29 Repository Audit

## Scope Reviewed

- `README.md` and current milestone narrative.
- `project_log/`, especially Phase 6.5-7.6 logs and `sprint_ppo22-27.md`.
- Analysis outputs for Phases 20, 20.5, 21, 22, 23, 24, 25, 26, 27, and 28.
- Scripts `scripts/explicit_controller_phase20*` through `scripts/explicit_controller_phase28*`.

## What Has Already Been Tested

- Phase 20 tested short-horizon predictive crossing-state planning and did not improve crossings, recoverability, CAPTURE, or success.
- Phase 20.5 tested horizon, replan interval, score, and action-space ablations; all stayed at crossing `12`, recoverable `0`, CAPTURE `12`, success `12`.
- Phase 21 tested explicit orbital energy/angular-momentum shaping; it did not expand crossing or capture reachability.
- Phase 22 introduced Burn A -> coast -> Burn B and created insertion windows, but not recoverable crossings.
- Phase 23 and 24 upgraded Burn B with deterministic and precision insertion solvers; conversion remained zero.
- Phase 25 mapped the recoverability basin and found tangential velocity was the dominant crossing-state blocker under relaxed threshold analysis.
- Phase 26 showed local vt correction exists but crossing-state vt does not improve.
- Phase 27 showed timing synchronization does not improve crossing-state sync and Burn B timing has no leverage on crossing-producing cases.
- Phase 28 mapped trajectory families and found dead windows dominate: window existence is not useful orbital geometry.

## Eliminated Hypotheses

- PPO/reward tuning is not the active path for this 2D explicit-controller sequence.
- Local reactive action changes do not expand global reachability.
- Short-horizon predictive planning does not solve orbital-class transition.
- Naive energy/angular-momentum shaping alone does not create useful crossings.
- Burn B brute force, precision insertion, vt correction, and timing synchronization are not sufficient.
- Relaxing thresholds can classify some states as recoverable, but does not improve real CAPTURE count.

## Controller Phases

- Controller/performance phases: 20, 21, 22, 23, 24, 26, 27, and Phase 29.
- Phase 20.5 is a controller ablation around Phase 20.
- Phase 22 is the first staged transfer architecture.
- Phase 23-24 modify Burn B only.
- Phase 26-27 modify insertion/corridor timing but preserve Burn A/coast concepts.

## Diagnostic Phases

- Phase 20.5: predictive-planner failure diagnosis.
- Phase 25: recoverability basin and threshold ablation.
- Phase 28: historical trajectory-family mapping.
- Phase 29 Part 0: repository audit and synthesis.

## True Performance Results

- Phase 7.6 remains the best broad 2D milestone in README: `217 / 270` successes in its local grid.
- On the reduced 48-case benchmark used by Phases 20-29, the repeated true performance result is crossing `12`, recoverable `0`, CAPTURE `12`, success `12`, overspeed usually `0`.
- Phase 22's insertion windows are a structural result, not an improved success result.

## Structural Negative Findings

- More horizon and more local candidates mostly increase coast selection and computation, not reachability.
- Window creation is not equivalent to useful crossing geometry.
- Burn B corrections can improve local quantities without changing crossing-state distributions.
- The current evidence points upstream of Burn B: early trajectory-family selection.

## README Narrative Files To Emphasize

- `README.md` for the Phase 7.6 milestone.
- `analysis/phase76_soft_hybrid/phase76_summary.md` for the best historical 2D controller milestone.
- `analysis/phase20_5_ablation/summary.md` for the local predictive-planner boundary.
- `analysis/phase22_two_burn_transfer/summary.md` for window creation.
- `analysis/phase25_recoverability_basin_mapping/summary.md` for basin structure.
- `analysis/phase28_trajectory_family_mapping/summary.md` for the current manifold-level conclusion.
- `analysis/phase29_repo_audit_and_family_selector/summary.md` after this run.

## Redundant Or Lower-Emphasis Files

- Repeated plot variants inside intermediate phase folders are useful evidence but should not dominate README narrative.
- Phase 23/24 Burn B solver plots are diagnostic negative results; emphasize their summaries rather than every candidate plot.
- Phase 26/27 local correction plots should be framed as negative structural evidence, not as performance progress.
- Untracked generated phase directories should be preserved, not reorganized or deleted.