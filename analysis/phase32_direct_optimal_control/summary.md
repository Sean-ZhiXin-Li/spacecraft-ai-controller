# Phase 32 Direct Optimal Control Baseline

## Scope

- Phase 32A coarse finite-horizon optimal-control prototype.
- CasADi-first dependency detection; SciPy direct-shooting fallback used when CasADi is unavailable.
- Dynamics preserve the project 2D gravity and thrust physics.
- No CAPTURE/LOCK, reward, threshold, or physics rewrite.
- CasADi status: `unavailable, fallback used: No module named 'casadi'`.
- Horizon: `512` physics steps with `64` control intervals.

## Objective Mode Results

| Objective mode | Solves | Crossings | Near recoverable crossing | Recoverable crossing | Recoverable state | Capture potential | Mean best sync | Mean best distance | Mean effort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `radius_only` | 4 / 4 | 0 | 0 | 0 | 0 | 0 | 12.5985 | 13.1636 | 0.00002 |
| `recoverability_target` | 4 / 4 | 1 | 1 | 1 | 2 | 1 | 3.9670 | 3.9670 | 0.01680 |
| `sync_error_minimization` | 4 / 4 | 2 | 1 | 1 | 1 | 1 | 5.6262 | 6.3847 | 0.00076 |
| `fuel_constrained_recoverability` | 4 / 4 | 0 | 0 | 0 | 2 | 0 | 4.0010 | 4.0010 | 0.00009 |

## Research Answers

1. Can optimal control outperform heuristic architectures? `yes`.
2. Can recoverable crossing be reached? `yes`.
3. Can CAPTURE improve? `not directly evaluated as a production CAPTURE rollout`; capture potential is reported instead.
4. Which objective formulation works best? `recoverability_target` by mean best recoverability distance.
5. Is recoverability theoretically reachable? `yes as a state in this coarse solve`.
6. Are prior failures mainly architecture failures? `plausibly`.
7. Does the current benchmark appear physically feasible? `partially`.
8. What should Phase 33 test? A real CasADi/IPOPT direct-collocation run after installing the declared CasADi dependency, then longer horizon or receding-horizon MPC.

## Success Criteria

- Minimum, stable optimal-control solve runs: `met`.
- Moderate, better sync / lower recoverability distance: `met`.
- Strong, near recoverable crossing: `met`.
- Major, first recoverable crossing: `met`.
- Breakthrough, CAPTURE/success improvement: `not evaluated`.

## Honest Interpretation

- This is an upper-bound prototype, not a production controller.
- CasADi was declared in repo environment files but unavailable in the checked runtime, so the run used SciPy direct shooting.
- If direct continuous optimization still cannot reach recoverability after a proper CasADi collocation run, the benchmark or thresholds may be structurally too hard.

## Artifacts

- `research_context.md`
- `phase32_results.csv`
- `phase32_trajectory_dataset.csv`
- `optimal_trajectory_examples.png`
- `state_error_vs_time.png`
- `control_profile.png`
- `sync_error_progression.png`
- `phase32_vs_phase31_comparison.png`
- `objective_mode_comparison.png`