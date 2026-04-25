# Final Phase 3 Audit v2

## Strong Conclusions

- The explicit controller is a valid local 2D insertion solution at the validated baseline: `dt=100`, `r0_over_target=1.00005`, `max_steps=100000`, `thrust_scale=10000`.
- The controller's success depends on reaching the crossing-triggered phase transition. In mechanism v2, the successful case enters `CAPTURE` and `LOCK`, while the nearest boundary failure and the `dt=200` failure remain in `DESCENT` for the full budget.
- The validated `dt=100` boundary is narrow: success is first lost at `r0_over_target=1.00006`, immediately above the validated `1.00005` point.
- Success is timestep-sensitive and not monotonic across `dt`. The v2 maps show failure bands and re-entry of success pockets rather than a smooth stability basin.
- Learning-only and naive residual results should be interpreted as structure failures: they do not reliably preserve the long descent, event transition, capture, and lock sequence.

## Weak Or Local Conclusions

- The phase-map conclusions are local to this 2D environment, this explicit controller, this target-radius setup, and the listed `dt` / `r0_over_target` grids.
- Strict success does not always imply a sampled radius sign crossing. Some successful grid points have `radius_crossings_total=0`, likely because the trajectory enters the tolerance window without a sampled sign flip.
- The refined success pockets around `dt=130..150` are evidence of numerical/controller coupling, not proof of broad controller robustness.
- The residual-learning conclusions apply to the tested residual constructions and conservative rollout gates, not to all possible structured hybrid controllers.

## Remaining Ambiguity

- The stability boundary is still only sampled on finite grids. There may be narrow sub-grid pockets or failures between tested points.
- The current plots can visually overemphasize discrete cells. They should be read as sampled experiment maps, not continuous stability surfaces.
- The role of Euler integration error versus controller logic is not fully separated. The `dt` sensitivity indicates coupling, but not a full numerical analysis.
- The project still lacks systematic perturbation robustness at the v2 stage because heavy Monte Carlo was intentionally excluded.

## Documentation Cleanup Suggestions

Primary reading order:

1. `README.md`
2. `analysis/final_phase3_summary_v2.md`
3. `analysis/phase_map_v2/boundary_refine_summary_v2.md`
4. `analysis/mechanism_compare_v2/mechanism_compare_summary_v2.md`
5. `analysis/residual_explicit_magnitude_only_result.md`
6. `analysis/next_stage_recommendation.md`

Historical only:

- `analysis/overnight_deep_dive_summary.md`
- `analysis/final_phase3_summary.md`
- `analysis/project_audit_phase3.md`
- `analysis/project_audit_phase3_extended.md`
- `analysis/project_audit_phase3_final.md`
- older `analysis/WEEK*`, `analysis/NEW_WEEK_*`, and early project-log notes

These historical files are useful context, but the v2 artifacts are the cleanest Phase 3 characterization trail.

## Low-Risk Cleanup Applied

- Added v2 scripts and v2 output directories instead of overwriting old outputs.
- Updated README with a small v2 artifact section.
- Left physics, controller logic, PPO, and learning experiments unchanged.
