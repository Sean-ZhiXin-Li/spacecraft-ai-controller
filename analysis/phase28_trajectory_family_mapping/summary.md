# Phase 28 Trajectory Family Quality Mapping

## Scope

- CSV-first analysis using Phase 22, 23, 24, 25, 26, and 27 outputs.
- No controller reruns and no changes to physics, thresholds, CAPTURE, LOCK, reward, or prior outputs.
- Burn-A-end state is only filled when prior phases recorded a compatible pre-burn state JSON; otherwise it is left blank.

## Dataset

- Normalized rows: `576`.
- Unique initial cases: `48`.
- Crossing rows: `144`.
- Window rows: `132`.
- Family label counts: `{"crossing_bad_sync": 86, "dead_geometry": 300, "near_recoverable_crossing": 58, "window_no_crossing": 132}`.
- Window quality counts: `{"dead_window_no_crossing": 132, "no_window": 444}`.
- Best-controller family counts by case: `{"crossing_bad_sync": 6, "dead_geometry": 36, "near_recoverable_crossing": 6}`.

## Controller Summary

| Controller | Rows | Crossings | Windows | Good windows | Dead windows | Mean quality | Mean sync |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_soft_linear_3e4` | 48 | 12 | 0 | 0 | 0 | 2.3139 | 3.4885 |
| `phase22_two_burn_transfer` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase23_windowed_insertion_solver` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase24_precision_insertion_geometry` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase26_burn_hold_burn` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase26_stronger_tangential` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase26_two_step_corridor` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase26_vt_aware_scoring` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase27_adaptive_sync_corridor` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase27_delayed_sync_burn` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase27_predicted_cross_vt_targeting` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |
| `phase27_split_phase_sync_burn` | 48 | 12 | 12 | 0 | 12 | 2.3916 | 3.5336 |

## Research Answers

1. Are window-producing cases and crossing-producing cases the same family? `no` under current data; window-producing rows are mostly `dead_window_no_crossing`.
2. Which early trajectory features best predict useful crossing? Best numeric correlation with quality is `sync_error_at_crossing` at `-0.803`; this is correlation, not proof.
3. Do insertion windows mostly represent good windows or dead windows? `dead windows`: good `0`, dead `132`.
4. Is Burn A selecting the wrong trajectory family? `likely yes` for window-producing transfer rows because windows are usually not paired with useful crossings.
5. Is the current architecture near a 2D explicit-control ceiling? `plausibly yes`; Phase 27 shows Burn B timing has no leverage on crossing-producing cases.
6. What should Phase 29 test? A Burn-A family selector that targets crossing-producing initial manifolds before window generation, with instrumentation at Burn A end.

## Candidate Family

- Candidate useful-family rule from observed useful rows: `angle 150.0-175.0, r0 1.000-1.000`.
- This is a descriptive range from historical rows, not a validated controller rule.

## Honest Limitations

- Burn A end geometry is mostly missing from prior CSVs, so Phase 28 cannot prove a causal Burn A rule.
- Phase 25 provides reconstructed crossing-state geometry for Phase 22-24, but Phase 22/23 insertion-window states remain uninstrumented.
- The strongest conclusion is family separation: window existence is not equivalent to useful crossing geometry.

## Artifacts

- `phase28_family_dataset.csv`
- `phase28_family_summary.csv`
- `family_energy_vs_angular_momentum.png`
- `family_initial_angle_vs_r0.png`
- `family_thrust_outcome.png`
- `crossing_vt_vs_vr_by_family.png`
- `sync_error_vs_predicted_crossing.png`
- `window_count_vs_quality.png`
- `feature_importance_or_correlation.png`