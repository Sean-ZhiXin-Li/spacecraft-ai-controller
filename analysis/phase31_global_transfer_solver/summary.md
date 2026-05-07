# Phase 31 Global Transfer Solver

## Scope

- Research context written to `research_context.md` before controller conclusions.
- Python-only 2D global transfer-family benchmark on the same reduced 48-case grid.
- Physics, CAPTURE/LOCK, thresholds, reward, PPO, coast, and Burn B concept are unchanged.
- Phase 31 searches bounded transfer families over burn timing, burn magnitude, burn direction, and coast duration.
- Transfer candidates mapped: `576`. Selected transfer previews with crossing: `21`.

## Controller Results

| Controller | Crossings | Near recoverable | Recoverable | CAPTURE | Success | Windows | Good windows | Dead windows | Mean sync | Burn-A abs h | Burn-A abs E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase31_direct_transfer` | 8 | 1 | 0 | 8 | 8 | 16 | 1 | 15 | 3.4811 | 0.2532 | 0.4600 |
| `phase31_energy_ladder_transfer` | 8 | 1 | 0 | 8 | 8 | 16 | 0 | 16 | 8.0718 | 0.1666 | 0.3299 |
| `phase31_hohmann_like_transfer` | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 12 | nan | 0.0436 | 0.0163 |
| `phase31_lambert_like_transfer` | 8 | 0 | 0 | 8 | 8 | 25 | 0 | 25 | 6.0882 | 0.2374 | 0.5061 |
| `phase31_phase22_baseline` | 12 | 5 | 0 | 12 | 12 | 12 | 0 | 12 | 3.5336 | 0.5370 | 0.7408 |
| `phase31_phase30_best_baseline` | 12 | 5 | 0 | 12 | 12 | 18 | 0 | 18 | 3.5336 | 0.1203 | 0.1633 |

## Research Answers

1. Does global transfer improve crossings? `no`. Best crossing count: `12` from `phase31_phase22_baseline`.
2. Does it reduce dead windows? `no`. Best dead-window count: `12` from `phase31_hohmann_like_transfer`.
3. Does it improve near-recoverable crossings? `no`. Best near-recoverable count: `5`.
4. Does it improve recoverable crossings? `no`.
5. Does it improve CAPTURE? `no`.
6. Which transfer family performs best? `phase31_phase22_baseline` by mean family quality.
7. Is architecture class the real bottleneck? `weakly suggested but not proven`.
8. What is Phase 32? If Phase 31 does not move crossing-state structure, Phase 32 should test true optimal control: direct collocation, MPC, or CasADi-based trajectory optimization.

## Success Criteria

- Minimum, global transfer families mapped: `met`.
- Moderate, dead windows reduced: `not met`.
- Strong, crossing count improves: `not met`.
- Major, recoverable crossings improve: `not met`.
- Breakthrough, CAPTURE/success improves: `not met`.

## Honest Interpretation

- Best mean quality variant: `phase31_phase22_baseline`.
- Compared with Phase 30 endpoint search, global transfer improvement in recoverable/CAPTURE/success: `no`.
- Mean sync improvement without crossing/CAPTURE improvement: `yes`.
- This phase maps named transfer families and tests whether transfer class alters downstream crossing structure.
- If performance does not move, even global transfer family redesign does not solve the current orbital insertion structure.

## Artifacts

- `research_context.md`
- `phase31_results.csv`
- `phase31_transfer_dataset.csv`
- `transfer_family_map.png`
- `energy_h_transfer_space.png`
- `periapsis_apoapsis_transfer_map.png`
- `transfer_quality_distribution.png`
- `phase31_vs_phase30_comparison.png`
- `controller_comparison.png`
