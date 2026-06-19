# Phase38D Feature Importance

Scope: descriptive statistical mining from existing CSV fields only. Importance here means evidence usefulness for separating `crossing_producing`, `near_crossing`, and `over_conservative_transfer` classes. It is not a machine-learning model.

## Strongest Descriptive Features

1. `r0_over_target`
   - Strongest class-conditioning feature in the reduced benchmark.
   - Useful for analysis, not a direct controller variable.

2. `best_crossing_potential`
   - Strong effect-size separation between crossing-producing, near-crossing, and over-conservative-transfer rows.
   - Must remain diagnostic because Phase36C showed it can move without actual crossings.

3. `closest_approach_step`
   - Strong timing separation between near-crossing and over-conservative-transfer failure signatures.
   - Does not by itself justify coast-duration or radial-timing implementation.

4. `best_post_cross_distance`
   - Strongly separates crossing/recoverability outcomes.
   - Most useful for post-cross interpretation, not upstream variable selection.

5. `min_abs_radius_error_ratio`
   - Useful for proximity analysis.
   - Not sufficient as success evidence.

## Weak Or Contradicted Implementation Features

- `commit_timing`: directly tested in Phase37A; zero new crossings.
- `radial_magnitude`: directly tested in Phase37A; medium magnitude degraded crossings.
- weak tangential setting: directly tested in Phase37B; zero selected crossings and poor regression preservation.

## Unknown Features

- `coast_duration`: not isolated in existing CSV evidence.
- angular momentum correction: not isolated in existing CSV evidence.

## Paper-Safe Interpretation

The mined evidence supports feature discovery and hypothesis registration. It does not yet support new controller implementation.
