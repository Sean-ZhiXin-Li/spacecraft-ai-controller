# Staged Recovery Guard Evidence Analysis v0

## Status

Stage 1A offline guard-evidence analysis completed; staged recovery execution remains unauthorized.

Analyzed: 2026-07-31

## Source Evidence

- Stage 0C result commit: `7844cc5824cf83dc84d8732e96d361d9f4b06aeb`
- Validation ID: `staged_recovery_instrumentation_validation_v0`
- Trace-manifest canonical hash: `23dd44711641eb2fcae9f1be81f405ee0660146862e8193bb8a0ebd871140680`
- Trace aggregate hash: `4f3d700422c47abf4ece93c0dd54770be5f3109a49ef49708485c41ca67e962e`
- Source events/transitions: `10` / `8`
- Branch/seed/prefix: `velocity_opposed_thrust_v0` / `0` / `27`

All source event hashes, ordering, equivalence evidence, field-completeness evidence, and aggregate trace identity validated before analysis.

## Signal Availability

The deterministic profile contains `209` numeric, `71` boolean, and `126` categorical/component profiles. `260` profiles contain valid evidence, `146` remain entirely unavailable, and `0` contain invalid evidence.

Observed minimum nonzero adjacent deltas are trace-resolution statistics only. They are not sensor-noise, process-noise, numerical-uncertainty, or guard-threshold estimates.

## Exact Inherited Guard Atoms

The analysis evaluates `14` exact inherited atoms. Realized overspeed uses strict `> 1.90`; clear uses `<= 1.90`. Phase34-compatible component checks preserve inclusive absolute bounds `0.0025`, `0.02`, and `0.25`.

Realized overspeed clear is true on `10` events and realized overspeed is true on `0` events. Predicted clear and realized clear remain separate.

The tangential recoverability component passes on `10` events. Radius and radial-velocity component pass counts are `0` and `0`; combined Phase34-compatible pass count is `0`.

Eligible crossing is observed on `0` events; no eligible crossing is true on `8` transition events.

## Threshold-Free Directional Evidence

The analysis evaluates `24` threshold-free directional/component atoms. Radius-gap improvement and target-directed radial motion are visible in the measured transitions. Tangential absolute error initially improves and later worsens after crossing zero. Overspeed headroom and the diagnostic absolute energy-proxy error improve over the checked path.

These atoms report direction only. They do not establish adequate magnitude, future crossing, recoverability, or phase readiness.

## Windowed Progress Evidence

All integer window lengths from one through eight realized transitions were evaluated. Window counts are: `1:8`, `2:7`, `3:6`, `4:5`, `5:4`, `6:3`, `7:2`, `8:1`.

Radius gap, radial recoverability magnitude, overspeed headroom, and diagnostic energy-proxy error improve across the full eight-transition window. Tangential-error direction is window-dependent because the signed error crosses zero. Crossing count remains zero.

No combined progress score, preferred window, minimum improvement, or stalled/progressing/regressing policy classification was created.

## Phase Observability

Fully observable phases: `0`. Partially observable phases: `6`. Future-evaluator-required phases: `3`. Implemented staged action laws: `0`. Authorized executable guards: `0`.

Hazard, radial, tangential, crossing, and recoverability evidence is structurally available to varying degrees. Instability, unsafe-state, handoff readiness, phase-runtime metadata, and correction authority remain unavailable or unsupported.

## Unresolved Parameters

All `9` no-progress and anti-chatter parameters remain unresolved: `NO_PROGRESS_WINDOW_LENGTH`, `NO_PROGRESS_MIN_RADIUS_GAP_IMPROVEMENT`, `NO_PROGRESS_MIN_RADIAL_COMPONENT_IMPROVEMENT`, `NO_PROGRESS_MIN_TANGENTIAL_COMPONENT_IMPROVEMENT`, `NO_PROGRESS_MIN_HEADROOM_IMPROVEMENT`, `NO_PROGRESS_REQUIRED_COMPONENT_COUNT`, `NO_PROGRESS_CONSECUTIVE_WINDOWS`, `NO_PROGRESS_MIN_PHASE_DWELL`, `NO_PROGRESS_COOLDOWN`.

No window length or minimum-improvement threshold is selected or authorized by this analysis.

## Strongest Supported Conclusion

The existing measured validation trace is sufficient to demonstrate that several hazard, kinematic-direction, recoverability, crossing, action, and component-wise progress guard atoms are deterministically observable and evaluable offline. It is not sufficient to select general numerical phase guards, no-progress thresholds, hysteresis parameters, action laws, or handoff criteria.

## Next Smallest Milestone

The next smallest evidence milestone is a predeclared Stage 1B hazard-arrest/stabilization observational trace set spanning repeated boundary conditions. It should estimate signal variability and evaluator availability without implementing a phase action or authorizing phase transitions.

## Claim Restrictions

This analysis does not establish recovery performance, phase-policy validity, false-positive or false-negative rates, general noise, safe hysteresis, optimal thresholds, controller superiority, formal safety, hardware validity, or deployment readiness.

Analysis-manifest canonical hash: `df96854e1caecd45560f8d7e78136bc751bbf3087eab78a35ee6a0051c5ba648`.
