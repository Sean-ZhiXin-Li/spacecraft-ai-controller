# Phase39 Logging Implementation Plan

Status: design only. Do not implement in this phase.

## Objective

Implement passive observability logging so future controller experiments can be interpreted scientifically. Phase39 logging must not change controller behavior, physics, thresholds, benchmark cases, or historical artifacts.

## Files Likely To Change

Candidate implementation files:

- `scripts/explicit_controller_phase34_post_cross_sync.py`
- `scripts/explicit_controller_phase36b_transfer_family_benchmark.py`
- `scripts/explicit_controller_phase37a_radial_commit_timing.py`
- `scripts/explicit_controller_phase37b_weak_tangential_subset.py`
- future Phase39 runner scripts, if approved later

Candidate shared helper:

- `scripts/rollout_observability.py`

The helper would compute passive summaries from existing rollout state/action histories. It should not import or modify controller logic.

Documentation to update after implementation:

- `docs/logging_schema_v2.md`
- `analysis/artifact_manifest.md`, only if new public artifacts are created
- phase-specific summary files for any future generated artifacts

## New Helper Module Proposal

`scripts/rollout_observability.py` should expose pure functions such as:

- `summarize_state_history(states, target_radius, mu, ...)`
- `compute_energy_summary(states, mu, reference_energy=None)`
- `compute_angular_momentum_summary(states, reference_h=None)`
- `summarize_actions(actions, dt, action_to_accel_context=None)`
- `build_phase_transition_log(phase_labels)`
- `closest_approach_snapshot(states, phase_labels, ...)`
- `crossing_snapshot(states, first_crossing_step, phase_labels, ...)`

All functions should be passive and deterministic.

## Backward Compatibility

Implementation must preserve existing CSV schemas unless a new output file is explicitly named schema-v2.

Allowed options:

1. Add companion files such as `phase39_observability_results_v2.csv`.
2. Append fields only in a new phase directory.
3. Keep historical Phase34/36/37 CSV files unchanged.

Not allowed:

- Rewriting historical CSVs.
- Changing existing field meanings.
- Changing benchmark thresholds.
- Changing `scripts/check_phase_results.py` expectations unless adding separate optional schema-v2 validation.

## Regression Risks

| Risk | Mitigation |
|---|---|
| Logging accidentally changes controller behavior | Keep observability helper pure; call after rollout or on copied data. |
| New fields are interpreted as success metrics | Mark all new fields diagnostic unless explicitly promoted in a future benchmark contract. |
| Unit ambiguity in energy/delta-v/work proxies | Document simulator units and label proxies clearly. |
| Schema drift breaks old analysis | Write schema-v2 outputs separately or append only in new phase outputs. |
| Historical artifacts are overwritten | Require new output directories for any future run. |
| Observability slows experiments | Add optional logging level and compact summaries instead of full trajectory dumps by default. |

## Validation Strategy

Before any future implementation is accepted:

1. Run existing regression guard:

```bash
python scripts/check_phase_results.py
```

2. Add unit tests for pure observability helper functions using synthetic states.

3. Verify that old CSV artifacts remain byte-identical when no new run is requested.

4. Verify that schema-v2 output is additive and contains required fields.

5. Validate that passive logging does not change crossing counts, recoverable crossing counts, overspeed, or instability on a small smoke run.

6. Run:

```bash
git diff --check
```

## Artifact Plan

Future implementation should write new observability artifacts under a new directory, for example:

- `analysis/phase39_observability/`

Potential files:

- `phase39_observability_results_v2.csv`
- `phase39_observability_summary.md`
- `schema_v2_field_coverage.md`
- `observability_validation.md`

Do not overwrite:

- `analysis/phase34_post_cross_sync/phase34_results.csv`
- `analysis/phase36b_transfer_family_benchmark/phase36b_results.csv`
- `analysis/phase36c_non_crossing_geometry_diagnosis/non_crossing_case_set.csv`
- `analysis/phase37a_radial_commit_timing/phase37a_results.csv`
- `analysis/phase37b_weak_tangential_subset/phase37b_results.csv`

## What Not To Change

- No controller implementation.
- No controller parameters.
- No physics.
- No thresholds.
- No benchmark case grid.
- No historical artifacts.
- No success definitions.
- No Phase39 controller search.

## Scientific Review

If these measurements existed, the following Phase38 unknowns would become more answerable:

- Whether non-crossing failures are energy-limited.
- Whether non-crossing failures are angular-momentum-limited.
- Whether closest approach occurs with incompatible velocity geometry.
- Whether coast duration is distinct from radial commitment timing.
- Whether weak tangential shaping failed because of angular-momentum mismatch, work direction, phase gating, or regression damage.
- Whether selected non-crossing cases differ from known crossing-producing regression cases before the first crossing opportunity.

Phase39 should therefore start with instrumentation and observability only. Controller implementation remains unapproved until observability identifies a source-backed variable with a registered hypothesis and regression guard.
