# Multi-Case Recovery Branch-State Registry v0

Status: Multi-case recovery branch-state registry frozen and validated; Stage 1B multi-case source-state prerequisite satisfied.

Completed: 2026-08-03

This registry contains complete, provenance-bound branch-point states produced by deterministic execution of existing frozen nominal-prefix behavior. The states were not reconstructed from incomplete logs, manually authored, or created by perturbing the legacy canonical state. Registry membership enables multi-case bounded recovery and shadow-runtime experiments, but does not demonstrate recovery performance, controller improvement, phase-policy validity, formal safety, or deployment readiness.

## Status

Four deterministic, executable members are registered.

## Original Stage 1B blocker

The previous single-state executor could not satisfy a four-case trace contract.

## Purpose

Freeze complete multi-case branch-point inputs without executing a recovery branch.

## Legacy canonical member

The published canonical artifact remains external and byte-identical.

## Source-case inventory

All 13 Final Veto cases were provenance-complete and eligible.

## Prefix extraction contract

Each case uses its repository-backed boundary. The legacy member remains after transition 27; generated members use the last valid state before their frozen monitor-off terminal transition.

## Candidate discovery

Twelve noncanonical cases were executed once for deterministic discovery.

## Frozen selection rules

Closest below, closest above, and strongest remaining tangential challenge were selected at each case's own frozen boundary without post-result rule changes. These are multi-boundary calibration inputs, not synchronized-time samples.

## Selected members

- Member A: `phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_150__thrust_8000`
- Member B: `phase34_known_recoverable_preservation_v1__r0_1p00__angle_175__thrust_8000` at transition `4792` and predicted ratio `1.0000000000000073`
- Member C: `phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_175__thrust_10000` at transition `26` and predicted ratio `1.9121328346751392`
- Member D: `phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_170__thrust_10000` at transition `25` and tangential error ratio `0.3807784431541385`

## Cartesian state completeness

All four members provide finite x, y, vx, and vy state values.

## Provenance

Every generated member binds source case, configuration, simulator, constants, transition, controller, action-trace, and state-trace hashes.

## Determinism validation

Each generated member exactly matched an independent fresh reproduction.

## Canonical reproduction

The legacy canonical payload hash and complete document reproduced exactly.

## Registry loader

Loading is member-ID based, immutable, hash-validating, and path constrained.

## Executor compatibility

The default executor path remains legacy-canonical; registry execution uses a separate validated member-ID entry point.

## Protected evidence

All protected aggregate hashes were recorded read-only and remained unchanged.

## Scientific limitations

This is input generation, not recovery or policy evidence.

## Stage 1B readiness

The four-case source-state prerequisite is satisfied.

## Next aggressive milestone

Resume Stage 1B: Staged Recovery Shadow Guard Runtime and Calibration Trace Set v0.
