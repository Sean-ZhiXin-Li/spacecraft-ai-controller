# Staged Recovery Instrumentation v0

## 1. Status

Observation schema and pure derivation layer implemented; runtime logger integration and staged recovery execution not implemented.

Completed: 2026-07-28

> This document defines a staged-recovery observation schema and pure orbital derivation layer only. It does not execute a simulator transition, integrate a runtime logger, implement a recovery phase action, select a phase, authorize a staged rollout, demonstrate task recovery, or support formal safety claims.

## 2. Purpose

Staged Recovery Architecture v0 requires status-bearing state, hazard, recoverability, progress, action, phase, and provenance evidence. The published one-case recovery log was sufficient to validate outcomes but omitted the per-step Cartesian and orbital quantities needed for physical mechanism diagnosis. This layer defines those fields and derives the supported quantities from explicitly supplied inputs without running a simulator.

The implementation is import-pure and deterministic. It accepts state and configuration values, returns immutable records, and has no rollout, transition, controller, logger, or file-writing behavior.

## 3. Evidence Basis

The layer reuses these repository sources rather than introducing new physics or thresholds:

- `analysis/staged_recovery_architecture_v0/architecture_manifest.json` and `runtime_assurance/staged_recovery_contract.py` define the 52 architecture signal IDs, phase identities, evidence statuses, and frozen adverse-stop priority.
- `analysis/recovery_action_branching_nonformal_v0/manifest.json` and `runtime_assurance/recovery_branch_executor.py` define the inertial Cartesian frame and positive tangential orientation.
- `scripts/explicit_controller_phase21_orbital_transfer_planner.py` defines the target circular speed, signed target errors, orbital components, and declared specific-energy diagnostic.
- `runtime_assurance/recovery_experiment_runner.py` defines the current realized speed ratio and ratio-denominator semantics.
- `runtime_assurance/final_veto_monitor.py` defines the strict `speed_ratio > 1.90` hazard boundary.
- `runtime_assurance/recovery_evaluators.py` defines the Phase34-compatible inclusive recoverability component bounds.
- `scripts/explicit_controller_phase34_post_cross_sync.py` defines the discrete signed-radius crossing rule.
- `simulator/phase34_35_transition.py` defines the existing component action limit and current softened gravity transition semantics. This module does not call it.

## 4. Relationship to Staged Recovery Architecture v0

The architecture contract declares what evidence a future state machine needs. This instrumentation contract declares how each field is represented and which values can be derived from supplied evidence. It does not evaluate phase guards or make phase decisions.

The source architecture commit is `0d416603027e8a27991baf4f89445f6f466b86e6`; its canonical manifest hash is `22fa7e0f01c7836ecb1f10838ef00c4cafa937d212bba579fffb25e2c8f11971`.

## 5. Raw Cartesian State Contract

`CartesianState2D` preserves the repository state order:

```text
(x, y, vx, vy)
```

Position uses metres and velocity uses metres per second. Every component must be a finite numeric value; booleans are not numeric state values. A missing state remains `not_evaluated`. A malformed state becomes `invalid`. No default state is invented, and no next state is computed.

The record can also preserve supplied recovery step, total transition count, simulation time, phase IDs, branch-state hash, case ID, seed, simulator-configuration hash, constants hash, and implementation commit.

## 6. Evidence-Status Semantics

Every `InstrumentedValue` carries:

- `value`
- `status`
- `reason`
- `units`
- `source_id`
- `source_step`
- `valid`
- sorted input-source IDs for derived evidence

The statuses are `measured`, `derived`, `one_step_predicted`, `multi_step_predicted`, `heuristic`, `not_evaluated`, and `invalid`, matching the architecture vocabulary.

`not_evaluated` means evidence was unavailable or unsupported. `invalid` means supplied evidence was malformed, contradictory, nonfinite, or outside the declared contract. Both carry a null value and `valid=false`. Unknown numeric values are not zero, and unknown booleans are not false. Predicted values remain separate from measured values.

## 7. Coordinate and Tangential Convention

The frame is two-dimensional inertial Cartesian. For position `r=(x,y)`:

```text
e_r = (x / ||r||, y / ||r||)
e_t = (-e_r_y, e_r_x)
```

`e_t` is a positive counterclockwise 90-degree rotation of `e_r`. The convention matches the frozen branch manifest and executor. No clockwise alternative is inferred.

## 8. Orbital Basis

For a finite state with strictly positive position norm, the pure basis derivation returns:

```text
radius = sqrt(x^2 + y^2)
speed_magnitude = sqrt(vx^2 + vy^2)
radial_unit_vector = e_r
tangential_unit_vector = e_t
```

Zero position norm is invalid. The implementation does not normalize a zero vector or derive geometry from a state hash.

## 9. Radial and Tangential Velocity

Signed velocity components are:

```text
radial_velocity = dot((vx, vy), e_r)
tangential_velocity = dot((vx, vy), e_t)
```

Negative radial velocity remains valid inward motion. Negative tangential velocity remains valid motion opposite the positive tangential basis. Absolute values are used only by separately named recoverability comparisons.

## 10. Target-State Errors and Ratios

With explicit positive `mu` and target radius:

```text
signed_target_radius_error = radius - target_radius
absolute_target_radius_error = abs(signed_target_radius_error)
radius_error_ratio = signed_target_radius_error / target_radius
target_circular_speed = sqrt(mu / target_radius)
radial_velocity_ratio = radial_velocity / (target_circular_speed + 1e-12)
tangential_velocity_error = tangential_velocity - target_circular_speed
tangential_velocity_error_ratio = tangential_velocity_error / (target_circular_speed + 1e-12)
```

Signed and absolute errors remain distinct. Invalid or nonpositive configuration values produce invalid evidence. The implementation does not guess a target, gravitational parameter, or denominator.

## 11. Speed Ratio and Overspeed Headroom

The realized ratio follows the full-horizon recovery runner:

```text
realized_speed_ratio = speed_magnitude / (target_circular_speed + speed_ratio_denominator_epsilon)
overspeed_headroom = 1.90 - realized_speed_ratio
overspeed = realized_speed_ratio > 1.90
```

Positive headroom is below this declared threshold, zero is exactly on it, and negative headroom is above it. Exactly `1.90` is not overspeed because the comparator is strict `>`. This headroom is not a formal safety margin.

When an explicit predicted state is supplied, predicted ratio, headroom, and overspeed status carry predicted evidence statuses. They do not replace realized values.

## 12. Specific Orbital Energy

Phase21 declares this diagnostic:

```text
specific_orbital_energy = 0.5 * speed_magnitude^2 - mu / (radius + 1e-12)
target_circular_specific_energy = -mu / (2 * target_radius)
specific_energy_error = specific_orbital_energy - target_circular_specific_energy
```

The value has units `J/kg` and is never multiplied by spacecraft mass. Under the current Phase34/35 transition, gravity uses a softened denominator, so this quantity is classified as a `declared_diagnostic_proxy`, not an exact conserved invariant of that softened discrete model. It is not fuel use or delta-v. Unsupported gravity-model IDs yield `not_evaluated`.

## 13. Phase34-Compatible Recoverability Components

The adapter delegates the combined decision to `evaluate_phase34_compatible_recoverability` and returns all components separately:

```text
abs(radius_error_ratio) <= 0.0025
abs(radial_velocity_ratio) <= 0.02
abs(tangential_velocity_error_ratio) <= 0.25
```

All boundaries are inclusive. All three components must be available and valid before the combined predicate can be true. A missing component returns `not_evaluated`; an invalid component returns `invalid`. Crossing, simulator success, Recovery Success v0, and hazard avoidance remain separate.

## 14. Crossing-Event Derivation

Crossing requires two explicit measured states. It follows the Phase34 signed-error rule:

```text
(previous_error > 0 and current_error <= 0)
or
(previous_error < 0 and current_error >= 0)
```

Both outside-to-inside and inside-to-outside crossings are retained with direction labels. A previous state exactly on target does not start a new crossing; a current state exactly on target can complete a crossing when the previous error was nonzero. Crossing at the branch step is recovery-eligible; crossing before it is not. No fractional interpolation is produced.

## 15. Threshold-Free Progress Samples

Given previous and current derived observations, the module returns current-minus-previous deltas for signed and absolute radius errors, radial velocity, tangential error, speed ratio, headroom, declared energy proxy, and each recoverability ratio. Transition-count and elapsed-time deltas are also supported when explicitly supplied.

The sample does not classify `progressing`, `stalled`, or `regressing`. It has no no-progress window, meaningful-improvement threshold, or combined progress score. Desired direction remains component-specific and policy-dependent.

## 16. Action Geometry

For explicit proposed and executed two-component actions, the module can derive magnitude, radial and tangential components, component saturation margin, exact proposed/executed equality, and suppression status. Radial and tangential action components require a valid current basis.

The module never generates an action. A branch name is not action geometry. Explicit abort carries no physical action, rather than a fabricated `(0,0)` action. A rejected action remains distinct from a physical zero action. Normalized action magnitude is not delta-v.

## 17. Phase and Provenance Fields

The schema can preserve current phase, previous phase, dwell count, transition count, raw transition reason, recent phase history, no-progress status, handoff readiness, retreat status, and explicit-abort status. These are supplied externally or remain `not_evaluated`.

The layer does not select phases, evaluate phase guards, switch controllers, generate phase actions, or execute transitions.

## 18. Architecture Field Coverage

All 52 architecture signals have catalog entries and an explicit coverage classification:

| Classification | Count |
| --- | ---: |
| Direct input supported | 16 |
| Pure current-state derivation supported | 14 |
| Requires previous state | 8 |
| Requires predicted state | 3 |
| Requires runtime phase integration | 9 |
| Requires future evaluator | 1 |
| Not yet supported | 1 |

The catalog contains 105 fields after adding provenance, component, progress, action-geometry, and canonical-record fields. This means every architecture field is represented by the pure schema. It does not mean every field is measured or logged at runtime. Stage 0A validates zero newly integrated runtime fields.

The nine runtime-integration fields are the two horizon-remaining values, two horizon-exhaustion values, no-progress status, phase dwell count, phase transition count, recent phase history, and phase-transition reason. Handoff readiness requires a future evaluator. Available correction authority remains unsupported.

## 19. Canonical Serialization

Records, field definitions, and manifests use UTF-8 JSON with sorted object keys, stable separators, deterministic list order, and finite JSON numbers. Catalog field order is explicit. Sets, object representations, memory addresses, and nonfinite JSON numbers are prohibited.

The record self-hash is excluded from its own payload. An optional volatile provenance timestamp is also excluded from the scientific record hash. Payload mutation invalidates the hash.

## 20. Missing and Invalid Evidence

Missing state, previous state, predicted state, action, phase evidence, evaluator output, or configuration remains explicit. Missing evidence cannot authorize a favorable state. Invalid evidence retains a reason and propagates through dependent derivations. Predicted evidence cannot overwrite measured evidence, and unsupported values are not synthesized.

## 21. Current Limitations

- No runtime logger calls this layer.
- No newly authorized trajectory has demonstrated record completeness.
- Handoff readiness and available correction authority lack frozen evaluators.
- Crossing direction prediction and crossing proximity require explicit predicted evidence.
- No phase action law, numerical guard, no-progress threshold, or hysteresis parameter is frozen.
- The energy field is a declared diagnostic proxy under the softened transition, not an exact invariant.
- Pure field availability does not demonstrate that a staged policy can recover the frozen state.

## 22. Stage 0B Runtime Logger Integration Boundary

The next smallest milestone is an import-safe, bounded logger adapter tested only with synthetic records and explicitly supplied states. Before any live trace is authorized, Stage 0B must prove complete field capture, deterministic event ordering, no physics changes, no controller changes, no phase actions, protected-path refusal, and missing/invalid propagation. It must not authorize staged recovery execution by itself.

## 23. Claim Restrictions

This instrumentation layer does not establish runtime completeness, task recovery, controller effectiveness, optimal phase logic, formal safety, hardware validity, deployment readiness, or cross-domain validity. Staged recovery execution remains `not_authorized` because runtime logger integration, phase action laws, numerical guards, no-progress thresholds, hysteresis parameters, and an execution path are not frozen and validated.
