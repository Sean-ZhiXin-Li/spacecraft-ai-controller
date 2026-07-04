# Logging Schema V2 Proposal

Status: design proposal only. This schema is not implemented yet and must not be used to reinterpret historical artifacts.

Goal: add passive observability fields that improve scientific interpretation without changing physics, controllers, thresholds, or historical outputs.

## Design Rules

- Existing fields remain backward compatible.
- New fields are appended or written to a companion schema-v2 artifact.
- Calculations must use existing rollout state, action, time-step, and simulator constants.
- Diagnostic fields must not become success criteria unless separately justified.
- Historical CSVs must not be overwritten.

## Proposed Fields

| Name | Type | Units | Calculation | Source | Expected use | Required / optional | Backward compatibility note |
|---|---|---|---|---|---|---|---|
| `schema_version` | string | unitless | Constant, e.g. `v2`. | Logging wrapper | Distinguish new logs from historical artifacts. | Required | Additive metadata. |
| `run_id` | string | unitless | Stable generated ID from phase, case, controller, and timestamp or explicit run label. | Logging wrapper | Traceability. | Required | Additive metadata. |
| `source_phase` | string | unitless | Phase name, e.g. `phase39_observability`. | Logging wrapper | Provenance. | Required | Additive metadata. |
| `specific_orbital_energy_initial` | float | simulator energy units per mass | Compute from initial position/velocity and gravitational parameter used by simulator. | Passive state summary | Identify initial energy regime. | Required if state and mu available | New field; does not affect existing metrics. |
| `specific_orbital_energy_final` | float | simulator energy units per mass | Same calculation at final state. | Passive state summary | Final energy diagnosis. | Required if state and mu available | Additive. |
| `specific_orbital_energy_at_closest` | float | simulator energy units per mass | Same calculation at closest approach step. | Closest-approach snapshot | Determine whether closest approach had useful energy. | Required if state and mu available | Additive. |
| `specific_orbital_energy_at_crossing` | float or blank | simulator energy units per mass | Same calculation at first crossing, blank if no crossing. | Crossing snapshot | Handoff state diagnosis. | Optional for non-crossing rows | Blank-compatible. |
| `energy_error_initial` | float | simulator energy units per mass | Initial specific energy minus explicit target/reference specific energy. | Passive state summary | Energy-limit analysis. | Optional until reference definition is approved | Requires documented reference. |
| `energy_error_at_closest` | float | simulator energy units per mass | Closest-approach energy minus target/reference energy. | Closest-approach snapshot | Explain closest-approach failures. | Optional until reference definition is approved | Additive. |
| `energy_error_final` | float | simulator energy units per mass | Final energy minus target/reference energy. | Passive state summary | End-state diagnosis. | Optional until reference definition is approved | Additive. |
| `angular_momentum_initial` | float | simulator length^2 / simulator time | Magnitude of 2D specific angular momentum from initial state. | Passive state summary | Initial geometry diagnosis. | Required if state available | Additive. |
| `angular_momentum_final` | float | simulator length^2 / simulator time | Same calculation at final state. | Passive state summary | Final geometry diagnosis. | Required if state available | Additive. |
| `angular_momentum_at_closest` | float | simulator length^2 / simulator time | Same calculation at closest approach. | Closest-approach snapshot | Explain failure at closest approach. | Required if state available | Additive. |
| `angular_momentum_at_crossing` | float or blank | simulator length^2 / simulator time | Same calculation at first crossing. | Crossing snapshot | Handoff-state diagnosis. | Optional for non-crossing rows | Blank-compatible. |
| `angular_momentum_error_at_closest` | float | simulator length^2 / simulator time | Closest angular momentum minus explicit target/reference angular momentum. | Closest-approach snapshot | Determine angular-momentum mismatch. | Optional until reference definition is approved | Requires documented reference. |
| `eccentricity_estimate_initial` | float | unitless | Standard two-body eccentricity estimate from state and gravitational parameter, if valid in simulator units. | Passive state summary | Orbit-shape diagnosis. | Optional | Must document simplified-simulator assumption. |
| `eccentricity_estimate_at_closest` | float | unitless | Same estimate at closest approach. | Closest-approach snapshot | Failure-shape diagnosis. | Optional | Additive. |
| `cumulative_delta_v_proxy` | float | simulator velocity units | Sum of action/thrust-induced velocity increments if available; otherwise documented effort proxy. | Action summary | Control effort comparison. | Required if action-to-velocity mapping is available; otherwise optional | Must be labeled proxy if not physical delta-v. |
| `cumulative_radial_effort_proxy` | float | action or velocity proxy units | Sum radial component of applied action/velocity increment. | Action summary | Identify radial intervention strength. | Optional | Additive. |
| `cumulative_tangential_effort_proxy` | float | action or velocity proxy units | Sum tangential component of applied action/velocity increment. | Action summary | Identify tangential intervention strength. | Optional | Additive. |
| `radial_work_proxy` | float | simulator work proxy units | Passive dot product or documented proxy using radial force/action and radial displacement/velocity. | Action-state summary | Test whether radial effort performs useful work. | Optional | Must be labeled proxy unless physically calibrated. |
| `tangential_work_proxy` | float | simulator work proxy units | Passive dot product or documented proxy using tangential force/action and tangential displacement/velocity. | Action-state summary | Test angular/tangential mechanism. | Optional | Must be labeled proxy unless physically calibrated. |
| `phase_transition_log_json` | JSON string | unitless | Ordered list of phase labels and transition steps. | Controller phase logger | Explain timing and handoff behavior. | Required when phase labels exist | Additive; can be blank for controllers without phase labels. |
| `time_in_phase_json` | JSON string | steps | Count steps spent in each controller phase. | Controller phase logger | Summarize phase behavior. | Required when phase labels exist | Additive. |
| `closest_state_json` | JSON string | mixed simulator units | State snapshot at closest approach: radius error, radial velocity, tangential velocity error, speed, phase, energy/angular momentum if available. | Closest-approach snapshot | Interpret why closest approach failed or crossed. | Required | Additive. |
| `crossing_state_json` | JSON string or blank | mixed simulator units | State snapshot at first crossing. Blank if no crossing. | Crossing snapshot | Handoff and recoverability analysis. | Required for crossing rows | Blank-compatible. |
| `pre_cross_summary_json` | JSON string | mixed simulator units | Min/mean/max/final-before-crossing summaries before crossing or closest approach. | State-history summary | Compare approach geometry. | Optional P1 | Additive. |
| `state_history_summary_json` | JSON string | mixed simulator units | Compact rollout summaries for radius error, radial velocity, tangential velocity error, speed, energy, angular momentum. | State-history summary | Interpret trajectory evolution without storing full trajectory. | Required | Additive. |
| `regression_guard_group` | string | unitless | Label such as `selected_non_crossing`, `known_crossing_guard`, or `full_benchmark`. | Benchmark runner | Protect known crossing cases. | Required | Additive. |
| `observability_notes` | string | unitless | Human-readable caveats for proxy fields. | Logging wrapper | Prevent overinterpretation. | Optional | Additive. |

## Minimal V2 Required Set

Minimum useful schema-v2 addition:

- `schema_version`
- `run_id`
- `source_phase`
- `specific_orbital_energy_initial`
- `specific_orbital_energy_at_closest`
- `angular_momentum_initial`
- `angular_momentum_at_closest`
- `cumulative_delta_v_proxy`
- `phase_transition_log_json`
- `time_in_phase_json`
- `closest_state_json`
- `state_history_summary_json`
- `regression_guard_group`

## Interpretation Rule

Schema V2 fields are observability fields. They do not change simulator outcomes and must not be used to redefine crossing, recoverable crossing, overspeed, instability, CAPTURE, LOCK, or simulator-defined success.
