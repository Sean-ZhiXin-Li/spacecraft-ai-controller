# Recovery Action Branching Nonformal v0 Summary

## Structural Validity

- Four-branch common-state bundle: valid
- Branch records: 4
- Decision events: 30001
- Synthetic fixture: no
- Scope: one-case nonformal diagnostic
- Common branch-state hash: `8b017254a8db2584a6732bcd086447ba405cf949d9e932cf03e71543b2cdb898`
- Manifest hash: `e9cb96eae714bc0d8ed66d1a85f29baed2819d0d425a3ce9742b7e77ac236bad`

## Hazard Outcomes

- Overspeed triggered: 0
- Instability triggered: 0
- Unsafe state triggered: 0
- Invalid simulation: 0
- Invalid recovery evaluation: 0

## State And Task Recovery

- Target-radius crossing: 0
- Phase34-compatible recoverable crossing: 0
- Recovery Success v0: 0
- Final simulator success: 0

| Branch | Overspeed | Crossing | Recoverable crossing | Recovery Success v0 | Simulator success | Terminal | Recovery outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `zero_action_reference_v0` | `clear` | `False` | `False` | `False` | `False` | `recovery_horizon_exhausted` | `hazard_avoided_task_stalled` |
| `velocity_opposed_thrust_v0` | `clear` | `False` | `False` | `False` | `False` | `recovery_horizon_exhausted` | `hazard_avoided_task_stalled` |
| `tangential_error_correction_v0` | `clear` | `False` | `False` | `False` | `False` | `recovery_horizon_exhausted` | `hazard_avoided_task_stalled` |
| `explicit_abort_v0` | `clear` | `False` | `False` | `False` | `False` | `explicit_recovery_abort` | `hazard_avoided_through_termination` |

## Cost And Intervention Burden

- `zero_action_reference_v0`: recovery_steps=10000, control_effort=0.0, delta_v_proxy=0.0, evaluations=10000, allows=10000, vetoes=0, intervention_rate=0.0, crossing_delay=None
- `velocity_opposed_thrust_v0`: recovery_steps=10000, control_effort=2500.0, delta_v_proxy=2770083.1024925155, evaluations=10000, allows=10000, vetoes=0, intervention_rate=0.0, crossing_delay=None
- `tangential_error_correction_v0`: recovery_steps=10000, control_effort=2500.0, delta_v_proxy=2770083.1024925155, evaluations=10000, allows=10000, vetoes=0, intervention_rate=0.0, crossing_delay=None
- `explicit_abort_v0`: recovery_steps=0, control_effort=None, delta_v_proxy=None, evaluations=0, allows=0, vetoes=0, intervention_rate=None, crossing_delay=None

## Failure Mechanisms

- `zero_action_reference_v0`: `recovery_horizon_exhausted`
- `velocity_opposed_thrust_v0`: `recovery_horizon_exhausted`
- `tangential_error_correction_v0`: `recovery_horizon_exhausted`
- `explicit_abort_v0`: `explicit_abort`

## Non-Claims

This one-case diagnostic does not establish formal safety, universal recovery, controller superiority, benchmark-wide effectiveness, cross-case generalization, hardware validity, deployment readiness, or cross-embodiment validation. A complete bundle is only structurally comparable and does not imply recovery success.
