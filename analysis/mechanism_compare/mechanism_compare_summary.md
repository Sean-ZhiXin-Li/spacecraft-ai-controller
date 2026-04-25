# Mechanism Compare Summary

## Cases

- `validated_success`: r0 `1.00005`, dt `100`, success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, phase_durations `{'DESCENT': 48276, 'CAPTURE': 17, 'LOCK': 203}`
- `near_boundary_failure`: r0 `1.00006`, dt `100`, success `False`, crossings `0`, first_crossing_step `None`, final_radius_error `4.401e+07`, phase_durations `{'DESCENT': 100000, 'CAPTURE': 0, 'LOCK': 0}`
- `dt_induced_failure`: r0 `1.00005`, dt `200`, success `False`, crossings `0`, first_crossing_step `None`, final_radius_error `5.417e+08`, phase_durations `{'DESCENT': 100000, 'CAPTURE': 0, 'LOCK': 0}`

## Mechanism Findings

- The successful rollout reaches the phase transition sequence and spends only a short time in capture before low-authority lock.
- The near-boundary failure does not reach the same crossing/capture sequence within the budget, so the controller remains in the descent mechanism too long.
- The dt-induced failure shows that the controller behavior is coupled to the numerical integration step, not only to the geometric initial condition.

## Artifacts

- `E:/spacecraft_ai_project/analysis/mechanism_compare/success_vs_failure_radius.png`
- `E:/spacecraft_ai_project/analysis/mechanism_compare/success_vs_failure_vr.png`
- `E:/spacecraft_ai_project/analysis/mechanism_compare/success_vs_failure_energy.png`
- `E:/spacecraft_ai_project/analysis/mechanism_compare/success_vs_failure_action_norm.png`
- `E:/spacecraft_ai_project/analysis/mechanism_compare/phase_duration_table.json`