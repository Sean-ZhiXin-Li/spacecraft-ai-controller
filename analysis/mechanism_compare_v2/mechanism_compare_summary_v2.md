# Mechanism Compare v2

## Cases

- `validated_success`: r0 `1.00005`, dt `100`, success `True`, first_crossing_step `48269`, crossings `1`, CAPTURE entered `True`, LOCK entered `True`, final_radius_error `2.766e+04`.
- `nearest_dt100_boundary_failure`: r0 `1.00006`, dt `100`, success `False`, first_crossing_step `None`, crossings `0`, CAPTURE entered `False`, LOCK entered `False`, final_radius_error `4.401e+07`.
- `dt_induced_failure`: r0 `1.00005`, dt `200`, success `False`, first_crossing_step `None`, crossings `0`, CAPTURE entered `False`, LOCK entered `False`, final_radius_error `5.417e+08`.

## Technical Interpretation

- The validated success spends `48276` steps in DESCENT, `17` in CAPTURE, and `203` in LOCK.
- Both failure cases spend the full budget in DESCENT and never enter CAPTURE or LOCK.
- The nearest dt=100 failure at r0 `1.00006` misses the crossing-triggered phase transition even though it differs from the validated start by only `1e-5` in r0_over_target.
- The dt-induced failure at dt `200` shows that the mechanism is coupled to numerical step size: changing dt can prevent the controller from reaching the crossing/capture event at the same initial radius.
- The key mechanism is therefore event access, not lock tuning: if DESCENT does not reach the crossing event, CAPTURE and LOCK never become active.

## Plot Notes

- Time is normalized to simulation days on every plot.
- Radius error and radial velocity expose whether descent is approaching the target band.
- Energy and angular momentum show the long retrograde-removal phase versus failure trajectories that do not trigger capture.