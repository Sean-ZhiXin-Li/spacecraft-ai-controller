# Residual Explicit IL Result

## Main Answer

- Does the residual policy preserve explicit-controller success? `Yes`
- Does it improve final radius error? `No`
- Does it improve tail radial velocity? `No`
- Does it destabilize the controller? `No`
- What does this imply about hybrid control vs learning-only control? Hybrid control preserves the verified explicit structure in this minimal zero-residual test, unlike learning-only policies that had to reproduce the full insertion behavior from scratch.

## Metrics

- alpha `0.2`
- residual success `True` vs explicit `True`
- residual crossings `1` vs explicit `1`
- residual first_crossing_step `48269` vs explicit `48269`
- residual final_radius_error `2.766e+04` vs explicit `2.766e+04`
- residual tail_mean_abs_vr `4.842e+01` vs explicit `4.842e+01`
- mean_abs_residual `0.000e+00`
- max_abs_residual `0.000e+00`