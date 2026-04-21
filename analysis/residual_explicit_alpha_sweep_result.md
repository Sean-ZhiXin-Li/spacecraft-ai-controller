# Residual Explicit Alpha Sweep Result

## Main Answer

- What alpha range preserves explicit-controller success? `0.00 to 0.20`
- Does any nonzero alpha improve final radius error? `No`
- Does any nonzero alpha improve tail radial velocity? `No`
- How sensitive is the controller to residual authority? No measurable sensitivity in this sweep because the loaded residual policy outputs zero correction.
- What does this imply for safe hybrid learning? Start from verified zero-residual preservation, keep residual authority small, and require explicit success-preservation checks before training nonzero corrections.

## Sweep

- alpha `0.00`: success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, tail_mean_abs_vr `4.842e+01`, max_abs_residual `0.000e+00`
- alpha `0.02`: success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, tail_mean_abs_vr `4.842e+01`, max_abs_residual `0.000e+00`
- alpha `0.05`: success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, tail_mean_abs_vr `4.842e+01`, max_abs_residual `0.000e+00`
- alpha `0.10`: success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, tail_mean_abs_vr `4.842e+01`, max_abs_residual `0.000e+00`
- alpha `0.20`: success `True`, crossings `1`, first_crossing_step `48269`, final_radius_error `2.766e+04`, tail_mean_abs_vr `4.842e+01`, max_abs_residual `0.000e+00`