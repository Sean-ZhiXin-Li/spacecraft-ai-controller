# Residual Explicit Magnitude-Only Result

## Main Answer

- Does magnitude-only residual preserve explicit-controller success? `Yes`
- Does it improve final radius error? `No`
- Does it improve tail radial velocity? `No`
- Does it destabilize the controller? `No`
- Did every nonzero magnitude-only residual harm success? `No`
- What does this imply about constrained hybrid control? Magnitude-only residual control preserved the explicit action direction, but this run did not justify moving away from the zero-residual checkpoint.

## Metrics

- alpha `0.05`
- raw_residual_clip `0.1`
- max_magnitude_delta `0.005`
- best_objective `2.766247e+04`
- tuned success `True` vs explicit `True`
- tuned crossings `1` vs explicit `1`
- tuned first_crossing_step `48269` vs explicit `48269`
- tuned final_radius_error `2.766e+04` vs explicit `2.766e+04`
- tuned tail_mean_abs_vr `4.842e+01` vs explicit `4.842e+01`
- tuned mean_abs_residual `0.000e+00`
- tuned max_abs_residual `0.000e+00`

## Optimization History

- step `0` accepted `True` reason `zero_initialization` candidate_bias `0.000e+00` objective `2.766247e+04` success `True` final_radius_error `2.766e+04` tail_mean_abs_vr `4.842e+01`
- step `1` accepted `False` reason `rejected_no_objective_improvement` candidate_bias `2.000e-02` objective `7.819011e+04` success `True` final_radius_error `7.819e+04` tail_mean_abs_vr `3.681e+01`
- step `2` accepted `False` reason `rejected_success_not_preserved` candidate_bias `-2.000e-02` objective `1.760444e+05` success `False` final_radius_error `1.760e+05` tail_mean_abs_vr `3.004e-02`