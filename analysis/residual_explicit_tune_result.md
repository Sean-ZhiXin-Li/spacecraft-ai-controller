# Residual Explicit Tuning Result

## Main Answer

- Does the tuned residual preserve explicit-controller success? `Yes`
- Does it improve final radius error? `No`
- Does it improve tail radial velocity? `No`
- Does it destabilize the controller? `No`
- Did every nonzero tuned residual harm success? `Yes`
- What does this imply about safe hybrid control? Every nonzero residual attempt harmed success, so the safe checkpoint remained at zero residual.

## Metrics

- alpha `0.05`
- best_objective `2.766247e+04`
- tuned success `True` vs explicit `True`
- tuned crossings `1` vs explicit `1`
- tuned first_crossing_step `48269` vs explicit `48269`
- tuned final_radius_error `2.766e+04` vs explicit `2.766e+04`
- tuned tail_mean_abs_vr `4.842e+01` vs explicit `4.842e+01`
- tuned mean_abs_residual `0.000e+00`
- tuned max_abs_residual `0.000e+00`

## Optimization History

- step `0` accepted `True` reason `zero_initialization` objective `2.766247e+04` success `True` final_radius_error `2.766e+04` tail_mean_abs_vr `4.842e+01`
- step `1` accepted `False` reason `rejected_success_not_preserved` objective `3.760961e+08` success `False` final_radius_error `3.761e+08` tail_mean_abs_vr `6.909e+01`
- step `2` accepted `False` reason `rejected_success_not_preserved` objective `3.779516e+08` success `False` final_radius_error `3.780e+08` tail_mean_abs_vr `7.261e+01`
- step `3` accepted `False` reason `rejected_success_not_preserved` objective `3.817727e+08` success `False` final_radius_error `3.818e+08` tail_mean_abs_vr `5.659e+02`
- step `4` accepted `False` reason `rejected_success_not_preserved` objective `3.705387e+08` success `False` final_radius_error `3.705e+08` tail_mean_abs_vr `5.584e+02`
