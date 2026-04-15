# Time Scaling Reachability

## Test Setup

- Controller: `probe_max_retrograde`
- `r0_over_target`: `1.00005, 1.00020`
- `max_steps` tested: `100000, 200000`
- `dt` tested: `50, 100`
- Strict tolerances:
  - `tol_r = 0.001`
  - `tol_v = 0.001`
  - `tol_ang = 0.02`

## Main Answers

- Crossing first became possible at `dt = 50` for `r0_over_target = 1.00005`.
- First crossing step: `96174`
- Total simulated time to first crossing: `4.809e+06` seconds

## Per-Configuration Results

- `dt=50`, `r0=1.00005`: crossings `1`, first_crossing_step `96174`, simulated_time `4.809e+06`, final_radius_error `8.143e+06`, tail_mean_abs_vr `44.809`
- `dt=50`, `r0=1.00020`: crossings `0`, first_crossing_step `None`, simulated_time `5.000e+06`, final_radius_error `1.118e+09`, tail_mean_abs_vr `44.321`
- `dt=100`, `r0=1.00005`: crossings `0`, first_crossing_step `None`, simulated_time `2.000e+07`, final_radius_error `5.417e+08`, tail_mean_abs_vr `242.134`
- `dt=100`, `r0=1.00020`: crossings `0`, first_crossing_step `None`, simulated_time `2.000e+07`, final_radius_error `1.666e+09`, tail_mean_abs_vr `242.114`

## Recommended Baseline

- Recommended new `dt`: `50`
- Supporting configuration: `r0_over_target = 1.00005`, `max_steps = 100000`
- Metrics: crossings `1`, first_crossing_step `96174`, simulated_time `4.809e+06`, final_radius_error `8.143e+06`