# Orbit Lock Benchmark

## Representative Setups

- `best_success`: `r0_over_target=1.00005`, `dt=100`, `thrust_scale=10000`
- `fastest_cross`: `r0_over_target=1.00002`, `dt=100`, `thrust_scale=10000`
- `representative_failure`: `r0_over_target=1.00002`, `dt=20`, `thrust_scale=10000`

## Results

- `best_success` / `explicit_orbit_lock`: crossings `1`, tail_crosses `True`, success `True`, final_radius_error `2.766e+04`, tail_mean_abs_vr `48.418`
- `best_success` / `ppo_speed_refine_50`: crossings `0`, tail_crosses `False`, success `False`, final_radius_error `7.432e+08`, tail_mean_abs_vr `688.923`
- `best_success` / `probe_max_retrograde`: crossings `1`, tail_crosses `False`, success `False`, final_radius_error `3.116e+07`, tail_mean_abs_vr `30.103`
- `fastest_cross` / `explicit_orbit_lock`: crossings `1`, tail_crosses `True`, success `True`, final_radius_error `7.914e+04`, tail_mean_abs_vr `97.264`
- `fastest_cross` / `ppo_speed_refine_50`: crossings `0`, tail_crosses `False`, success `False`, final_radius_error `5.183e+08`, tail_mean_abs_vr `688.923`
- `fastest_cross` / `probe_max_retrograde`: crossings `1`, tail_crosses `False`, success `False`, final_radius_error `2.571e+08`, tail_mean_abs_vr `29.913`
- `representative_failure` / `explicit_orbit_lock`: crossings `0`, tail_crosses `False`, success `False`, final_radius_error `1.229e+08`, tail_mean_abs_vr `19.405`
- `representative_failure` / `ppo_speed_refine_50`: crossings `0`, tail_crosses `False`, success `False`, final_radius_error `2.256e+08`, tail_mean_abs_vr `138.492`
- `representative_failure` / `probe_max_retrograde`: crossings `0`, tail_crosses `False`, success `False`, final_radius_error `1.229e+08`, tail_mean_abs_vr `19.405`

## Main Answers

- Explicit controller successes: `2` / `3`
- Probe controller successes: `0` / `3`
- PPO successes: `0` / `3`
- The explicit controller should be compared as a structured insertion controller, not as a generic baseline.