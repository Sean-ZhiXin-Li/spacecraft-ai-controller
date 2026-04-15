# Orbit Lock Generalization

## Sweep

- `r0_over_target`: `1.00002, 1.00005, 1.00010, 1.00020`
- `dt`: `20, 50, 100`
- `thrust_scale`: `10000, 20000, 40000`
- `max_steps = 100000`
- Controller: `explicit_orbit_lock`

## Main Answers

- Crossing generalizes to `5` / `36` tested setups.
- Tail crossings persist in `5` setups.
- Strict success is reached in `5` setups.

## Best Observed Regime

- `r0_over_target = 1.00005`
- `dt = 100`
- `thrust_scale = 10000`
- crossings `1`
- tail_crosses_target_radius `True`
- success `True`
- final_radius_error `2.766e+04`
- tail_mean_abs_vr `48.418`

## Interpretation

- Crossing sensitivity is now a joint function of `r0`, `dt`, and thrust rather than a single-setup artifact.
- The next benchmark should use one successful setup, one crossing-without-success setup if present, and one failure setup from this table.