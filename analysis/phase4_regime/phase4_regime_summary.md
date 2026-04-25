# Phase 4 Nearby-Regime Sweep

## Setup

- Controller: unchanged explicit `DESCENT -> CAPTURE -> LOCK` controller.
- Environment physics: unchanged 2D `OrbitEnv`; only task parameters and initial condition are varied.
- Fixed `dt`: `100`.
- Fixed `max_steps`: `100000`.
- Default target radius reference: `7.500e+12` m.

## Aggregate Results

- Total regimes: `108`.
- Strict successes: `15/108` (13.9%).
- CAPTURE entered: `15/108` (13.9%).
- LOCK entered: `15/108` (13.9%).
- CAPTURE without LOCK: `0` regimes.
- LOCK without strict success: `0` regimes.
- In this grid, CAPTURE entry, LOCK entry, and strict success are the same set of regimes.

## Direct Answers

1. The capture mechanism is broader than one exact baseline, but it is still local. It succeeds in `15/108` nearby regimes, not only at `r0=1.00005`, `angle=170`, `thrust=10000`, `target scale=1.00`.
2. `thrust_scale` matters most in this sweep: `8000` and `12000` produce `0/36` successes each, while `10000` produces all `15/36` successes. Angle and target-radius scale are also strong: `165 deg` outperforms `170 deg`, `175 deg` is nearly absent, and `target scale=1.02` has only `1/36` success. The tested `r0` values matter, but less abruptly than thrust.
3. The results show success pockets, not a broad connected band. At `thrust=10000`, the successful pocket shifts with angle and target-radius scale; outside that thrust slice, the controller does not enter CAPTURE at all.
4. There are no CAPTURE-without-LOCK regimes in this grid. When the controller reaches CAPTURE, it also reaches LOCK and satisfies strict success. Failures are mostly pre-capture failures.
5. The explicit controller should be treated as a local phase mechanism with some nearby transfer, not as a general 2D multi-regime controller.

## Dimension Sensitivity

- Success by `r0_over_target`: `1.00002`: `6/27`; `1.00005`: `4/27`; `1.00008`: `3/27`; `1.00012`: `2/27`.
- Success by `initial_velocity_angle_deg`: `165`: `9/36`; `170`: `5/36`; `175`: `1/36`.
- Success by `thrust_scale`: `8000`: `0/36`; `10000`: `15/36`; `12000`: `0/36`.
- Success by `target_radius_scale`: `0.98`: `8/36`; `1.00`: `6/36`; `1.02`: `1/36`.
- CAPTURE entry by `r0_over_target`: `1.00002`: `6/27`; `1.00005`: `4/27`; `1.00008`: `3/27`; `1.00012`: `2/27`.
- CAPTURE entry by `initial_velocity_angle_deg`: `165`: `9/36`; `170`: `5/36`; `175`: `1/36`.
- CAPTURE entry by `thrust_scale`: `8000`: `0/36`; `10000`: `15/36`; `12000`: `0/36`.
- CAPTURE entry by `target_radius_scale`: `0.98`: `8/36`; `1.00`: `6/36`; `1.02`: `1/36`.

## Near Misses

- r0 `1.00002`, angle `175`, thrust `10000`, target scale `1.00`: min abs radius error `3.890e+07` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `170`, thrust `10000`, target scale `1.02`: min abs radius error `5.676e+07` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `175`, thrust `10000`, target scale `1.02`: min abs radius error `1.288e+08` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `175`, thrust `12000`, target scale `0.98`: min abs radius error `1.470e+08` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `175`, thrust `8000`, target scale `0.98`: min abs radius error `1.470e+08` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `170`, thrust `12000`, target scale `0.98`: min abs radius error `1.470e+08` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `170`, thrust `8000`, target scale `0.98`: min abs radius error `1.471e+08` m, CAPTURE `False`, LOCK `False`.
- r0 `1.00002`, angle `165`, thrust `12000`, target scale `0.98`: min abs radius error `1.471e+08` m, CAPTURE `False`, LOCK `False`.

## Interpretation

- The capture mechanism is not treated here as redesigned or retuned; every result is for the fixed Phase 3 controller.
- Strict success and CAPTURE entry are concentrated around a small subset of the grid, so the controller should be read as a local construction rather than a broad nearby-regime solution.
- Success pockets indicate that the phase mechanism can transfer across some neighboring 2D regimes, but pockets are weaker evidence than a connected success band.
- This sweep did not find regimes that enter CAPTURE but fail LOCK. For these nearby cases, the dominant failure mode is failure to reach CAPTURE at all.
- These results are still local to the sampled 2D grid and should not be generalized to 3D or large orbital changes.
