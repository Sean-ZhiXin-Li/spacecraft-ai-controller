# Phase 5 Reachability Analysis

## Setup

- Source data: `analysis/phase4_regime/regime_grid.csv`.
- Selected representative cases: `20`.
- Detailed traces saved under `analysis/phase5_reachability/traces/`.
- Controller, environment physics, PPO, and learning experiments were not modified.

## Classification Counts

- Success cases in selected set: `6`.
- CAPTURE entered in selected set: `6`.
- `timing_miss`: `0`.
- `geometry_miss`: `6`.
- `energy_limited`: `8`.
- `overshoot_over_energy`: `0`.

## Answers

1. Dominant failure mode: `energy_limited` (`8` selected failures).
2. Most selected failures fail before CAPTURE because the DESCENT phase does not deliver the spacecraft into the narrow radius/energy/velocity window needed by the phase transition. The controller applies a fixed retrograde descent law, so parameter changes alter closest-approach timing and energy state without any adaptive targeting.
3. The capture window is defined operationally by reaching the target-radius neighborhood with compatible radial velocity and orbital energy. In this implementation, CAPTURE itself is triggered by a radius-error sign crossing, while strict success also requires sustained radius, speed, and angle tolerances.
4. The missing capability is regime-adaptive descent targeting: the controller lacks a way to modulate energy removal and crossing timing across thrust, initial velocity angle, radius offset, and target-radius scale. It can stabilize once the right window is reached, but it does not reliably steer into that window.

## Closest Failed Cases

- `case_07_r0_1p00002_a_175_th_10000_ts_1p00`: mode `energy_limited`, min abs radius error `3.890e+07` m, min energy error `4.868e+06` J/kg, CAPTURE `False`, LOCK `False`.
- `case_08_r0_1p00002_a_170_th_10000_ts_1p02`: mode `energy_limited`, min abs radius error `5.676e+07` m, min energy error `4.810e+06` J/kg, CAPTURE `False`, LOCK `False`.
- `case_09_r0_1p00002_a_175_th_10000_ts_1p02`: mode `energy_limited`, min abs radius error `1.288e+08` m, min energy error `4.810e+06` J/kg, CAPTURE `False`, LOCK `False`.
- `case_10_r0_1p00002_a_175_th_12000_ts_0p98`: mode `energy_limited`, min abs radius error `1.470e+08` m, min energy error `5.682e+06` J/kg, CAPTURE `False`, LOCK `False`.
- `case_11_r0_1p00002_a_175_th_8000_ts_0p98`: mode `energy_limited`, min abs radius error `1.470e+08` m, min energy error `4.095e+06` J/kg, CAPTURE `False`, LOCK `False`.

## Caution

This is a selected-case reachability analysis, not a new controller validation sweep. It explains representative Phase 4 failures and should be read together with the full Phase 4 grid.