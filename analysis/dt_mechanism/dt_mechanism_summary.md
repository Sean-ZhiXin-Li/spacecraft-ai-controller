# dt Mechanism Analysis

## Setup

- Fixed `r0_over_target`: `1.00005`.
- Fixed `max_steps`: `100000`.
- Fixed `thrust_scale`: `10000`.
- Swept `dt`: `50, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300`.

## Results

- Successful dt values: `100, 130, 140, 150`.
- Non-monotonic success in dt: No. Success is re-entrant over this sweep, with failures and successes interleaved.
- Crossed successes: `100, 150`.
- Near-band successes without a sign crossing: `130, 140`.
- Failures with a radius crossing: `0`.
- Failures without a radius crossing: `7`.
- Local success-to-failure transitions: `dt=100` succeeds but `dt=110` fails; `dt=150` succeeds but `dt=200` fails.

## Closest Failed Cases

- `dt=300`: min abs radius error `5.349e+04` m, signed error at closest approach `5.349e+04` m, time of closest approach `13.31` days, no crossing.
- `dt=90`: min abs radius error `1.351e+08` m, signed error at closest approach `1.351e+08` m, time of closest approach `104.17` days, no crossing.
- `dt=50`: min abs radius error `1.953e+08` m, signed error at closest approach `1.953e+08` m, time of closest approach `57.87` days, no crossing.

## Mechanism Interpretation

- In this sweep, failed cases do not cross the target radius; crossing access is the dominant separator.
- `dt=130, 140` succeeds without a sign-change crossing. These runs come close enough to satisfy the strict near-radius condition while staying on the same side of the target radius in the sampled trace.
- The no-crossing failures (`dt=50, 80, 90, 110, 120, 200, 300`) approach the target band with different closest-approach timing, then drift away or remain outside the capture condition instead of activating the later phases.
- The re-entrant pockets are most plausibly a discrete-time event-alignment effect: dt changes when the descent trajectory samples the narrow capture window and how much energy is removed before that sample.
- The measured failures are mostly missed-window cases rather than delayed successes: none of the failed runs crosses later within the 100000-step budget. `dt=300` is the closest failed run by radius error, but it still does not sustain the phase sequence and later diverges.
- The energy traces should therefore be read with the radius-error traces: successful dt values reach the crossing or near-radius window after a compatible energy evolution, while nearby failures miss that window or sample it with poor timing.
- This is a local explanation for the fixed 2D setup only; it should not be generalized beyond the current controller, initial condition, and environment settings.