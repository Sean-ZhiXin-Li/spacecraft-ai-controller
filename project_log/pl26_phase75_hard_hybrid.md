# PL26 - Phase 7.5 Hard Hybrid

## Goal

Combine the reachability of `prewindow_radial_medium` with the stability of `adaptive_soft` using a simple threshold switch.

## Method

Tested hard-switch variants where DESCENT used `prewindow_radial_medium` outside a radius-error threshold and `adaptive_soft` inside it. No action blending, new phases, learning, or physics changes were introduced.

## Key Result

Best hard hybrid:

- `hard_hybrid_1e4`: 170 successes, 170 CAPTURE entries, 42 near-misses.

References:

- `adaptive_soft`: 172 successes, 44 near-misses.
- `prewindow_radial_medium`: 209 successes, 56 near-misses.

## Interpretation

Hard switching reduced near-misses but damaged reachability. The boundary separated two behaviors that needed coordination rather than abrupt handoff.

## Limitations

Threshold choice was brittle. The hard switch did not preserve the pre-window success set.

## Next Step

Replace hard switching with continuous action blending between pre-window shaping and adaptive WS.
