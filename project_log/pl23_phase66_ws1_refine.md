# PL23 - Phase 6.6 WS-1 Refinement

## Goal

Refine WS-1 with a compact one-knob-at-a-time parameter sweep while preserving the same evaluator, physics, WS candidate structure, and CAPTURE/LOCK equations.

## Method

Tested variants around near-band width, reduced-retrograde scale, and energy score weight on the same 270-regime grid.

## Key Result

The best variant was `retro_scale_0p80`:

- Original WS-1: 157 successes, 157 CAPTURE entries, 42 near-misses.
- `retro_scale_0p80`: 172 successes, 172 CAPTURE entries, 47 near-misses.

## Interpretation

Increasing the reduced-retrograde scale improved success and CAPTURE count, but it also increased near-misses. The success set shifted as well as widened.

## Limitations

The sweep was local and one-factor-at-a-time. It improved reachability but did not solve the reachability/stability trade-off.

## Next Step

Test adaptive WS schedules that vary reduced-retrograde behavior smoothly inside the WS band.
