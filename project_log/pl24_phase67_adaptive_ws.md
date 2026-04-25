# PL24 - Phase 6.7 Adaptive Window-Seeking

## Goal

Test adaptive WS schedules to preserve the Phase 6.6 reachability gain while reducing near-misses.

## Method

Reused the same 270-regime evaluator, WS candidate structure, CAPTURE/LOCK logic, and physics. Compared fixed WS references against adaptive schedules, including `adaptive_soft`.

## Key Result

`adaptive_soft` was the best adaptive schedule:

- `fixed_best_0p80`: 172 successes, 172 CAPTURE entries, 47 near-misses.
- `adaptive_soft`: 172 successes, 172 CAPTURE entries, 44 near-misses.

## Interpretation

Adaptive WS stabilized window entry modestly but did not expand the success count. Remaining failures likely came from approach geometry before WS activation.

## Limitations

The adaptive behavior only acted inside the WS band, so it could not correct global pre-window approach geometry.

## Next Step

Add a pre-window shaping stage before WS activation while keeping CAPTURE/LOCK and physics unchanged.
