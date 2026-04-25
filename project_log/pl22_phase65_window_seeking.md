# PL22 - Phase 6.5 Window-Seeking

## Goal

Introduce WS-1, a local one-step window-seeking explicit controller, to improve target-radius crossing and CAPTURE entry without changing physics or CAPTURE/LOCK equations.

## Method

WS-1 activates only inside a near-target radius band and evaluates a small fixed candidate set with one-step lookahead: pure retrograde, inward-biased retrograde, and reduced retrograde. Candidate scoring uses radius error, radial velocity, tangential velocity error, and energy error.

## Key Result

WS-1 strongly improved reachability compared with the baseline explicit controller on the 270-regime grid:

- Baseline: 64 successes, 65 CAPTURE entries, 73 near-misses.
- WS-1: 157 successes, 157 CAPTURE entries, 42 near-misses.

## Interpretation

The main gain came from better local window access before CAPTURE, not from post-CAPTURE stabilization changes. CAPTURE and LOCK logic remained unchanged.

## Limitations

WS-1 is still local to the near-target band and does not shape the global approach geometry before window entry.

## Next Step

Refine WS-1 parameters in the same 2D Python evaluator, especially reduced-retrograde magnitude and score sensitivity.
