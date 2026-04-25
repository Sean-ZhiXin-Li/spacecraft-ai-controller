# Phase 6.5 Window-Seeking Summary

## Setup

- Compared controllers: baseline explicit controller and Window-Seeking v1 (`WS-1`).
- WS-1 keeps CAPTURE and LOCK equations unchanged and only changes DESCENT behavior near the target-radius band.
- Near-target band: `|radius error| / target_radius < 8e-5`.
- Candidate actions in window-seeking mode: pure retrograde; retrograde plus small inward radial bias; reduced retrograde magnitude.
- Candidate selection uses one-step explicit Euler lookahead with a simple score over radius error, radial velocity, tangential-speed error, and energy error, plus a small bonus for more capture-compatible states.

## Results

- Baseline: success `64`, capture `65`, lock `65`, near-miss `73`.
- WS-1: success `157`, capture `157`, lock `157`, near-miss `42`.

## Answers

1. Yes. WS-1 improves CAPTURE reachability and strict success substantially on this grid: success rises from `64` to `157`, and CAPTURE rises from `65` to `157`.
2. The gain is overwhelmingly from better window access, not better post-CAPTURE stabilization. CAPTURE and LOCK remain tightly coupled for both controllers, so the main change is that WS-1 reaches CAPTURE much more often.
3. In this local 2D study, one-step explicit lookahead helps. The candidate-selection logic converts many near-miss and energy-limited pre-CAPTURE failures into successful CAPTURE entries while leaving the downstream CAPTURE/LOCK laws unchanged.
4. The main improvements are in near-miss and energy-limited cases. The representative comparisons show strong recovery on both of those failure types, while the geometry-miss example remains unchanged, which is consistent with WS-1 only acting inside a near-target band.
5. Best next step: stay in 2D and refine the WS-1 near-target activation band and score weights, then test a small WS-2 family that keeps the same three-candidate structure but varies the candidate magnitudes and scoring weights locally rather than returning to broad Phase 6 parameter sweeps.

## Caution

WS-1 is still a local 2D explicit controller. Any improvement should be read as evidence about pre-CAPTURE reachability in this narrow regime, not as a broad generalization result.
