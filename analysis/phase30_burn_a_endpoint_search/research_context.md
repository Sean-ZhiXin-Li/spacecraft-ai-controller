# Phase 30 Research Context

## Scope Reviewed

- `README.md` for the Phase 7.6 milestone and current 2D framing.
- Phase 25 recoverability basin mapping.
- Phase 28 trajectory-family mapping.
- Phase 29 repo audit and Burn-A family selector outputs.
- Phase 22 and Phase 29 controller scripts.

## Why Burn B Is Likely Downstream

- Phase 23 and Phase 24 changed Burn B search and insertion geometry without producing recoverable crossings.
- Phase 26 showed tangential correction can occur locally but does not move the crossing-state distribution.
- Phase 27 showed Burn-B timing does not reduce crossing sync error on the cases that actually cross.
- Those results imply Burn B is acting after the relevant orbital family has already been selected.

## Why Timing And VT Are Insufficient

- Phase 25 identified tangential velocity as the dominant recoverability blocker.
- Phase 26 reduced local vt after Burn B but crossing vt remained unchanged.
- Phase 27 targeted predicted crossing timing but the same 12 crossing cases and sync error persisted.
- The active issue is not only vt magnitude; it is whether the early orbit family ever creates a useful crossing manifold.

## Why Phase 29 Failed

- Phase 29 changed Burn-A heuristics and directly instrumented Burn-A-end geometry.
- It did not improve crossings, recoverability, CAPTURE, success, or mean crossing sync error.
- Several heuristic selectors reduced energy or angular-momentum error at Burn-A end, but downstream crossing quality did not move.
- That suggests behavior heuristics are too weak; the next layer must optimize endpoint families explicitly.

## Why Endpoint Search Is The Next Structural Layer

- Burn A determines the passive coast arc and therefore the future crossing family.
- Endpoint search asks which post-Burn-A orbital state should be reached, rather than which instantaneous action is intuitive.
- The search remains bounded and interpretable: duration, thrust norm, radial bias, and tangential bias.
- Candidate endpoints are scored by physical diagnostics and passive preview, not by random controller spam.

## Physical Meaning Of Endpoint Family

- Energy controls whether the resulting conic has enough orbital size to cross the target radius.
- Angular momentum controls tangential support and therefore vt mismatch at crossing.
- Periapsis and apoapsis proxies describe whether the passive orbit geometrically intersects the target radius.
- Eccentricity proxy describes how strongly the endpoint has been shaped into an ellipse rather than a near-circular dead window.
- Predicted crossing step and preview sync estimate whether the endpoint can produce a useful crossing before Burn B begins.

## Phase 30 Hypothesis

- If Burn-A endpoint optimization can find better orbital families, crossing count, dead-window count, or near-recoverable crossings should improve before any Burn-B redesign.
- If it still fails, the evidence shifts toward a global orbital transfer solver or Lambert-like transfer layer for Phase 31.