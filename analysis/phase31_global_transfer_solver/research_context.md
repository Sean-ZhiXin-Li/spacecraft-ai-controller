# Phase 31 Research Context

## Scope Reviewed

- `README.md` for the Phase 7.6 milestone and current 2D framing.
- Phase 25 recoverability basin mapping.
- Phase 28 trajectory-family mapping.
- Phase 29 repo audit and Burn-A family selector outputs.
- Phase 30 Burn-A endpoint search outputs.
- Phase 22 and Phase 30 controller scripts.

## Why Burn A/B May Be Structurally Insufficient

- Phase 23 and Phase 24 changed Burn B search and insertion geometry without producing recoverable crossings.
- Phase 26 showed tangential correction can occur locally but does not move the crossing-state distribution.
- Phase 27 showed Burn-B timing does not reduce crossing sync error on the cases that actually cross.
- Those results imply Burn B is acting after the relevant orbital family has already been selected.
- Phase 30 searched Burn-A endpoints and still did not move crossing-state structure.
- The Burn A -> coast -> Burn B architecture may not be expressive enough to synchronize radius, vr, and vt.

## Why Endpoint Manifold Search Failed

- Phase 25 identified tangential velocity as the dominant recoverability blocker.
- Phase 30 mapped 6912 bounded Burn-A endpoint candidates.
- Selected endpoints altered energy and angular momentum but left crossings, near-recoverability, CAPTURE, and success unchanged.
- Endpoint selection optimizes a state, not a complete transfer path.

## Why Local Geometry May Not Solve Global Transfer

- A useful endpoint is not enough if the transfer arc and insertion timing are not globally designed.
- Transfer path geometry determines when radius crossing, radial velocity, and tangential velocity align.
- Prior phases repeatedly changed local behavior without changing crossing-state families.

## Why Transfer-Class Reasoning Is Required

- Phase 31 searches named transfer classes instead of action heuristics.
- Direct, Hohmann-like, Lambert-like, and energy-ladder families encode different physical assumptions.
- Each candidate explicitly contains burn timing, burn magnitudes, burn directions, coast arcs, and insertion-state preview.
- This is architecture-level transfer design, not random brute force.

## Control Architecture Comparison

- Local control chooses immediate action and failed to expand reachability.
- Staged burn control creates insertion windows but not recoverable crossings.
- Endpoint search chooses a post-Burn-A state but not a complete orbital path.
- Global transfer search chooses a path class connecting initial state to a target-family crossing geometry.

## Physical Definition Of Global Transfer

- Orbital energy ladder: the sequence of energy changes across burns and coast arcs.
- Angular momentum transition: deliberate changes in tangential support before insertion.
- Periapsis shaping: lowering or raising closest approach to create target-radius intersection.
- Apoapsis shaping: controlling the far point of the transfer ellipse.
- Transfer arc geometry: the coast segment that brings the spacecraft to the target radius.
- Insertion-state targeting: previewing radius error, vr, vt, energy, and angular momentum near the future crossing.

## Phase 31 Hypothesis

- If architecture class is the missing layer, at least one global transfer family should reduce dead windows or improve crossing-state quality.
- If this still fails, Phase 32 should move from explicit transfer families to true optimal control, direct collocation, or MPC.