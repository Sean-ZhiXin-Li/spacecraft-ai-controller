# Stage 2A Active Hazard Arrest Authority Boundary Preflight v0

## Status

Future hazard-arrest authority boundary frozen; no active staged authority granted.

Completed: 2026-08-12

## Evidence Basis

The frozen Stage 1B calibration result validates 216 shadow candidates and
2,808 offline replays. `engineering_candidate_v0` comes from
`shadow_candidate_hc2_d4_w2_r3_n1_cd0_tb8` and remains shadow-only, without
active authority or scientific threshold validation. Its selected replay has
zero hazard-arrest entries, zero invalid guard observations, 2,817 unavailable
guard observations, and zero nominal-handoff recommendations.

## Ready Now

- measured Cartesian state and pure orbital derivations;
- distinct realized and one-step predicted speed ratios;
- strict overspeed `speed_ratio > 1.90` and clear complement `<= 1.90`;
- signed headroom, validity, action, prediction, provenance, and counter evidence;
- unchanged one-step Final Veto for every non-abort proposal;
- validated existing `velocity_opposed_thrust_v0` generation and execution path;
- terminal explicit-abort semantics with no action and zero transition.

## Not Ready

No active entry policy, completion guard, release hysteresis, active dwell,
intervention budget, correction-authority evaluator, handoff-readiness evaluator,
retreat action, or autonomous abort policy is frozen. Shadow parameters are not
active thresholds. Missing and unsupported evidence cannot become clear or
favorable evidence.

## Future Boundary

The smallest future authority is one predeclared provisional velocity-opposed
proposal for `hazard_arrest` only, evaluated through unchanged Final Veto. Its
future pre-action trigger is valid predicted overspeed for the normal branch
while the measured current state remains realized-clear. A trace permits at most
one intervention proposal. Release requires fresh valid realized-clear and
resumed-branch predicted-clear evidence with no adverse stop, and returns only
to the exact predeclared existing recovery branch. Realized overspeed remains an
adverse terminal condition. All other staged phases remain shadow-only, and
explicit abort remains terminal rather than physical action.

## Non-Claims

This preflight demonstrates no recovery improvement, action safety, threshold
validity, stability, controller optimality, handoff readiness, retreat capability,
formal safety, hardware validity, or deployment readiness. No simulator or
recovery branch was executed to produce it.
