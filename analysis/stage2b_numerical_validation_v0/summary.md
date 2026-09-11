# Stage 2B Numerical Propagation Independence Validation v0

## Scope

This Stage 2B validation compares the frozen Phase34/35 semi-implicit Euler
one-step propagator against an independent SciPy `solve_ivp` DOP853
propagator.

The comparison preserves:

- identical exact Cartesian initial state;
- identical frozen physical parameters;
- identical frozen action;
- identical 100 s controller decision interval;
- identical strict overspeed criterion:
  `predicted_speed_ratio > 1.90`.

The action is held fixed throughout each 100 s control interval. DOP853 may
use adaptive internal integration steps, but those internal steps do not
change controller decision frequency.

This validation is offline numerical analysis only. It does not authorize or
modify Stage 2A runtime control behavior.

## Evidence Coverage

The frozen Stage 2A exact-state evidence contains:

- 4 exact Phase35 veto states;
- 4 nominal proposals;
- 10 evaluated physical alternative proposals;
- 14 total one-step proposal evaluations.

The evaluated physical alternatives include:

- `zero_action_reference_v0`;
- `velocity_opposed_thrust_v0`;
- `tangential_error_correction_v0`.

`explicit_abort_v0` is terminal semantics with no physical action proposal and
is therefore not propagated by either numerical method.

## Results

All 14 evaluated proposals produced identical strict overspeed
classification under the two propagators.

Nominal proposals:

- Euler / DOP853 classification agreement: 4 / 4;
- all four nominal proposals remained above the strict 1.90 threshold;
- all four remained Final-Veto-classifiable as overspeed proposals.

Physical alternatives:

- Euler / DOP853 classification agreement: 10 / 10;
- all ten evaluated physical alternatives remained at or below 1.90;
- all ten therefore retained their allowed safety classification.

Overall:

- total evaluated proposals: 14;
- classification matches: 14;
- classification mismatches: 0;
- all classifications match: true.

The maximum observed absolute speed-ratio difference between DOP853 and the
recomputed Euler result was:

`2.886579864025407e-15`

The closest evaluated allowed proposal to the 1.90 threshold was the
angle-150 / thrust-8000 zero-action alternative, with DOP853 speed ratio:

`1.8906024003603126`

giving threshold headroom:

`0.009397599639687337`

This headroom is approximately `3.26e12` times larger than the maximum
observed Euler-DOP853 speed-ratio discrepancy.

## Interpretation

Within the audited one-step exact-state coverage, the observed Final Veto
classification structure is numerically consistent across the frozen
semi-implicit Euler implementation and the independent DOP853 propagator.

In particular, the audited structure remains:

- nominal overspeed proposal -> VETO;
- evaluated physical safe alternative -> ALLOW.

For the canonical Phase35 boundary, the more specific structure also remains:

- nominal proposal -> VETO;
- zero action -> ALLOW;
- velocity-opposed thrust -> ALLOW;
- tangential correction -> ALLOW.

## Claim Boundary

This result does **not** prove that the complete spacecraft project is
independent of numerical integration method.

The validated claim is limited to:

- the 4 audited exact Phase35 states;
- the 14 frozen one-step proposals;
- the frozen Phase34/35 dynamics;
- the 100 s control interval;
- the strict `speed_ratio > 1.90` safety classification.

This validation does not yet establish numerical-method independence for:

- multi-step trajectories;
- crossing time or crossing location;
- recovery-horizon outcomes;
- recoverability milestones;
- long-horizon orbital insertion behavior;
- controller decisions after state divergence accumulates;
- model fidelity relative to real spacecraft dynamics;
- thrust uncertainty;
- state-estimation error;
- unmodeled forces or perturbations.

DOP853 provides an independent higher-order numerical integration check. It
does not validate the physical model itself.

## Artifacts

Primary result:

`analysis/stage2b_numerical_validation_v0/results.json`

Runner:

`scripts/run_stage2b_numerical_validation_v0.py`

Checker:

`scripts/check_stage2b_numerical_validation_v0.py`

Independent propagator:

`runtime_assurance/dop853_validation.py`

Focused unit tests:

`Tests/test_stage2b_numerical_validation.py`