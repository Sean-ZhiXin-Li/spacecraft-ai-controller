# Recovery Branch Mechanism Diagnosis v0

## Status

Published recovery evidence analyzed; no new trajectory executed.

Completed: 2026-07-25

## Source Evidence

- Result commit: `5f31c3fd74dbf8e8ea5a60d70d7b88f5a9def7c8`
- Implementation commit: `2e1fffbb00789c185256d0b13dff65150f21ba50`
- Branch-state canonical hash: `8b017254a8db2584a6732bcd086447ba405cf949d9e932cf03e71543b2cdb898`
- Manifest canonical hash: `e9cb96eae714bc0d8ed66d1a85f29baed2819d0d425a3ce9742b7e77ac236bad`
- Branch records: `4`
- Decision events: `30001`
- `manifest.json` SHA-256: `c317f94f937412f4fb5ac826fd97b21002c47cfc9bba6d9ad4eda6c4be1a921b`
- `branch_state.json` SHA-256: `b9fbcdd3544f527c7431d1b3bc5795ea755935a4973a18cbb8d8b710685d64fc`
- `results.csv` SHA-256: `c13abb4e15a6f04a9322c6c7955553464b9815d1b4d5c58374a3100eb4ccc668`
- `decision_log.jsonl` SHA-256: `43cfc05100648b6d0d652a8ac1d9a35f7179ebec78ff0138eaa5e7ab846096b4`
- `summary.md` SHA-256: `3e93152ef22e05a58650561111d4d3d96206391ea324a497993398eab3f8e8c0`
- `comparison.png` SHA-256: `d18310b4e15a9eb26bcc7884e8b56d9d6a90a4b231a999e7b8ed251dc4d902cb`

## Structural Validation

- Four frozen branches: confirmed in frozen order.
- Common branch-point Cartesian state: confirmed by the recovery-step 1 current-state hash.
- Event counts: 10,000 zero-action, 10,000 velocity-opposed, 10,000 tangential-correction, and one explicit-abort event.
- Source artifacts were read only. No rollout, interpolation, state reconstruction, or branch execution was performed.

## Branch-By-Branch Findings

### Zero Action

Zero action remained numerically ballistic at the endpoint: derived specific orbital energy changed by only `-0.00113432854414` J/kg and remained `22422353.775663` J/kg above the target circular energy. Radius moved toward the outer target by `6141180493.4` m, only `4.094430%` of the initial gap, while the final radius gap remained `143847482414.344727` m. Speed ratio declined slightly from `1.890602400` to `1.890151809` and remained within the exploratory 0.05 headroom band for all 10,000 events. The artifact does not contain per-step radius, so it cannot establish closest approach, a stable stalled region, or eventual recovery under a longer horizon.

### Velocity-Opposed Thrust

Velocity-opposed thrust reduced speed ratio from `1.824760376` to `0.046573769`. It reduced endpoint radial velocity from `6140.671312` to `151.984933` m/s and tangential velocity error from `848.475678` to `-4083.494966` m/s. Derived endpoint specific orbital energy ended `-9191994.827395` J/kg below the target circular energy. It therefore suppressed useful and hazardous motion together without restoring target geometry in this case.

### Tangential Correction

Tangential correction changed endpoint tangential error from `848.475678` to `-262.968555` m/s, but final radial velocity remained `6140.487528` m/s and `143848083404.778320` m of radius gap remained. Its speed ratio declined from `1.849451626` to `1.734683506`, while endpoint specific orbital energy remained `17434892.742626` J/kg above the target circular value. This supports tangential-component correction without task-geometry recovery, not a claim about all tangential policies.

### Explicit Abort

Explicit abort executed `0` recovery transitions and terminated as `explicit_abort`. It prevented further exposure through termination and did not provide task recovery.

## Cross-Branch Findings

- The three physical branches shared the exact recovery-step 1 current-state hash and first diverged in next-state hash at recovery step `1`.
- No exact recorded state-hash convergence occurred between physical branches after divergence. Different hashes establish nonidentity, not physical distance.
- Velocity-opposed and tangential correction each used magnitude 0.25 for 10,000 transitions. Their equal effort of 2,500 and equal delta-v proxy follow from the same norm and duration; their action directions and state hashes were distinct.
- The two active branches never proposed exactly equal or exactly opposite recorded actions at the same step. Their state-dependent directions changed at every consecutive boundary, so the experiment tested persistent single-mode rules, not fixed inertial action vectors.
- Velocity-opposed final speed ratio was `0.046573769`; tangential-correction final speed ratio was `1.734683506`. Equal scalar cost did not produce equivalent trajectories.
- Velocity-opposed improved the endpoint radial-velocity ratio to `0.036125254` but degraded tangential error ratio to `-0.970604729`; tangential correction improved tangential error ratio to `-0.062504919` while radial-velocity ratio remained `1.459530692`. All endpoint radius and radial-velocity margins remained outside the Phase34-compatible limits.
- Final Veto allowed all 30,000 physical proposals. Post-branch stalling was not caused by repeated veto intervention.
- Endpoint radius summaries show limited net progress, but per-step target geometry is unavailable. No closest-approach or longer-horizon recovery claim is supported.

## Mechanism Diagnosis

### Directly Supported

- `A_hazard_only_correction`: All 30000 physical proposals were allowed and all realized speed ratios remained below 1.90.
- `C_excessive_braking_or_energy_removal`: Velocity-opposed speed ratio fell from 1.824760 to 0.046574.
- `D_tangential_only_correction_limitation`: Tangential endpoint error changed from 848.475678 to -262.968555 m/s while no crossing occurred.
- `I_insufficient_observability_in_published_artifacts`: Decision events contain state hashes, actions, and speed ratios but no per-step state vectors.

### Consistent Or Partial

- `B_insufficient_radial_commitment` (`partially_supported`): Velocity-opposed final radial velocity fell from 6140.671312 to 151.984933 m/s and reduced only 0.035715% of the initial radius gap.
- `G_state_region_irrecoverability_under_tested_actions` (`consistent_with_evidence`): None of the three declared physical responses recovered from the identical branch state within 10000 transitions.
- `H_missing_phase_switching` (`consistent_with_evidence`): Each physical branch used one response rule for the entire horizon and none reached crossing.

### Unevaluable Or Unsupported

- `E_static_action_limitation` (`not_supported`): Velocity-opposed and tangential actions changed recorded direction on 9999 and 9999 consecutive boundaries; they were recomputed from current state rather than repeating one fixed vector.
- `F_horizon_limitation` (`not_evaluable`): Recovery-horizon exhaustion alone does not establish that a longer horizon would recover.

## Strongest Supported Conclusion

For this frozen one-case diagnostic, the three tested physical responses prevented realized overspeed but did not restore target crossing or Phase34-compatible recoverability within 10,000 transitions. Zero action retained the branch-point energy and made limited endpoint radius progress; velocity-opposed thrust over-suppressed radial and tangential motion; tangential correction improved the endpoint tangential component without resolving the radius and radial-velocity components. These are endpoint and logged-speed findings, not a proof that the state is irrecoverable under other policies.

## Next Architecture Requirement

The next recovery policy should separate hazard arrest from task recovery and use state-dependent staged decisions. At minimum it should monitor radius progress, radial velocity, tangential error, orbital-energy change, and the Phase34 recoverability component vector; stop a single-mode response when progress stalls; and switch deliberately among hazard arrest, radial recommitment, tangential alignment, crossing, retreat, and termination. This is a design requirement inferred from the diagnosed failure modes, not a validated policy.

## Evidence Limitations

Per-step Cartesian state, radius, radial velocity, tangential velocity, target-radius error, orbital energy, and recoverability-component margins are `not_available_in_published_artifacts`. State hashes cannot be inverted into those quantities. Only branch-point and final endpoint orbital summaries support physical derivation.

## Claim Restrictions

This analysis does not establish branch optimality, universal failure of velocity-opposed thrust, universal failure of tangential correction, state irrecoverability under all controllers, formal safety, hardware validity, benchmark-wide recovery performance, recovery under a longer horizon, or success of any proposed future policy.
