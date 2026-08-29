# Stage 2A Q1/E Eligibility Audit V0

## Verdict

`Q1_NOT_AUTHORIZED_ALREADY_CONSUMED_NO_ELIGIBLE_CANDIDATE`

The requested premise that Q1's qualification directory is absent and its invocation count is zero is contradicted by frozen repository evidence. Q1 was executed and published on 2026-08-21 at commit `2a82d396cf6291123d3a62c02b6dcfad3c586fff`, before D1 and D2. Its authoritative result inspected 384 valid Stage 1B states using 768 offline normal predictions, executed zero physical transitions, found zero eligible boundaries, and froze `selection_status: no_eligible_boundary`.

The qualifier uses new-only atomic publication and rejects an existing output directory. Therefore its single authoritative slot is consumed and cannot be rerun without violating the frozen contract.

## Q1

Q1 is the offline qualification over the 13 frozen Stage 1B traces. At each valid pre-transition state with remaining infrastructure capacity it requires:

- current realized speed ratio `<= 1.90`;
- an existing normal branch (`zero_action_reference_v0` or `tangential_error_correction_v0`) predicting ratio `> 1.90` and rejected by unchanged Final Veto; and
- the existing `velocity_opposed_thrust_v0` proposal allowed by unchanged Final Veto.

Candidates are ordered lexically by registry member, source trace, prefix transition count, and normal branch. Intervention quality is not optimized.

## Stage 2A-E

E is the separately guarded measured one-intervention experiment. It requires a committed clean Q1 selection, exact frozen-prefix reproduction, one baseline run, one fresh active run, unchanged Final Veto, at most one hazard-arrest proposal and one intervention transition, no automatic retry, and exact prediction/realization identity checks. Its output directory is absent, but E is not eligible because Q1 selected no candidate.

## D2 separation

D2 is discovery-only: `active_authority_granted=false` and `hazard_arrest_interventions=0`. Its implementation rejects imports of active Stage 2A authority. It does not invoke Q1 or E, and its `NO_BOUNDARY` result does not alter Q1's trace set, threshold, comparator, branch identities, or selection. D2's closest ratio `1.8906024003603095` is not a Q1 candidate and cannot reopen the consumed qualification.

## Integrity

All 13 trace identities listed in `analysis/staged_recovery_shadow_calibration_trace_set_v0/trace_index.json` exist and match their recorded SHA-256 hashes. Trace-set aggregate hash: `ab4fd8a70e2aa446e4996126a53685999f55a24baa2522a688ed72b0c2d5cfa0`. The frozen Q1 checker reports: `STAGE2A_QUALIFICATION: passed; eligible=0; physical_executions=0`.

No Q1 invocation or physical transition was performed by this audit. Stage 2A-E was not executed. The active Stage 2A branch remains stopped unless a genuinely new, independently motivated scientific question is preregistered later.
