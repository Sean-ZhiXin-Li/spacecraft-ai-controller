# Milestones

This folder is reserved for curated milestone documentation. The detailed result directories remain under `analysis/` so existing scripts and README links keep working.

## Current Milestone

The current milestone is **Stage 2B Numerical Propagation Independence Validation v0**.

Stage 2B introduced an independent SciPy DOP853 propagator to test whether the frozen Phase34/35 one-step Final Veto classifications depend on the baseline semi-implicit Euler implementation.

Frozen validation coverage:

- 4 exact Phase35 states;
- 4 nominal proposals;
- 10 evaluated physical alternative proposals;
- 14 total one-step proposal evaluations.

Result:

- Euler / DOP853 classification agreement: `14 / 14`;
- classification mismatches: `0`;
- maximum observed absolute speed-ratio difference:
  `2.886579864025407e-15`;
- focused Stage 2B tests: `5 / 5 PASS`;
- Stage 2B manifest integrity: `PASS`.

The validated claim is intentionally narrow. This establishes one-step numerical classification consistency only within the audited exact-state coverage. It does not establish multi-step trajectory independence, physical-model fidelity, or real-spacecraft safety.

Stage 2B v0 was frozen in commit:

`cf887c87fa4f6323fd7cbff0069913f4dd483c64`

Current references:

- [Stage 2B summary](../../analysis/stage2b_numerical_validation_v0/summary.md)
- [Stage 2B results](../../analysis/stage2b_numerical_validation_v0/results.json)
- [Stage 2B manifest](../../analysis/stage2b_numerical_validation_v0/manifest.json)
- [Project logs index](../project_logs_index.md)
- [Research direction](../research_direction.md)

## Next Planned Validation

The next planned numerical-validation task is **Stage 2B-M Multi-Step Numerical Divergence Validation**.

The goal is to measure how Euler / DOP853 state differences accumulate over repeated 100 s control intervals before making any broader numerical-independence claim.

Initial comparison horizons:

- 1;
- 2;
- 4;
- 8;
- 16;
- 32;
- 64;
- 128 control intervals.

Candidate metrics include:

- position divergence;
- velocity divergence;
- radius difference;
- speed-ratio drift;
- overspeed classification;
- crossing status;
- recovery-related component consistency.

This is currently a planned experiment, not a completed result.

## Previous Research Trail

The earlier Phase34-Phase38 sequence remains part of the scientific history:

- Phase34 established the fixed terminal/post-cross controller.
- Phase36B tested transfer-family variants.
- Phase36C diagnosed non-crossing cases.
- Phase37A tested radial commitment timing and magnitude.
- Phase37B tested weak tangential shaping and produced a negative diagnostic result.
- Phase38 defined an evidence-based search direction before the later runtime-assurance work.

Historical references:

- [Phase36B summary](../../analysis/phase36b_transfer_family_benchmark/summary.md)
- [Phase36C summary](../../analysis/phase36c_non_crossing_geometry_diagnosis/summary.md)
- [Phase37A summary](../../analysis/phase37a_radial_commit_timing/phase37a_summary.md)
- [Phase37B summary](../../analysis/phase37b_weak_tangential_subset/phase37b_summary.md)
- [Phase37B postmortem](../../project_log/phase37b_weak_tangential_postmortem.md)
- [Phase38 evidence-based search space](../phase38_evidence_based_search_space.md)

## Earlier Local-Controller Milestone

Phase7.6 remains the strongest local 2D explicit-controller result and should be treated as earlier controller evidence, not the current project frontier:

- [Phase7.6 Soft Hybrid Summary](../../analysis/phase76_soft_hybrid/phase76_summary.md)
- [PL22-PL27 project log trail](../project_logs_index.md#earlier-local-controller-milestone)