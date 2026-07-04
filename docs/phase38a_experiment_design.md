# Phase38A Experiment Design

Status: design only. This document proposes a future experiment. It does not implement a controller, run a new benchmark, change physics, change thresholds, or modify historical artifacts.

## 1. Objective

Phase38A should test the smallest justified upstream crossing-generation variable after Phase36B, Phase36C, Phase37A, and Phase37B.

The objective is not to improve Phase34 post-cross recovery. Phase34 already improves simulator-defined recoverability for crossing-producing cases. The unresolved problem is creating new target-radius crossings on the Phase36B baseline non-crossing cases while preserving the existing `8 / 24` crossing-producing cases.

## 2. Evidence Basis From Phase36B/C, Phase37A, Phase37B

Phase36B evidence:

- Four tested transfer families each produced `8 / 24` crossings and `8 / 24` recoverable crossings.
- No tested transfer family expanded crossing beyond the Phase34 baseline.
- Source: `analysis/phase36b_transfer_family_benchmark/summary.md`.

Phase36C evidence:

- Phase36B baseline had `16 / 24` non-crossing cases.
- Baseline non-crossing labels split into `8` near-crossing cases and `8` over-conservative-transfer cases.
- Family variants moved closest-approach and crossing-potential metrics without producing new target-radius crossings.
- Source: `analysis/phase36c_non_crossing_geometry_diagnosis/summary.md`.

Phase37A evidence:

- Six radial commitment variants across `144` rollouts created `0` new crossings on baseline non-crossing cases.
- Delayed variants preserved `8 / 24` crossings and recoverable crossings.
- Early and mid variants degraded the existing crossing set.
- Source: `analysis/phase37a_radial_commit_timing/phase37a_summary.md`.

Phase37B evidence:

- Weak tangential shaping created `0 / 4` selected-case crossings.
- It preserved only `4 / 8` regression crossings.
- Closest approach improved in `3 / 4` selected cases, but this was not sufficient because no new target-radius crossing occurred and regression preservation failed.
- Source: `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`.

## 3. Candidate Variable Chosen

Chosen candidate variable: coast duration / commit-window timing.

This should be treated as a small, bounded design variable, not as a broad planner search.

Rationale:

- Phase36C indicates that some non-crossing cases are over-conservative rather than unstable or overspeed-limited.
- Phase37A shows that commitment timing changes the behavior of closest approach and regression preservation.
- Delayed commitment variants preserved the existing crossing-producing cases better than early or mid commitment.

## 4. Why This Variable Is More Justified Than Rejected Variables

| Candidate variable | Decision | Reason |
|---|---|---|
| Coast duration / commit-window timing | Selected | Most consistent with Phase36C over-conservative-transfer diagnosis and Phase37A delayed-commit preservation. |
| Broad radial timing sweep | Rejected for Phase38A | Phase37A already tested radial timing and created `0` new crossings; repeating a larger radial-only sweep is low ROI. |
| Tangential corridor shaping | Rejected for Phase38A | Phase37B created `0 / 4` selected crossings and damaged regression preservation. |
| Angular momentum correction | Deferred | Physically plausible, but existing evidence is weaker and risks becoming another heuristic without a bounded design. |
| Full planner search | Rejected | Too broad before submission and inconsistent with Phase38 design-first discipline. |
| MPC, RL, 3D, C++, SPICE | Rejected | Out of scope for Phase38A and not justified by current crossing-basin evidence. |

## 5. Selected Cases

Selected non-crossing design cases should come from the Phase36C over-conservative-transfer group, with priority on the Phase37B selected cases:

- `r0_over_target=1.02`, angle `150`, thrust `10000`
- `r0_over_target=1.02`, angle `165`, thrust `10000`
- `r0_over_target=1.02`, angle `170`, thrust `10000`
- `r0_over_target=1.02`, angle `175`, thrust `10000`

These cases are useful because they showed diagnostic movement in Phase37A/37B but did not cross.

Do not present these as representative of the full 24-case benchmark unless Phase38A is later run on the full benchmark with regression guards.

## 6. Regression Guard Cases

Phase38A must protect the existing `8 / 24` crossing-producing cases from the Phase36B/Phase34 baseline.

Regression requirement:

- Preserve target-radius crossing on all eight known crossing-producing cases.
- Preserve recoverable crossing on all eight known crossing-producing cases.
- Preserve `0` overspeed and `0` instability.

Any candidate that creates a selected-case crossing but destroys the existing crossing-producing set should be treated as not globally usable.

## 7. Parameters

Phase38A should define parameters before implementation. Suggested design-only parameter family:

| Parameter | Role | Design constraint |
|---|---|---|
| `coast_duration_steps` | Delay or coast interval before radial commitment. | Small discrete set only; no broad grid. |
| `commit_window_start` | Time or state condition for beginning commitment. | Must be fixed before any run. |
| `commit_window_end` | Time or state condition for ending commitment. | Must not be tuned after seeing selected-case outcomes. |
| `radial_magnitude_label` | Reuse known safe labels where possible. | Do not invent stronger radial magnitudes before design review. |
| `terminal_controller` | Fixed downstream controller. | Must remain Phase34 `radius_priority`. |

No full 4D grid is allowed in Phase38A.

## 8. Metrics

Primary metrics:

- target-radius crossing count;
- recoverable crossing count;
- selected-case new crossings;
- regression crossing preservation;
- regression recoverable-crossing preservation;
- overspeed count;
- instability count.

Diagnostic metrics:

- closest approach;
- crossing potential;
- radial velocity error;
- tangential velocity error;
- crossing sync error.

Diagnostic metrics may explain why a candidate failed or merits later testing. They must not be counted as success.

## 9. Success Criteria

Phase38A would justify implementation only if the design can specify a small search that, when later implemented, would be judged by these criteria:

- At least one selected non-crossing case becomes an actual target-radius crossing.
- Any new crossing is evaluated for recoverable crossing under fixed Phase34 `radius_priority`.
- All eight regression crossing-producing cases remain crossings.
- All eight regression crossing-producing cases remain recoverable crossings.
- Overspeed remains `0`.
- Instability remains `0`.
- The result is reported separately for selected cases and regression cases.

Closest-approach improvement alone is not success.

## 10. Stopping Criteria

Stop Phase38A design or implementation if:

- the parameter set grows into a broad planner search;
- the design requires changing physics, thresholds, or Phase34 terminal behavior;
- the design requires MPC, RL, 3D, C++, SPICE, or a full 4D grid;
- selected-case metrics are being optimized without regression guards;
- closest approach is being treated as equivalent to crossing.

## 11. No-Go Criteria

No-go for implementation if:

- the design cannot protect the existing eight crossing-producing cases;
- the design cannot define a small pre-registered parameter set;
- the only expected improvement is closest approach or crossing potential;
- implementation would overwrite historical artifacts;
- the design depends on post-hoc tuning or manual case-by-case intervention.

## 12. What Result Would Justify Implementation

Implementation would be justified only after the design is reviewed and remains:

- small;
- source-backed;
- regression-protected;
- compatible with fixed Phase34 terminal behavior;
- explicit about primary metrics versus diagnostic proxies.

The planned implementation must write any new outputs to a new Phase38A artifact directory and must not overwrite Phase34, Phase36B, Phase36C, Phase37A, or Phase37B artifacts.

## 13. What Result Would Cancel Phase38A

Cancel Phase38A if design review concludes that:

- coast duration / commit-window timing is only repeating Phase37A;
- the selected variable is not separable from rejected tangential or broad planner variables;
- the existing `8 / 24` crossing-producing regression set cannot be protected;
- the proposal requires broad search before a small diagnostic test;
- the only plausible win condition is improved closest approach without actual crossing.

## Bottom Line

Phase38A is a design-only step. Current status is GO for design review and NO-GO for controller implementation until the design is approved with fixed selected cases, fixed regression guards, fixed parameters, and primary success criteria based on actual target-radius crossings and recoverable crossings.
