# Phase38 - Evidence-Based Search Space

## 1. Purpose

Phase38 is an understanding and planning phase before new controller code.

The goal is to determine why crossing-basin expansion has continued to fail in the simplified 2D orbital-control sandbox. Phase38 should define the smallest justified next experiment, not start with a new heuristic controller.

The fixed downstream assumption remains:

- Phase34 `radius_priority` is the terminal/post-cross controller.
- Physics, CAPTURE/LOCK thresholds, recoverability thresholds, and termination checks remain unchanged.
- Geometric crossing and recoverable crossing must be reported separately.

## 2. Evidence Summary

Phase36B tested `baseline_phase34`, `spiral_approach`, `grazing_corridor`, and `redesigned_delayed_crossing` on the full 24-case reduced benchmark. Every family produced `8 / 24` geometric crossings and `8 / 24` recoverable crossings. No family expanded the crossing basin.

Phase36C isolated the `16 / 24` baseline non-crossing cases. The baseline labels split into `8` `near_crossing` cases and `8` `over_conservative_transfer` cases. Across Phase36B families, closest-approach and crossing-potential metrics moved, but no new target-radius crossings appeared.

Phase37A tested radial commitment timing and bounded radial magnitude across `144` rollouts. It created `0` new crossings on the Phase36B baseline non-crossing cases. `delayed_commit_low` and `delayed_commit_medium` preserved `8 / 24` crossings and `8 / 24` recoverable crossings, while early and mid commitment degraded the existing crossing set.

Phase37B tested a weak tangential perturbation on a narrow subset: four Phase37A-improved `over_conservative_transfer` cases plus eight Phase36B regression crossing cases. It created `0 / 4` selected-case crossings, `0 / 4` selected-case recoverable crossings, and preserved only `4 / 8` regression crossings under weak tangential shaping. It produced no overspeed or instability, but it failed regression preservation.

## 3. Candidate Variable Ranking

### Rank 1 - Coast Duration / Commit Window Timing

Evidence supporting it:

- Phase36C identified `over_conservative_transfer` as one of the two dominant non-crossing labels.
- Phase36C showed closest-approach and crossing-potential movement without crossings, suggesting timing and commitment windows may matter.
- Phase37A showed that changing when radial commitment becomes active changes closest-approach behavior.

Evidence against it:

- Phase37A radial timing alone created `0` new crossings.
- Early and mid commitment degraded the existing crossing set.

Risks:

- A timing variable can overfit selected non-crossing cases while damaging the `8 / 24` crossing-producing regression set.
- It may move closest approach without producing crossing.

Allowed in next experiment:

- Allowed only as a small, written-design search variable.
- It should be tested with regression guards and strict no-tuning discipline.

Postponed:

- A full coast-duration grid is postponed.

### Rank 2 - Angular Momentum Correction

Evidence supporting it:

- Phase36C failure labels and family deltas suggest that geometry metrics can move without crossing; angular momentum may be one missing corridor variable.
- Tangential velocity and angular momentum are physically linked in the 2D orbital setting.
- Phase36B family logic already used angular-momentum-related shaping in limited ways, but did not isolate it as a controlled variable.

Evidence against it:

- No existing phase has shown that direct angular momentum correction creates new crossings.
- If coupled too strongly to tangential correction, it may reproduce Phase37B-style regression degradation.

Risks:

- Can become another named heuristic if not parameterized carefully.
- May improve local metrics while failing to create crossings.

Allowed in next experiment:

- Allowed only as an analysis variable or one bounded diagnostic variable after a written design.

Postponed:

- Broad angular-momentum control policies are postponed.

### Rank 3 - Tangential Corridor Shaping

Evidence supporting it:

- Phase36C kept tangential corridor quality as a diagnostic concern.
- Phase37B weak tangential shaping slightly improved closest approach in `3 / 4` selected cases.

Evidence against it:

- Phase37B created `0 / 4` selected-case crossings.
- Phase37B preserved only `4 / 8` regression crossings and recoverable crossings under the weak tangential setting.
- The measured closest-approach deltas were tiny and not sufficient to justify success claims.

Risks:

- Can damage the existing crossing-producing set.
- Can inflate interpretation if closest-approach improvement is treated as success.

Allowed in next experiment:

- Not allowed as the primary next search variable unless a new analysis shows stronger evidence than Phase37B.

Postponed:

- Full 24-case tangential grid is postponed.

### Rank 4 - Radial Timing / Radial Magnitude

Evidence supporting it:

- Phase37A found small closest-approach improvements in a narrow `over_conservative_transfer` subset.

Evidence against it:

- Phase37A created `0` new crossings on the `16` baseline non-crossing cases.
- Medium radial magnitude collapsed crossing performance.
- Early and mid commitment degraded the existing crossing set.

Risks:

- Blind expansion repeats a tested negative result.
- Stronger radial action may further damage crossing-producing cases.

Allowed in next experiment:

- Not allowed as a standalone next search variable.

Postponed:

- Additional radial-only timing or magnitude sweeps are postponed.

## 4. Case Prioritization

Most useful diagnostic cases:

- The selected `over_conservative_transfer` cases from Phase37B:
  - `r0_over_target=1.02`, angle `150`, thrust `10000`
  - `r0_over_target=1.02`, angle `165`, thrust `10000`
  - `r0_over_target=1.02`, angle `170`, thrust `10000`
  - `r0_over_target=1.02`, angle `175`, thrust `10000`

These cases improved closest approach under Phase37A/37B diagnostics but did not cross. They are useful for understanding why metric movement fails to become a discrete crossing event.

Cases requiring caution:

- The `near_crossing` cases at `r0_over_target=0.98`.

Phase37A worsened these cases under radial commitment variants. They should not be used to justify stronger radial timing.

Regression guard cases:

- The eight Phase36B baseline crossing-producing cases at `r0_over_target=1.00`.

Any future search must protect these cases. A method that creates a new crossing but destroys the known crossing set is not an acceptable global architecture.

## 5. What NOT To Test Yet

Do not test:

- MPC-lite
- RL / PPO / SAC / DDPG
- 3D dynamics
- C++ rewrites
- SPICE integration
- full 4D grid
- full 24-case tangential grid
- uncontrolled broad planner search

These directions are premature because the current evidence does not yet identify a reliable upstream crossing-generation variable.

## 6. Minimum Next Experiment Requirements

The next experiment must:

- be justified by Phase36B, Phase36C, Phase37A, and Phase37B evidence
- have a written design before code
- protect the 8 baseline crossing-producing cases
- include a regression guard
- report geometric crossing and recoverable crossing separately
- report overspeed and instability
- avoid treating closest-approach improvement as success unless crossings improve
- keep Phase34 `radius_priority` fixed as the terminal/post-cross controller

## 7. Go / No-Go Criteria

GO only if:

- a candidate variable has stronger evidence than Phase37B weak tangential shaping
- the design explains how it protects the `8 / 24` baseline crossing-producing cases
- the search space is small enough to run once without post-hoc tuning

NO-GO if:

- evidence only shows tiny closest-approach deltas without crossings
- regression crossing cases are likely to degrade
- the experiment requires a full 4D grid or broad planner framework
- the proposal treats crossing potential or closest approach as equivalent to crossing

## 8. Bottom Line

Phase38 should be an understanding phase, not a controller-writing phase.
