# Phase37B - Weak Tangential Subset Postmortem

## 1. Objective

Phase37B was run after Phase37A because Phase37A showed a narrow closest-approach signal in four `over_conservative_transfer` cases, but created `0` new crossings on the Phase36B baseline non-crossing set.

The question was deliberately limited:

Can a weak pre-cross tangential correction convert the four Phase37A-improved non-crossing cases into target-radius crossings while preserving the existing Phase36B crossing-producing cases?

Phase37B was a subset diagnostic, not a full controller. It did not introduce MPC, RL, a full tangential grid, or a broad planner framework.

## 2. Experiment Setup

Phase37B used:

- 4 selected Phase37A-improved non-crossing cases.
- 8 Phase36B baseline crossing-producing regression cases.
- 2 settings:
  - `early_commit_low_radial_only`
  - `early_commit_low_plus_weak_tangential`
- 24 total rollouts.
- Phase34 `radius_priority` as the fixed terminal/post-cross controller.
- No MPC.
- No RL.
- No full tangential grid.

The weak tangential correction was tested only as a bounded diagnostic perturbation before first target-radius crossing. It was not allowed to modify Phase34 post-cross synchronization, CAPTURE, LOCK, physics, or thresholds.

## 3. Results

Phase37B produced:

- selected-case new crossings: `0 / 4`
- selected-case recoverable crossings: `0 / 4`
- regression crossing preservation under weak tangential: `4 / 8`
- regression recoverable preservation under weak tangential: `4 / 8`
- overspeed: `0`
- instability: `0`
- closest-approach comparison against radial-only on selected cases: `3` improved, `0` worsened, `1` unchanged

The weak tangential perturbation moved closest approach slightly in several selected cases, but it did not create a target-radius crossing.

## 4. Regression Degradation

The weak tangential diagnostic did not preserve the Phase36B regression crossing set. Under the weak tangential setting, only `4 / 8` regression crossing-producing cases remained crossing and recoverable.

This makes the method unsafe as a global policy. A usable upstream controller must protect the known crossing-producing basin while attempting to expand the non-crossing basin.

This result does not invalidate Phase34 or Phase36B:

- Phase34 remains the fixed terminal/post-cross synchronization controller for crossing-producing cases.
- Phase36B remains the reference transfer-family benchmark showing `8 / 24` crossings and `8 / 24` recoverable crossings for each tested family.
- Phase37B only invalidates this specific early-commitment plus weak tangential perturbation as a candidate expansion mechanism.

## 5. Failure Analysis

### Radial Timing

Phase37A already showed that `early_commit_low` could improve closest approach in a narrow subset, but also degraded the existing crossing set. Phase37B inherited that risk. The regression degradation indicates that early radial commitment is not safe as a global pre-cross policy.

### Tangential Shaping

The weak tangential correction produced tiny closest-approach improvements in `3 / 4` selected cases, but no new crossings. This means tangential shaping may move local geometry metrics without producing the discrete geometric event that matters: target-radius crossing.

### Gating

Case-gating restricted the weak tangential term to the intended selected cases. That helped isolate the diagnostic question, but it does not solve generalization. The surrounding early-commitment diagnostic action still failed to preserve the known crossing-producing regression set.

### Handoff

No new Phase34-compatible crossings were created. Phase34 had no new downstream recovery opportunity because the selected cases never reached first target-radius crossing.

## 6. Interpretation

Phase37B is a negative diagnostic.

It suggests that small tangential correction is not enough to expand the crossing basin under the tested setup. It should not be expanded into a full tangential grid.

The most important lesson is that closest-approach improvement alone is not a sufficient success metric. A useful upstream search must produce actual target-radius crossings and must preserve the existing crossing-producing regression cases.

## 7. Phase38 Implications

Phase38 should not start by writing another controller.

The next step should be an evidence-based search-space definition:

- analyze why crossing-basin expansion keeps failing
- rank candidate variables based on Phase36B, Phase36C, Phase37A, and Phase37B evidence
- define the smallest justified next experiment before code
- protect the existing `8 / 24` crossing-producing cases as regression guards
- report geometric crossing and recoverable crossing separately

The project should not move directly to MPC-lite, RL, a full tangential grid, or a broad planner framework.

## 8. Bottom Line

Phase37B did not create new crossings and failed regression preservation. The next step is an evidence-based Phase38 search-space definition, not another heuristic controller.
