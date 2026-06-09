# Phase37B — Weak Tangential Subset Diagnostic

## Scope

- Subset diagnostic only: 4 selected Phase37A-improved non-crossing cases plus 8 Phase36B baseline crossing-producing regression cases.
- Two settings:
  - `early_commit_low_radial_only`
  - `early_commit_low_plus_weak_tangential`
- Fixed Phase34 `radius_priority` terminal/post-cross controller after first target-radius crossing.
- No full 24-case tangential grid, no MPC, no RL, and no coast-duration search.
- Simplified 2D orbital-control sandbox, not real spacecraft validation.

---

## Control Perturbation

- **Radial overlay**: `u_r = clamp(0.055 * direction_to_target - 0.16 * vr_ratio, -0.055, 0.055)`  
- **Weak tangential overlay**: `u_t = clamp(-0.06 * vt_error_ratio, -0.020, 0.020)`  
- Combined diagnostic action norm limit: `0.060`  
- Weak tangential shaping acts only:
  - On the 4 selected `r0 = 1.02`, `thrust = 10000`, over_conservative_transfer cases
  - Before first target-radius crossing
  - Before Phase34 handoff
  - While outside the target-radius corridor  
- Does **not** act inside:
  - Phase34 post-cross synchronization
  - CAPTURE
  - LOCK
  - Already-crossed trajectories
  - Unrelated full-benchmark cases

---

## Aggregate Results

| Setting | Rollouts | Selected crossings | Selected recoverable | Regression crossings | Regression recoverable | Overspeed | Instability |
|---------|---------:|-----------------:|-------------------:|-------------------:|---------------------:|-----------|------------|
| early_commit_low_radial_only | 12 | 0 | 0 | 4 | 4 | 0 | 0 |
| early_commit_low_plus_weak_tangential | 12 | 0 | 0 | 4 | 4 | 0 | 0 |

- Total rollouts: 24  
- Selected-case new crossings: 0 / 4  
- Selected-case recoverable crossings: 0 / 4  
- Regression crossing preservation: 4 / 8 (baseline requirement: 8 / 8)  
- Overspeed: 0  
- Instability: 0  

---

## Closest-Approach Deltas (Selected Cases)

| Case ID | Delta | Effect |
|---------|------:|--------|
| selected_01 | 0.00000000 | unchanged |
| selected_02 | -1.04197805e-08 | improved |
| selected_03 | -1.27103552e-08 | improved |
| selected_04 | -1.36119823e-08 | improved |

Interpretation: weak tangential shaping slightly improved some trajectories’ closest approach, but did not produce new crossings.

---

## Decision

- Diagnostic classification: **unsafe globally**  
- Interpretation: The weak tangential action did **not** preserve the Phase36B baseline regression crossing set.  

**Recommendation**: Do not expand Phase37B unless:
- At least one new selected-case crossing appears
- Regression set remains intact

If only closest approach improves, this is a weak signal requiring further analysis before any larger tangential search.

---

## Artifacts

- `analysis/phase37b_weak_tangential_subset/phase37b_results.csv`
- `analysis/phase37b_weak_tangential_subset/phase37b_summary.md`
- `analysis/phase37b_weak_tangential_subset/phase37b_comparison.png`

> Note: The CSV and PNG are currently ignored by git. If intended as public evidence, add targeted unignore rules or force-add.

---

## Notes

- Phase37B subset diagnostic confirms that **radial + weak tangential shaping** is insufficient to generate new crossings.
- The experiment is strictly a diagnostic on a subset and **does not affect the global Phase34/36B crossing set** beyond what is measured.
- Provides causal evidence to guide whether future tangential shaping is justified for Phase37C or beyond.