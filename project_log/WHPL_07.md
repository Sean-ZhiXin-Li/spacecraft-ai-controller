# WHPL_07 — Minimal 2×2 Grid to Expose Regime Boundary (Thrust × Difficulty)

**Date:** 2026-02-12
**Controller:** ExpertImproved
**Physics:** Frozen
**Reward:** Frozen
**Environment:** OrbitEnv (Voyager-scale, target_r = 7.5e12 m)

---

## 1. Objective

The objective of WHPL_07 is to expose the regime boundary between convergent and non-convergent orbital behavior using the *minimum possible additional sampling points*.

We freeze:

* Physical parameters
* Reward definition
* Controller logic

We only vary two axes:

1. **Thrust magnitude** (800 N vs 2000 N)
2. **Initial orbital radius offset** (Hard ≈ 1.25 × target_r vs Easy ≈ 1.05 × target_r)

The goal is not to sweep broadly, but to determine which axis defines the dominant regime separation.

---

## 2. Experimental Grid

Minimal 2×2 grid:

| Thrust (N) | Difficulty | r0_over_target |
| ---------- | ---------- | -------------- |
| 800        | Hard       | ~1.25          |
| 2000       | Hard       | ~1.25          |
| 800        | Easy       | ~1.05          |
| 2000       | Easy       | ~1.05          |

Implementation notes:

* Hard = default OrbitEnv reset (1.25× target radius)
* Easy injected via environment variable `R0_OVER_TARGET=1.05`

---

## 3. Execution Commands

### Hard runs

```powershell
Remove-Item Env:\R0_OVER_TARGET -ErrorAction SilentlyContinue
$env:DIFFICULTY_TAG="Hard"
$env:THRUST_NEWTON="800";  python src/quick_compare_v3_v4.py
$env:THRUST_NEWTON="2000"; python src/quick_compare_v3_v4.py
```

### Easy runs

```powershell
$env:R0_OVER_TARGET="1.05"
$env:DIFFICULTY_TAG="Easy"
$env:THRUST_NEWTON="800";  python src/quick_compare_v3_v4.py
$env:THRUST_NEWTON="2000"; python src/quick_compare_v3_v4.py
```

### Data pipeline

```powershell
python scripts/make_ablation_clean_csv.py
python scripts/validate_ablation_csv.py
python scripts/plot_ablation_heatmap_whpl07.py
```

Validation status:

* rows = 5
* dedup_unique = 5
* nan_total_reward = 0

---

## 4. Results (Clean CSV)

| Thrust (N) | Difficulty | Final Radius Error | Total Reward |
| ---------- | ---------- | ------------------ | ------------ |
| 800        | Hard       | ~1.87e12           | ~-2.08e04    |
| 2000       | Hard       | ~1.87e12           | ~-2.27e04    |
| 800        | Easy       | ~3.75e11           | ~-1.36e04    |
| 2000       | Easy       | ~3.75e11           | ~-1.52e04    |

Heatmap visualization confirms two distinct horizontal bands:

* Hard band ≈ 1.87e12
* Easy band ≈ 3.75e11

No visible vertical gradient across thrust axis.

---

## 5. Regime Boundary Analysis

### 5.1 Thrust Axis

Increasing thrust from 800 N to 2000 N does **not** significantly reduce final orbital radius error in either difficulty band.

Within the tested range:

* Hard(800) ≈ Hard(2000)
* Easy(800) ≈ Easy(2000)

Conclusion:

> There is no observable thrust threshold in the 800–2000 N range under the current controller structure.

---

### 5.2 Difficulty Axis (Initial Radius Offset)

Reducing initial radius offset from ~1.25 to ~1.05 reduces final radius error by approximately 5×:

1.87e12 → 3.75e11

Conclusion:

> The dominant regime separation is governed by initial-condition difficulty (r0_over_target), not thrust magnitude.

---

## 6. Structural Observation (Critical Insight)

Across all Easy runs:

* Final radius remains near initial radius (~1.05× target_r)
* Radius error remains nearly constant over time

This indicates:

> ExpertImproved does not produce effective radial convergence.

Thrust influences energy and velocity direction, but does not significantly drive radius toward target.

Therefore:

Increasing thrust cannot overcome structural limitations in control policy.

---

## 7. Final Conclusion of WHPL_07

WHPL_07 successfully eliminates the hypothesis that failure is caused by insufficient thrust.

The regime boundary is not defined by thrust magnitude.

Instead, it is defined by initial orbital radius offset.

This establishes that the current control architecture lacks effective radial correction capability.

---

## 8. Next Steps (WHPL_08 Candidates)

Two possible directions:

**A. Boundary Refinement**

* Fix thrust = 800 N
* Sweep r0_over_target: 1.10 / 1.15 / 1.20
* Identify transition band precisely

**B. Control Structure Analysis**

* Decompose thrust vector into radial vs tangential components
* Inspect whether controller produces meaningful radial thrust
* Evaluate reward shaping impact on radial convergence

WHPL_07 closes the minimal 2×2 grid and exposes the dominant regime axis.

Further expansion should be guided by this structural understanding, not by blind parameter sweeps.
