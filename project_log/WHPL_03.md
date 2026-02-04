# WHPL_03 — One-Day Engineering Log

**Date:** 2026-02-04  
**Type:** Single-day, freeze-safe WHPL  
**Inherited from:** WHPL_02 (interface compression confirmed)

---

## Only Engineering Question

Can one experiment run be materialized as **exactly one persistent CSV row**  
(i.e. the first concrete carrier of a 2D result)?

Formally:

> Does *one run = one row* hold in practice?

---

## What Was Done (Facts Only)

- Modified `src/quick_compare_v3_v4.py` to introduce an **append-only CSV output path**
- Fixed today’s experiment coordinate:
  - Controller: `ExpertImproved`
  - Scenario: `weak_thrust_far`
  - Thrust: `800 N`
  - Difficulty tag: `Hard`
- Ran **exactly one episode**
- Appended **exactly one row** to:

```
analysis/results/ablation_thrust_x_difficulty.csv
```

No plots, no parameter sweeps, no controller comparison.

---

## CSV Schema (Frozen Today)

```csv
thrust_newton,
difficulty_tag,
r0_over_target,
avg_radius_error,
final_radius_error,
saturation_rate_mean,
total_reward
```

---

## Row Written Today

```text
thrust_newton          = 800.0
difficulty_tag         = Hard
r0_over_target         ≈ 1.25000004
avg_radius_error       ≈ 1.874990e+12
final_radius_error     ≈ 1.874958e+12
saturation_rate_mean   = 0.0
total_reward           ≈ -2.08e+04
```

---

## Verification

- CSV file exists ✅
- Header written once, not duplicated ✅
- One run produced exactly one appended row ✅
- Re-running the script appends a new row without overwriting ✅

---

## Conservative Conclusion

Evidence suggests that an experiment outcome can now be represented as a **persistent, append-only 2D data point**.

---

## Explicit Non-Goals (Respected)

- No OrbitEnv modifications  
- No controller performance comparison  
- No interpretation of convergence or failure  
- No visualization  
- No multi-day planning  

WHPL_03 resolves **only the output-materialization problem**.
