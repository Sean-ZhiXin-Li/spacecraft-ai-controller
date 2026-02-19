# WHPL_12 — Same-Coordinate Variant Contrast

Date: 2026-02-17
Project: Spacecraft AI Controller (2D Regime Mapping Track)

---

## 1. Theme

Same-Coordinate Variant Contrast

> Under an identical coordinate (thrust, difficulty, r0, scenario), determine whether different `controller_variant` labels produce measurable metric differences.

This is the first strict structural contrast experiment.

---

## 2. Hard Boundaries (Strictly Enforced)

No changes were made to:

* Controller logic
* Parameters
* Reward function
* Physics model
* Scenario
* Initial condition

Only modification:

* `CONTROLLER_VARIANT` label is now read from environment variable

```python
import os
CONTROLLER_VARIANT = os.getenv("CONTROLLER_VARIANT", "whpl11_variant_tracking")
```

No other structural or algorithmic changes were introduced.

---

## 3. Locked Coordinate

All runs were executed under the exact same coordinate:

* thrust_newton: 800.0 (env override)
* difficulty_tag: Hard
* r0_over_target: 1.2500000396629334
* scenario: weak_thrust_far
* controller: ExpertImproved
* steps: 2000

This coordinate was intentionally fixed to ensure pure structural contrast.

---

## 4. Experimental Method

Two consecutive runs were executed:

Run A:

```
CONTROLLER_VARIANT = "whpl12_always_on"
```

Run B:

```
CONTROLLER_VARIANT = "whpl12_gated"
```

Command used in both cases:

```
python src/quick_compare_v3_v4.py
```

No additional randomness or seed changes were introduced.

---

## 5. Evidence (CSV Extract)

File:

```
analysis/results/ablation_thrust_x_difficulty.csv
```

Relevant rows (same coordinate, different variant labels):

```
800.0,Hard,1.2500000396629334,1874961238019.5425,1874878662669.3965,0.0245,-22817.66169421043,59b69475,whpl12_always_on

800.0,Hard,1.2500000396629334,1874961238019.5425,1874878662669.3965,0.0245,-22817.66169421043,68605d9e,whpl12_gated
```

All coordinate fields match exactly.

---

## 6. Metric Comparison

| Metric               | always_on             | gated                 |
| -------------------- | --------------------- | --------------------- |
| total_reward         | -22817.66169421043    | -22817.66169421043    |
| avg_radius_error     | 1.8749612380195425e12 | 1.8749612380195425e12 |
| final_radius_error   | 1.8748786626693965e12 | 1.8748786626693965e12 |
| saturation_rate_mean | 0.0245                | 0.0245                |

Observed differences:

* controller_variant label
* dedup_key

No measurable metric differences detected.

---

## 7. Interpretation

At (thrust=800, Hard, r0=1.25, weak_thrust_far):

The structural distinction between `always_on` and `gated` does not manifest as a measurable dynamical difference under the currently logged metrics.

This suggests one of the following:

1. The gating condition is always active under this regime.
2. The gating condition is never active under this regime.
3. The structural change collapses to equivalent behavior in this coordinate region.

This is a scientifically valid negative result.

---

## 8. Scientific Significance

WHPL_12 establishes:

* Variant label control is operational and verifiable.
* Same-coordinate structural contrast is now experimentally reproducible.
* Structural equivalence in at least one regime is confirmed.

This transforms the project from "parameter accumulation" into "regime mapping".

---

## 9. Conclusion

Under identical coordinate conditions, the two controller variants are metrically indistinguishable.

Structural divergence is not active in this regime.

WHPL_12 is complete.

---

## 10. Next Step (WHPL_13 Preview)

Objective:

Identify the first coordinate at which structural divergence becomes measurable.

Method:

Repeat same-coordinate dual-variant contrast at a new coordinate, such as:

* thrust_newton = 200
* or modified r0_over_target
* or different difficulty level

Goal:

Detect the boundary of structural activation in 2D parameter space.

This is the foundation of the April 2D regime conclusion.
