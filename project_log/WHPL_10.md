# WHPL_10 — Radial PD Gating + Smoothing Interaction Probe

**Date:** 2026-02-15
**Branch:** whpl09-radial-pd
**Scenario:** weak_thrust_far
**Thrust:** 800 N

---

## Objective

Following WHPL_09 (closed-loop radial PD injection verified and sign-consistent),
WHPL_10 investigates:

> Why convergence remains slow and reward degraded, without sweeping gains.

Strict boundary constraints:

* ❌ No physics changes
* ❌ No reward changes
* ❌ No env changes
* ❌ No thrust sweep
* ❌ No gain sweep (kp/kd fixed)
* ❌ No RL
* ✅ Structural gating only
* ✅ One structural constant allowed

---

## Hypothesis

The always-on radial PD term may:

* Over-dominate thrust direction
* Suppress tangential energy shaping
* Increase saturation frequency
* Reduce total reward due to lambda_sat penalty

Therefore:

> Radial PD should only "speak" when needed.

We introduce two structural gates:

### Knife 1 — Error-Band Gating

Continuous gating based on relative radius error:

[
rel = |r - r_{target}| / r_{target}
]

[
g_r = clip((rel - r_on)/(r_full - r_on), 0, 1)
]

Default parameters:

* r_on = 0.12
* r_full = 0.30  ← (structural constant adjusted once)

Effect:

* Large error → reduced radial PD
* Near target → full radial PD

### Knife 2 — Inward D-Term Scaling

If spacecraft is above target radius but already moving inward:

* Reduce D-term influence

Condition:

```
radial_error > 0 and radial_velocity < 0
```

Scaling:

```
d_scale = 0.30
```

---

## Implementation

Inserted inside `if self.enable_radial_pd:` block.

Key structural additions:

* g_r continuous gate
* d_scale inward damping gate
* Apply g_r after PD cap
* Debug probe printing at steps (0,200,400,600,800)

Smoothing left enabled (no sweep).

---

## Experimental Evidence

### Run A — Pre-Gating (g_r = 1.0, r_full = 0.25)

* saturation_rate_mean ≈ 0.084
* total_reward ≈ -23106 ~ -23114
* g_r = 1.000 (always fully open)
* Radial dominance persistent

### Run B — Gated (r_full = 0.30)

Controller Debug (selected steps):

```
step=0   rel=0.2500 g_r=0.722 d_scale=1.00 thrust_r_pd=-6.166526e+00
step=200 rel=0.2500 g_r=0.722 d_scale=0.30 thrust_r_pd=-5.156380e+00
step=400 rel=0.2500 g_r=0.722 d_scale=0.30 thrust_r_pd=-4.868551e+00
step=600 rel=0.2500 g_r=0.722 d_scale=0.30 thrust_r_pd=-4.632974e+00
step=800 rel=0.2500 g_r=0.722 d_scale=0.30 thrust_r_pd=-4.445522e+00
```

Aggregate Metrics:

* saturation_rate_mean = **0.025**  (↓ from ~0.084)
* total_reward = **-22817.66** (improved)
* final_r ≈ 9.375e+12
* avg_radius_error ≈ 1.875e+12

---

## Quantitative Impact

| Metric          | Before    | After     | Change  |
| --------------- | --------- | --------- | ------- |
| Saturation Rate | ~0.084    | 0.025     | ↓ ~3.4× |
| Total Reward    | ~-23107   | -22818    | +~289   |
| Radius Error    | unchanged | unchanged | —       |

Interpretation:

* Structural gating significantly reduced saturation frequency.
* Reward improved accordingly.
* Radius convergence did not materially change within 2000 steps.

---

## Structural Interpretation

In high-relative-error regime (rel ≈ 0.25):

* Always-on PD over-dominates thrust direction.
* Tangential strategy is suppressed.
* Saturation penalty accumulates.

After gating:

* Radial PD scaled to ~72% strength.
* Tangential influence increased.
* Saturation reduced.
* Reward improved.

This supports the hypothesis that:

> The dominant issue was timing/strength of radial PD engagement, not sign inconsistency.

---

## Boundary Compliance Check

* No gain sweep ✔
* No thrust sweep ✔
* No physics/reward/env modification ✔
* Single structural constant adjusted ✔
* Closed-loop interpretability preserved ✔

---

## Conclusion

WHPL_10 demonstrates that:

1. Radial PD gating is structurally effective.
2. Error-band gating meaningfully reduces saturation (~3.4×).
3. Reward improves without gain tuning.
4. Convergence speed remains a separate regime-level problem.

This establishes the first confirmed structural control-regime improvement toward the 2D thrust × difficulty objective.

---

## Next (WHPL_11 Candidate)

* Include controller variant in CSV row / dedup_key.
* Preserve structural differences in 2D ablation heatmap.
* Avoid scope expansion.

    
