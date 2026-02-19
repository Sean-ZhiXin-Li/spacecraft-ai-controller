# WHPL_09 — Radial Error Feedback Injection

**Date:** 2026-02-14
**Branch:** `whpl09-radial-pd`

---

## Objective

Introduce an explicit radial PD feedback term into `ExpertControllerImproved` to transform the radial channel from a heuristic open-loop correction into a structurally interpretable closed-loop system.

Hard constraints:

* No physics modification
* No reward modification
* No thrust sweep
* No action interface change
* Controller-layer minimal patch only

---

## Hypothesis

If a bounded radial PD term is injected directly into `thrust_r`, then:

1. `thrust_r_pd` will respond consistently to `(r - r_target)` and `v_r`
2. When `r_err > 0`, the PD term should produce inward thrust
3. As `v_r` becomes negative (inward motion), the PD magnitude should decay (damping behavior)
4. Radius should exhibit a measurable downward trend over time

---

## Implementation

### Injection Location

Inside `__call__()` immediately after base radial correction:

```
thrust_r = -self.radial_gain * tanh(...)
thrust_r += thrust_r_pd
```

### PD Structure

```
p_term = -k_p * tanh(r_err / scale)
d_term = -k_d * tanh(v_r / scale_v)
thrust_r_pd = clip(p_term + d_term) * cap
```

### Safety

* Bounded by `radial_pd_cap_frac * thrust_limit`
* No modification to tangential channel
* No modification to smoothing, scheduler, or alignment blocks

---

## Evidence (selected steps)

### step = 0

```
r=9.375000297472e+12
r_err=1.875000164352e+12
v_r=+1.881502e+03
thrust_r_pd=-8.538265e+00
```

Interpretation: positive radial error + outward velocity → inward corrective thrust.

---

### step = 200

```
v_r=-3.345664e+02
thrust_r_pd=-6.958549e+00
```

Velocity flips sign (now inward). PD magnitude decreases.

---

### step = 400

```
v_r=-2.430924e+03
thrust_r_pd=-5.482762e+00
```

Increasing inward motion → smaller corrective magnitude.

---

### step = 600

```
v_r=-4.096503e+03
thrust_r_pd=-4.404481e+00
```

Clear damping trend.

---

### step = 800

```
v_r=-5.430059e+03
thrust_r_pd=-3.633475e+00
```

PD continues decaying as inward velocity grows.

---

## Radius Trend

* r(step=0)   = 9.375000297472e+12
* r(step=1800)= 9.374904877056e+12

Decrease ≈ 9.54e7 meters (~95,000 km)

Note: `radius_error stats` (min/mean/max) remain visually constant due to 1e12-scale rounding; exponential-format prints reveal the true trend.

---

## System Behavior Observations

* `thrust_r_pd` sign is consistent with closed-loop expectations
* PD magnitude decays over time → damping-like behavior
* Saturation rate increased (~0.067)
* Total reward worsened (due to current shaping)

This indicates structural correction works, but reward alignment and gating require later refinement.

---

## Conclusion

WHPL_09 successfully converts the radial channel from heuristic correction into an explicitly interpretable bounded PD closed-loop structure.

The system now demonstrates:

* Correct sign response to radial error
* Damping response to radial velocity
* Measurable inward radius trend

Closed-loop radial behavior is now structurally established.

---

## Next (WHPL_10 Preview)

Investigate why convergence remains slow and reward degrades:

* Possible structural gating of radial PD
* Interaction with smoothing block (alpha=0.05)
* Radial/tangential coordination

No parameter sweep planned.
