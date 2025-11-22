# NEW_WEEK_3 Failure Catalog — ProjectLog Style

This catalog summarizes the three engineered failure scenarios generated in Week 3:
**weak_thrust_far**, **oscillation_noise**, and **misaligned_entry**.
Each section follows a consistent ProjectLog style: narrative + technical diagnosis + energy‑view interpretation + controller weaknesses + preliminary fixes.

---

## 1. Weak Thrust Far  
*Scenario: target=7.5e12 m, weak thrust (scale=0.1), start near 2× target radius*

### Narrative Summary
The spacecraft begins far outside the target orbit with insufficient thrust to meaningfully modify orbital energy.  
From the XY plot, the spacecraft barely moves relative to the vast target circle.  
Radial error decreases only microscopically across 20,000 steps.

### Observable Evidence
- Trajectory cluster barely moves on XY plot.
- Radial error decreases from approx **3.75e12 → 3.725e12**, a tiny ~0.7% shift.
- Thrust histogram shows almost all thrust magnitude near **1.0**, but because the scale is 0.1, the effective acceleration is extremely small.

### Energy‑View Metrics
- `energy_convergence_step = 97`
- `energy_drift_percent ≈ 3080%` (huge drift due to tiny thrust unable to stabilize conditions)
- `angular_momentum_error_final ≈ 0.497`
- `thrust_energy_ratio ≈ 894` (controller overuses thrust relative to the orbital energy actually changed)
- `energy_oscillation_index ≈ 5e-05`

### Diagnosis
The controller continuously attempts corrective actions, but the physical acceleration is too small;  
thus, almost no orbital energy transfer occurs. The spacecraft essentially “hovers” around its initial orbit.

### Controller Weakness
- The controller assumes nominal thrust and does not re‑evaluate feasibility under thrust‑limited conditions.
- No mode-switching for “low-thrust long-burn” strategies.

### Possible Fixes
- Add low-thrust mode using continuous tangential burn strategy.
- Add detection for “insufficient thrust regime” and increase planning horizon.
- Introduce energy‑gradient based controller to maintain progress even under micro‑thrust.

---

## 2. Oscillation Noise  
*Scenario: nominal thrust, but velocity vector heavily perturbed by noise each step*

### Narrative Summary
This scenario injects random oscillations in direction commands.  
In the XY plot, the spacecraft begins far above the target orbit and drifts slowly while experiencing random micro‑corrections.

### Observable Evidence
- XY plot shows spacecraft starting far outside the orbit.
- Radial error drifts downward gradually, very smooth due to noise averaging.
- Thrust histogram shows a broad distribution (0.1–1.4+), indicating unstable control effort.

### Energy‑View Metrics
- `energy_convergence_step = 7` (false convergence; noise confuses the metric)
- `energy_drift_percent ≈ 11.86%`
- `angular_momentum_error_final ≈ 0.240`
- `thrust_energy_ratio ≈ 1880`
- `energy_oscillation_index ≈ 0.498` (very high)

### Diagnosis
Noise dominates the correction direction, so the thrust vector does not align cleanly with the desired tangential orbit-raising direction.  
Energy oscillates instead of converging.

### Controller Weakness
- Controller lacks noise filtering.
- No angular‑momentum smoothing.
- No “trust region” for direction control (RL typically uses these).

### Possible Fixes
- Add moving‑average filter on commanded direction.
- Add angular-momentum regularization (penalize frequent direction flips).
- Add “thrust‑smoothing” to limit jerk.

---

## 3. Misaligned Entry  
*Scenario: initial velocity direction offset by large angle, causing an “elliptical graze” pattern*

### Narrative Summary
The spacecraft enters the environment with a velocity direction far from tangent.  
This produces a slow spiral pattern where radial error grows, not shrinks.

### Observable Evidence
- XY plot shows nearly elliptical motion before controller stabilizes direction.
- Radial error increases stepwise across 20,000 steps.
- Thrust histogram shows near‑zero values for most steps, indicating the controller frequently “gives up”.

### Energy‑View Metrics
- `energy_convergence_step = 141`
- `energy_drift_percent ≈ -0.0095%` (tiny drift)
- `angular_momentum_error_final ≈ -0.0048`
- `energy_oscillation_index ≈ 5e-05`
- `thrust_energy_ratio ≈ 1106`

### Diagnosis
The misaligned velocity produces an initial radial oscillation.  
The controller tries to cancel it, but thrust is not applied in a consistent angular-momentum correcting direction.  
Thus, orbital energy barely changes, and radial error accumulates.

### Controller Weakness
- Controller lacks a mechanism for correcting large initial angular momentum mismatch.
- No “insertion phase” logic to align velocity before pursuing orbit-shaping.
- Overly conservative thrust usage at early steps.

### Possible Fixes
- Add “entry correction” mode that aligns velocity tangent before mission begins.
- Add reward shaping for angular-momentum matching.
- Add detection of high e (eccentric) initial orbits and trigger a dedicated circularization sub-controller.

---

## Week 3 Summary

Weak-thrust, noise-driven, and misaligned-entry failures each expose a different blindspot in the current ExpertController:

1. **Weak-thrust** → lacks low-thrust strategy  
2. **Oscillation-noise** → lacks smoothing and trust-region controls  
3. **Misaligned-entry** → lacks velocity-alignment / circularization logic  

Together, these failures map out exactly the “robustness envelope” the Week 4–5 controller must address.

---

*End of NEW_WEEK_3 Failure Catalog*

