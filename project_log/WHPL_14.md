# WHPL14 – Session Log

Date: 2026-02-17
---

# Objective

Complete the 2×2 ablation matrix comparing:

* Controller Variant: gated vs always_on
* Thrust Level: 800N vs 2000N
* Initial Radius Offset: r0 ∈ {1.005, 1.05, 1.20}

Primary goals:

1. Verify structural correctness of radial gating across thrust scales.
2. Determine whether thrust scaling yields monotonic performance improvement.
3. Identify behavioral differences between gated and always_on control injection.

---

# Experimental Configuration

Environment:

* Scenario: weak_thrust_far
* Target radius: 7.5e12
* Simulation length: 2000 steps

Control variants:

* gated (radial PD scaled by g_r)
* always_on (radial PD always injected)

Thrust levels:

* 800 N
* 2000 N

Initial offsets:

* r0 = 1.005 × target
* r0 = 1.05 × target
* r0 = 1.20 × target

---

# Structural Verification of Gating

Across both thrust levels, the gating behavior remains consistent:

Regime A – Near Target (rel ≈ 0.005)

* g_r = 0
* thrust_r_pd = 0 (gated only)

Regime B – Moderate Error (rel ≈ 0.05)

* g_r = 0.10 (floor activation)
* small PD injection

Regime C – Large Error (rel ≈ 0.20)

* g_r ≈ 0.444
* partial linear injection

Conclusion:
Gating structure is invariant under thrust scaling. Increasing thrust does not alter regime logic.

---

# Performance Summary

## 1) Near Target (r0 = 1.005)

800N:

* gated: negative reward
* always_on: similar but slightly worse

2000N:

* gated: +5380 (positive reward)
* always_on: -1123 (negative reward)

Observation:
Under high thrust, gated control significantly improves reward near equilibrium. always_on continues injecting PD despite minimal error and degrades performance.

Interpretation:
Near equilibrium, non‑intervention yields better global stability than aggressive correction.

---

## 2) Moderate Error (r0 = 1.05)

800N:

* gated: lower action norm, no saturation
* always_on: higher action norm, saturation ≈ 0.021

2000N:

* gated: no saturation
* always_on: saturation ≈ 0.009

Observation:
always_on consistently produces larger control magnitude and occasional clipping. gated remains conservative.

Conclusion:
Gating reduces unnecessary control energy and prevents saturation events.

---

## 3) Large Error (r0 = 1.20)

Increasing thrust from 800N to 2000N does not improve performance.

Observed dynamics:

* Rapid radial velocity sign flip
* Large inward plunge
* Increasing negative reward magnitude

Conclusion:
Higher thrust induces overcorrection and amplifies nonlinear dynamics rather than improving convergence.

---

# Key Findings

1. Gating preserves regime structure independent of thrust level.
2. always_on bypasses effective gating behavior and injects PD even when g_r = 0.
3. Thrust scaling is non‑monotonic:

   * Improves performance near equilibrium
   * Degrades performance at moderate and large offsets
4. Saturation events are more frequent under always_on control.
5. Intelligent non‑intervention (g_r = 0) improves stability near target orbit.

---

# Theoretical Interpretation

This system exhibits nonlinear response characteristics:

* Near equilibrium, small corrective impulses disrupt orbital balance.
* At moderate offsets, controlled injection (floor‑gated PD) prevents overshoot.
* At large offsets, higher thrust accelerates inward plunge, amplifying velocity penalties.

The results indicate that optimal control strength depends on state regime. Uniformly increasing thrust does not uniformly improve performance.

---

# Session Conclusion

The 2×2 thrust × controller ablation is complete.

The experiment successfully demonstrates:

* Structural validity of the gating mechanism
* Clear behavioral divergence between gated and always_on
* Non‑monotonic scaling of performance with thrust

The system exhibits regime‑dependent stability domains.

Next step suggestion:
Visualize total_reward vs r0 for both thrust levels and controller variants to illustrate stability regions graphically.

---

