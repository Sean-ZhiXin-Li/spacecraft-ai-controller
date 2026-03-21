# PL_W03 — r0 Sweep & High-Thrust Diagnosis

## Objective

Test sensitivity of controller separation to initial orbital offset (r0), and determine whether the observed behavior in W02 persists under different initial conditions.

---

## Key Question

> Does the parallel branch structure (always_on vs gated) persist across different r0?

---

## Experiment Design

* Fixed thrust (baseline): 1500 N
* Controllers:

  * always_on
  * gated
* Initial radius offsets:

  * 1.002
  * 1.005
  * 1.01
  * 1.02

Additional diagnostic:

* High-thrust test at 100000 N
* Minimal configuration:

  * r0 = 1.01
  * controllers = always_on, gated

---

## Results

### 1. r0 Sweep at 1500 N

* avg_radius_error increases approximately linearly with r0
* always_on and gated curves overlap almost perfectly

Observation:

> No visible controller separation under baseline thrust.

---

### 2. High-Thrust Diagnostic (100000 N)

* always_on and gated begin to diverge in behavior
* Differences observed in:

  * total_reward
  * saturation usage

However:

* avg_radius_error remains at the same order as initial offset
* No significant convergence toward target orbit

---

## Interpretation

### Key Insight 1 — Controller Difference Exists (Conditional)

* At low thrust (1500 N):

  * controllers are effectively indistinguishable
* At high thrust (100000 N):

  * controllers diverge

Conclusion:

> Controller logic is active, but its effect depends on actuation strength.

---

### Key Insight 2 — No Radial Convergence

Despite controller differences:

* radial error does not decrease significantly
* system does not enter a convergence regime

Conclusion:

> Current control strategy is not aligned with reducing radial error.

---

### Key Insight 3 — Regime Identification

System behavior is dominated by initial condition:

> r0 → error propagation (linear)

Rather than:

> controller → error correction

Conclusion:

> The system is still in an "initial-condition-dominated regime", not a true control regime.

---

## W03 Conclusion

* Separation between controllers is not robust at baseline thrust
* High thrust reveals controller differences, but not effective control
* Current controller does not achieve radial error convergence

---

## Next Step — W04

Shift focus from parameter sweep to dynamics diagnosis:

> Analyze time series of r_err(t) and v_r(t)

Goal:

* Determine whether controller:

  * suppresses radial velocity
  * or actively drives convergence

This marks transition from "structure observation" to "mechanism understanding".

---

## Commit Message

W03: r0 sweep shows no separation at 1500N; high-thrust reveals divergence without convergence
