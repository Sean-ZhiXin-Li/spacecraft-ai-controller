## Day 7 — Mechanism Analysis (Tangential Control vs Convergence)

### Objective

The goal of Day 7 was to move beyond observing PPO behavior and instead explain **why PPO behaves differently** from classical controllers (always_on, gated) under high-thrust conditions.

---

### Setup

All experiments were conducted within the same W03 pipeline:

* Same environment
* Same thrust setting (high-thrust regime)
* Same initial condition (r0 = 1.01)
* Controllers compared:

  * always_on
  * gated
  * PPO

Metrics extracted from real rollouts:

* Radial velocity (v_r)
* Thrust direction (cos_tr, cos_tt)
* Radius (r)
* Target radius (target_r)

---

### Key Observations

#### 1. Control Structure Difference

* always_on / gated:

  * cos_tr ≈ -1
  * cos_tt ≈ 0
    → purely radial control

* PPO:

  * mean cos_tr ≈ -0.76
  * mean cos_tt ≈ 0.64
    → strong tangential component

Conclusion:

> PPO introduces a new control direction (tangential thrust) that is not used by radial controllers.

---

#### 2. Radial Velocity Behavior

* always_on: fastest divergence (most negative v_r)
* gated: moderate divergence
* PPO: slowest divergence

Conclusion:

> PPO significantly reduces radial velocity collapse and stabilizes the system dynamics.

---

#### 3. Radius Convergence (Critical Finding)

Using radius error and relative error plots:

* always_on: fastest reduction in radius error
* gated: moderate reduction
* PPO: slowest reduction

Conclusion:

> PPO improves stability but reduces convergence speed toward the target radius.

---

### Core Insight (Day 7 Result)

There exists a clear trade-off:

> **Stability vs Convergence**

* Radial controllers:

  * aggressive convergence
  * poor stability

* PPO:

  * strong stability
  * weak convergence

---

### Mechanism Explanation

PPO learns to redistribute thrust direction:

* reduces radial thrust component
* introduces tangential thrust

This leads to:

1. Lower radial acceleration
2. Reduced velocity divergence
3. More stable trajectories

However:

* reduced radial push slows down approach to target orbit

---

### Sprint Milestone Conclusion

> PPO discovers a stabilizing control mechanism (tangential thrust) that mitigates radial collapse, but this comes at the cost of slower radius convergence.

More formally:

> Under high thrust, radial-only control is unstable, while tangential control introduces stability but weakens convergence toward the target orbit.

---

### Implication

This suggests that:

* The problem is not model capacity
* The issue lies in the objective (reward design / optimization target)

PPO is currently optimizing for stability rather than orbital insertion.

---

### Next Step (Day 8)

Focus:

> Can PPO achieve both stability and convergence?

Planned approach:

* Modify reward to include:

  * stronger radius error penalty
  * progress-based reward (reduce error step-by-step)

Goal:

> Resolve the stability–convergence trade-off.

---

### Summary

Day 7 achieved a transition from:

* observing behavior → explaining mechanism

Key takeaway:

> PPO is not random; it is solving a physical constraint (stability), but not yet the full task (orbital insertion).
