# Day 8 — Reward Shaping Comparison for Orbital Control (PPO)

## 1. Objective

Evaluate how different reward shaping strategies affect policy behavior in a 2D orbital control task trained with PPO.

Key question:

> Can reward design alone induce stable orbital control?

---

## 2. Experimental Setup

* Environment: 2D Newtonian gravity + thrust
* Agent: PPO (continuous actions)
* Training: 200 epochs per setup
* Same initialization & hyperparameters across runs

### Reward configurations

1. **Base**

   * Shaping (radius, velocity, alignment)
   * Fuel penalty + bonus

2. **Radius**

   * Base + radius error penalty

3. **Progress**

   * Base + radial progress term

4. **Combined**

   * Radius + Progress

---

## 3. Observations

### Base

* Radius: monotonic increase (divergence)
* Velocity: grows continuously
* Behavior: tangential acceleration → escape

**Interpretation:**
No explicit convergence signal → policy maximizes motion without control.

---

### Progress

* Radius: steady outward drift
* Velocity: relatively stable

**Interpretation:**
Progress reward encourages forward motion along trajectory,
not correction toward target radius.

→ Tangential bias dominates

---

### Radius

* Radius: shows corrective movement
* But: unstable / oscillatory

**Interpretation:**
Agent begins radial control, but lacks stability mechanism.

---

### Combined

* Radius: direction corrected (no longer purely diverging)
* Still: no convergence / no closed orbit

**Interpretation:**
Competing objectives:

* progress → move forward
* radius → correct position

→ partial improvement but no stable solution

---

## 4. Key Findings

### 1. Reward influences behavior direction

* Radius term → radial correction
* Progress term → tangential motion

### 2. PPO exploits reward shortcuts

* Learns simple strategies instead of orbital dynamics

### 3. Trade-off emerges

* Stability vs convergence cannot be solved by simple reward mixing

---

## 5. Limitation Identified

> Reward shaping alone is insufficient for orbital control.

Main issue:

* No constraint on system energy / velocity
* Agent can maintain incorrect energy while maximizing reward

---

## 6. Next Step (Day 9)

Introduce **velocity / energy constraint**:

* Penalize deviation from circular velocity
* Prevent uncontrolled acceleration

Goal:

> Reduce reward-induced drift and improve stability

---

## 7. Conclusion

Reward design changes policy behavior,
but does not guarantee physically valid control.

Stable orbit likely requires:

* Additional constraints
* Or structured control policies
