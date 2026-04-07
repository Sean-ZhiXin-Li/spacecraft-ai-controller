# PL Day 12 – Orbit Control: From Approach to Stabilization Failure

## 1. Overview

Today’s work focused on improving the reinforcement learning reward function for orbital control using PPO. The goal was to transition the agent from simple trajectory shaping to stable orbit maintenance.

Significant progress was made in shaping the agent’s behavior, especially in terms of tangential alignment and smoother trajectory formation. However, stable orbit closure was not achieved due to reward imbalance and control instability.

---

## 2. Key Observations

### 2.1 Trajectory Behavior

* The trajectory shows a clear **arc-like motion**, resembling the first half of an orbital path.
* The agent successfully:

  * Moves outward toward the target radius
  * Begins turning tangentially
* However:

  * It fails to maintain the orbit
  * The trajectory eventually diverges outward (escape behavior)

---

### 2.2 Alignment (cos(angle))

* Alignment decreases smoothly from ~1.0 toward ~0.3
* Compared to previous runs:

  * No oscillation back to radial alignment
  * Indicates more stable directional control
* However:

  * The agent does not sustain a near-zero alignment (ideal tangential state)

---

### 2.3 Radius Evolution

* Radius increases **monotonically**
* No stabilization around the target radius
* Indicates:

  * Lack of radial velocity suppression
  * No effective orbital “locking”

---

### 2.4 Velocity Behavior

* Velocity increases continuously over time
* No braking or energy regulation observed
* Suggests:

  * The agent is not learning energy-constrained motion
  * Control policy degenerates into acceleration

---

## 3. Reward Design Issues

### 3.1 Previous Problem (Day 11)

* Insufficient penalty on radial velocity (`v_r`)
* Result:

  * Agent enters orbit region but cannot maintain it

---

### 3.2 Current Problem (Day 12)

* Overly aggressive penalty on radial velocity
* Example changes:

  * Large coefficients on `abs(vr_norm)`
  * Hard penalties near target radius

### Consequence:

* **Reward collapse**
* Agent stops exploring meaningful control strategies
* Falls back to trivial or unstable behaviors (e.g., drifting outward)

---

## 4. Key Insight

> Stable orbit learning requires **balanced shaping**, not maximum constraint.

There is a critical trade-off:

* Too weak → cannot maintain orbit
* Too strong → agent stops learning

The current model has crossed into the second regime.

---

## 5. Lessons Learned

1. **Reward smoothness is critical**

   * Sharp penalties destroy PPO gradient signals

2. **Continuous rewards > event-based rewards**

   * Orbit is a sustained behavior, not a single-state objective

3. **Radial velocity is the main instability source**

   * But must be controlled gradually

4. **Positive reinforcement is more effective than hard penalties**

   * Encourage correct behavior instead of punishing all deviations

---

## 6. Next Steps (Day 13 Plan)

* Rebalance reward function:

  * Reduce radial velocity penalty magnitude
  * Remove hard threshold penalties
* Introduce:

  * Smooth near-orbit shaping
  * Stronger sustained tangential reward
* Goal:

  * Achieve **radius stabilization plateau**
  * Prevent monotonic escape behavior

---

## 7. Conclusion

The agent has successfully learned:

* Directional control (partial tangential alignment)
* Smooth trajectory shaping

However, it has not yet learned:

* Orbital stabilization
* Energy regulation

This marks a transition from **“learning to reach orbit”** to **“learning to stay in orbit”**, which is the final and most challenging stage of this project.

---
