# PL13 – PPO Orbit Training Debug Notes

## Overview

This session focused on diagnosing instability in PPO-based orbital control training. The agent was expected to learn smooth orbital trajectories, but instead produced discontinuous, piecewise-linear motion.

---

## Key Observations

### 1. Trajectory Shape

* The trajectory appears as **L-shaped / staircase-like paths** instead of smooth curves.
* Indicates the policy outputs nearly constant actions over extended periods.

### 2. Radius Behavior

* Radius vs time shows **step-like jumps** instead of gradual changes.
* Suggests lack of fine-grained control.

### 3. Alignment Metric

* |cos(angle)| quickly stabilizes near zero.
* The agent successfully learns **tangential velocity alignment**.
* However, it fails to control orbital radius.

### 4. Reward Curve

* Reward generally increases over time.
* However, large oscillations and drops indicate **unstable learning**.

---

## Root Causes

### 1. Action Smoothing Constraint

The environment limits how much the action can change per step:

```python
max_action_delta = 0.15
action_delta = np.clip(action - self.prev_action, -max_action_delta, max_action_delta)
action_smooth = np.clip(self.prev_action + action_delta, -1.0, 1.0)
```

**Effect:**

* Forces actions to change slowly
* Leads to long periods of nearly constant thrust direction
* Produces straight-line motion instead of curves

---

### 2. Large Time Step (dt = 2.0)

* Each step causes a large state transition
* Makes control effectively discrete and coarse

---

### 3. Weak Reward Constraints

* Reward encourages alignment but not continuous control
* No penalty for keeping thrust direction constant

---

## Fixes Applied / Suggested

### 1. Remove Action Smoothing (Critical)

Replace with direct action usage:

```python
action_smooth = action
self.prev_action = action.copy()
```

---

### 2. Reduce Time Step

```python
DT = 0.5  # or even 0.2
```

---

### 3. Reduce Thrust Scale

```python
THRUST_SCALE = 5000.0
```

* Prevents overly aggressive control
* Encourages gradual trajectory shaping

---

### 4. Disable Expert Initialization

Training should be purely PPO-based to avoid bias from imperfect datasets:

```bash
python ppo_orbit/ppo.py --epochs 100 --no-expert
```

---

## Interpretation

The current behavior is not random failure but a **specific failure mode**:

> The policy converges to piecewise constant control instead of continuous control.

This is common in RL when:

* Action changes are constrained
* Dynamics are too coarse
* Rewards do not penalize control stagnation

---

## Next Steps

* Verify trajectory becomes smooth after removing action smoothing
* Check radius curve becomes continuous
* Monitor if reward variance decreases

---

## Conclusion

The system is close to working:

* It already learns correct velocity direction
* The remaining issue is **control smoothness and temporal resolution**

Fixing environment constraints should allow PPO to learn realistic orbital motion.
