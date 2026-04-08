# PL14 Progress Log — Orbit PPO (Smooth Reward)

## 1. Goal

Train a PPO agent to achieve stable orbit (not escape or crash), and produce clean trajectory plots.

---

## 2. Current Setup

### Environment

* Custom `orbit_env`
* Continuous thrust control
* Uses `compute_reward` from `rewards_utils.py`

### Training

* PPO (clip objective)
* GAE enabled
* KL adaptive
* Entropy regularization

### Initialization

* ❌ Removed offline expert dataset
* ✅ Using online expert warm start (optional)

---

## 3. Reward Mode

Current:

```
REWARD_MODE = orbit_smooth_v2
```

Key components:

* Radius error
* Velocity magnitude
* Alignment (cos angle)
* Progress term (currently problematic)

---

## 4. Observed Behavior

### Trajectory

* Nearly straight line
* No curvature → not orbiting

### Speed

* Monotonically increasing
* Indicates continuous thrusting

### Alignment

* cos(angle) → 0.8+
* Moving radially instead of tangentially

### Radius

* Decreasing steadily
* Agent drifting toward center

### Reward Curve

* Starts high (~20000)
* Collapses to negative (~ -13000)

---

## 5. Diagnosis

Main issue: **Reward misguidance**

Agent learns:

* Increasing speed gives reward
* Moving toward target gives reward

But fails to learn:

* Tangential motion
* Stable orbital balance

---

## 6. Fix Strategy

### (1) Penalize radial velocity

```python
radial_penalty = -2.0 * abs(v_r / v_target)
```

### (2) Encourage tangential alignment

```python
tangential_reward = 2.0 * (1 - abs(cos_angle))
```

### (3) Reduce or disable progress reward

```bash
W_PROGRESS=0
```

### (4) Penalize speed deviation

```python
speed_penalty = -1.5 * abs(v - v_target) / v_target
```

---

## 7. Recommended Run Config

```bash
export REWARD_MODE=orbit_smooth_v2
export W_RADIUS=0.5
export W_PROGRESS=0.0
export W_SPEED=0.8

python ppo.py
```

---

## 8. Expected Improvements

After fixes:

* Trajectory becomes curved
* Speed stabilizes
* cos(angle) → 0
* Reward curve stabilizes (no collapse)

---

## 9. Next Steps

* Tune reward weights
* Save best PPO checkpoint
* Generate final plots
* Compare before/after behavior

---

## 10. Summary

Current model is not failing due to PPO.

It is failing because:

> The reward function is encouraging escape instead of orbit.

Fixing reward shaping is the key to achieving stable orbit behavior.
****