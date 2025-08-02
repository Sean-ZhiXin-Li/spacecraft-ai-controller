
# Project Log – Day 18: Long-Horizon Orbit Simulation with V5 Imitation Controller

## Summary

Today I conducted a full-length closed-loop simulation using the **V5 imitation controller**, trained previously on expert datasets with normalized input states. The goal was to evaluate how well the controller can handle a long-duration orbital maneuver starting from a low orbit and spiral outward toward a high circular target orbit, similar to a Voyager-like mission.

---

## Simulation Settings

| Parameter | Value |
|----------|--------|
| Central Mass (M) | 1.989e30 kg (Sun) |
| Target Radius     | 7.5e12 m |
| Spacecraft Mass   | 721.9 kg (Voyager 1 approx.) |
| Initial Position  | 1/3 × target_radius (≈2.5e12 m) |
| Initial Velocity  | 1.2 × circular speed at r₀ |
| Time Step (dt)    | 2000 seconds |
| Steps             | 10,000,000 |

---

## Controller

- **Type**: `MLPRegressor` with 3 layers (128, 64, 32), `tanh` activations
- **Input**: `[x, y, vx, vy]` (scaled using `StandardScaler`)
- **Output**: `[Tx, Ty]`, clipped to [-1, 1]
- **Trained on**: Expert datasets via supervised learning

---

## Results

### 1. r(t) – Radius over Time

![r(t) plot](plots/radius_vs_time_v5_long.png)

- The spacecraft initially showed a slow outward spiral.
- Around 1.2e10 seconds, it entered **uncontrolled acceleration** and continued drifting outward.
- Final radius reached **~1.75e15 m**, indicating **full orbital escape**.

---

### 2. Orbit Trajectory

![trajectory](plots/trajectory_v5_long.png)

- The spacecraft began with a brief arc around the Sun.
- It never stabilized into a circular orbit and instead followed a one-sided **escape path**.

---

### 3. Error Metrics

```
[V5 Long Run]
Mean radial error: 5.29e+14 m
Std deviation:     5.52e+14 m
```

- The error magnitude confirms uncontrolled spiral behavior.
- No convergence near the target radius.

---

## Visual Diagnostics

- `plots/error_v5_long.png`: Radial error over time
- `plots/hist_v5_long.png`: Histogram of radial error distribution

---

## Issues & Observations

- The model failed to recognize when it had reached the desired orbit.
- There is **no "capture behavior"** (i.e., reducing thrust when close to goal).
- Mimicry alone is insufficient: the controller **learned thrust mappings**, but not mission context.
- There's no cost-to-go or goal-aware shaping in pure imitation.

---

## Lessons Learned

- **Imitation-only controllers need additional signal** (error, cos(θ), ∥v∥ mismatch, etc.).
- Stable orbit maintenance requires either:
  - richer feature space in IL,
  - or post-finetuning via reinforcement learning (RL).
- Expert strategies must include “orbit lock-on” behavior, otherwise model over-thrusts and escapes.

---

> “When the thrust keeps firing, the orbit keeps drifting. Stability is not about motion, it’s about knowing when to stop.”
