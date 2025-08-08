# Project Log Day 24

 **Date**: August 8, 2025  
 **Project**: AI Orbital Control – PPO Phase (Expert Init)  
 **Author**: ZhiXin Li

---

## Summary of Today’s Work

Today marked the launch of the **PPO training phase initialized from the ExpertController**. The main training script (`ppo.py`) was configured to:

- Initialize the `ActorCritic` model with weights from the ExpertController using `load_expert_into_actor_critic()`
- Normalize observations via `normalize_state()`
- Use `Adam` optimizer with `lr=3e-5`
- Train over **800 epochs** on the Voyager 1–style long-range orbit

Training ran successfully after resolving an initialization bug (`NameError: 'optimizer' is not defined`), and produced a complete training curve.

---

## Results & Observations

**Training Curve**: `PPO Training Curve (Expert Init)`

- Initial reward (Epoch 1): **-25269.89**
- Plateau: ~**-28000** from Epochs 100–600
- Instability: Sudden drop to **-31000+** near Epoch 700
- Final reward (Epoch 800): **still oscillating near -28500**

### Interpretation:

- PPO **did not outperform ExpertController** (whose reward was ~-27000)
- Training curve suggests **early learning**, then **stagnation**
- A **reward collapse** occurred late in training, indicating instability or bad policy drift

---

## Issues Encountered

1. **Initialization Error**:
   - `optimizer = optim.Adam(actor_critic.parameters(), lr=LR)` failed initially due to missing `actor_critic` definition (should be `model`)
   - Fixed by correcting the name to `model.parameters()`

2. **Unstable Reward**:
   - PPO failed to converge to better performance than expert
   - Likely causes: improper reward shaping, excessive exploration, unstable gradients

---

## Hypotheses Going Forward

- **Expert Init alone is not enough** — PPO may require **additional imitation guidance or stronger shaping signals**
- Model may suffer from:
  - Overfitting to poor trajectories
  - Lack of advantage normalization
  - High variance due to improper `clip_range` or entropy weight

---

## Takeaway

While initializing PPO from an expert policy provides a strong start, it is **not guaranteed to improve performance** unless:
- The reward is properly shaped
- The training remains stable
- The PPO architecture is aligned with the expert policy’s structure and distribution

This experiment confirms the **need for hybrid approaches**, such as warm-start imitation or combined reward-BC loss (e.g., DAPG-style optimization).

---

