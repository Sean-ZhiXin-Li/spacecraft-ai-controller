# Project Log – Day 12

**Date:** 2025-07-27  
**Project:** PPO Controller for Orbital Trajectory  
**Focus:** Trajectory Visualization & Training Diagnostics

---

## Today's Progress

1. **Completed PPO Model Training and Saving:**
   - Trained the PPO controller over 10 epochs and successfully saved it as `ppo_controller.pth`.
   - The model follows an Actor-Critic architecture, using Generalized Advantage Estimation (GAE) and clipped loss for stable training.

2. **Final Trajectory Visualization Implemented:**
   - Used the trained PPO policy to simulate the full trajectory.
   - The plot clearly shows the spacecraft's path, the central body (Sun), and the target orbit radius.

3. **Reward Curve Visualization:**
   - Successfully plotted the return sum (total reward) for each training epoch.
   - Saved as `ppo_reward_curve.png`. The downward trend in the curve reflects the PPO model’s poor convergence and can be used to evaluate training quality.

---

## Problems Encountered

- **Lack of Linear Algebra & PyTorch Background:**
   - I don't fully understand operations like `logits - actions`, `.sum(dim=1)`, `.squeeze()`, or `.detach()` — they seem mathematical but I don’t yet know the linear algebra behind them.
   - I can follow code structure like `nn.Linear`, `forward()`, and tensor manipulations, but cannot explain *why* they work — I just copy the pattern for now.
   - Especially the computation of `logprob` is unclear: I don’t know what it mathematically represents or why it’s being used that way in PPO.

- **Unclear PPO–Environment Interaction:**
   - I still don’t fully understand how `env.step(action)` fits into PPO training logic.
   - The concepts of state normalization, how reward is computed from orbit distance, and how the PPO agent uses this to learn are still very vague.

---

## Saved Artifacts

- `ppo_controller.pth` — final trained PPO policy.
- `ppo_reward_curve.png` — reward plot across epochs.
- Trajectory visualizations showing simulated orbital path.

