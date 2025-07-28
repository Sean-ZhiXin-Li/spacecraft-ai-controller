# Project Log Day 13 – PPO Training Plateau Analysis

## Date: 2025-07-28

## What I Did Today

Today I continued training the PPO agent using a Gaussian policy over 600 epochs. I revised the environment configuration by:

- Setting `max_steps=5000` in `OrbitEnv`.
- Increasing `thrust_scale` to 5.0 to overcome gravitational pull.
- Enlarging the neural network hidden layers to `[256, 256]` for both actor and critic.
- Switching to a learnable `log_std` parameter in the actor output.
- Normalizing the input state vector to stabilize training.

The PPO agent was trained with:
- `GAMMA = 0.99`, `LAMBDA = 0.95`
- `EPOCHS = 600`, `TRAIN_ITERS = 20`, `LR = 1e-4`

## What I Observed

- From **epoch 7 onwards**, the PPO agent's total return **collapsed to -83000** and remained almost constant across ~300 epochs.
- Despite a higher thrust scale and deeper network, **the learning stagnated**, indicating no effective gradient signal.
- Final trajectory visualization shows chaotic or minimal adjustment, with **no convergence toward target radius**.
- I plotted the reward curve: it shows no upward trend, consistent with return stagnation.
- I suspected that reward shaping was insufficient and environment signal might be weak.
- I also observed that the blue trajectory curve (AI) barely adjusted, suggesting **poor thrust direction control** or **gradient vanishing**.

## Problems Encountered

- PPO training **failed to improve after early epochs**, indicating:
  - Weak or misleading reward shaping.
  - Initial state too close to a failure-prone orbit.
  - Thrust too weak or badly directed.
- Thrust scale of 5.0 is likely still insufficient given the orbital scale (7.5e12 radius).
- The environment may not provide **enough positive rewards** to guide the PPO learning.
- Most runs quickly terminate or produce flat performance, suggesting PPO is stuck in a **poor local policy basin**.

## Files Updated Today

- `ppo_train_gaussian.py` (main PPO training logic, revised hyperparameters)
- `OrbitEnv` (updated with max_steps and thrust_scale input)
- PPO reward curve plot: `ppo_reward_curve_gaussian.png`
- Trajectory visualization: `Final PPO Gaussian Trajectory`

