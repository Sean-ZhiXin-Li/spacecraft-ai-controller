# Project Log — Day 29

## Summary
Today, I continued training the PPO agent for the orbital control task.  
The code is now running with adaptive learning rate, KL-based update iteration adjustment, and an expert warm start (either from offline data or an online physics-based expert).  
Logging and plotting are fully operational, with regular checkpoints saved every 100 epochs.

## Training Status
- **Environment**: Single-orbit control (current version)
- **Agent**: PPO with entropy schedule, value loss schedule, and KL-driven training iteration adaptation
- **Initialization**: Dataset-based warm start skipped (action magnitudes too small), switched to online physics expert
- **Performance trend**: Gradual improvement in rewards; value function explained variance shows occasional spikes
- **Checkpoint**: Saved every 100 epochs

## Key Observations
- The adaptive learning rate mechanism increases update aggressiveness when KL is low, and backs off when KL is high.
- Value function warmup via TD(λ) for one epoch appears to stabilize early training.
- Exploration is well-maintained with the entropy coefficient schedule and minimum standard deviation clamping.

## Tomorrow’s Plan
Starting tomorrow, I will:
- Introduce **multi-orbit** scenarios into the environment.
- Gradually increase complexity (e.g., variable orbital radii, inclined planes, or gravitational perturbations).
- Adjust reward shaping to accommodate multi-orbit transitions.

## Longer-Term Plan
After the school semester starts, I plan to:
- Perform systematic hyperparameter tuning (learning rates, clipping parameters, entropy schedules).
- Experiment with larger network architectures for the actor-critic model.
- Possibly integrate curriculum learning for progressively harder orbital maneuvers.

## Notes
- Current logs include reward, actor loss, critic loss, KL divergence, and explained variance.
- The PPO code is fully annotated in English for clarity and maintainability.
