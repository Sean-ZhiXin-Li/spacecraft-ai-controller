# Project Log – Day 23

 **Date:** August 7, 2025  
 **Project:** PPO-based AI Controller for Orbital Insertion  
 **Focus:** Reward function redesign, full PPO retraining, reward logging and visualization

---

## Today’s Progress

1. **Implemented Fully Optimized PPO Reward Function**
   - Added orbital shaping terms (radius error, speed error, angular misalignment).
   - Added continuous Gaussian-like bonus near optimal orbit.
   - Penalized fuel usage with a soft linear term.
   - All components are continuous, bounded, and smooth—ideal for PPO stability.

2. **Updated PPO Hyperparameters**
   - `GAMMA = 0.99`, `LAMBDA = 0.95`, `LR = 1e-5`
   - `THRUST_SCALE` adjusted for improved orbital correction behavior.

3. **Full PPO Re-training**
   - Ran 800 epochs using the new reward design and PPO config.
   - Trained model showed more stable reward convergence compared to earlier unstable PPO runs.

4. **Reward Component Logger**
   - Logged individual components: `shaping`, `bonus`, `penalty`, `r_error`, `v_error`.
   - All values written to `reward_breakdown.csv` for each episode.

5. **Training Curve Visualization**
   - Generated total reward plot (`training_curve.png`).
   - Observed oscillations followed by partial stabilization after 300 epochs.
   - Overall reward hovered between `-33000` and `-34500`, no clear upward trend yet.

---

## Observations & Insights

- Early-stage instability (~Epoch 0–100) is expected as PPO explores.
- Around Epoch 300, reward stabilizes into a low-variance band, suggesting learned behaviors.
- However, the lack of strong reward growth implies that the policy is either:
  - Not reaching ideal orbits consistently,
  - Or staying in a suboptimal orbit with low variance.

---

## Issues Encountered

- Large variance in early PPO returns.
- No significant improvement in average reward beyond Epoch 400.
- Trained model might still be:
  - Struggling with misalignment,
  - Failing to fully close orbit (low bonus),
  - Or inefficient in fuel use.

---

## Files Added or Updated

- `ppo.py`: PPO logic with reward breakdown logging.
- `reward.py`: Full English-commented reward function.
- `reward_breakdown.csv`: Reward components per episode.
- `training_curve.png`: Total reward plot over 800 epochs.

---
 
*End of Day 23 Log*
