# Project Log – Day 14

## Progress Today

- Wrote and executed `evaluate_ppo_orbit.py` to test the trained PPO model.
- PPO model (`ppo_best_model.pth`) was successfully loaded using `weights_only=True`.
- Environment initialized with:
  - Initial position: `[target_radius, 0.0]`
  - Initial velocity: circular orbit speed at 30° from x-axis
- Full PPO inference loop was implemented, including:
  - Feeding normalized states into the model
  - Recording `[x, y, vx, vy, Tx, Ty]` for trajectory reconstruction
  - Generating plots using `plot_trajectory()`, `plot_radius_vs_time()`, and `plot_thrust_quiver()`

## Issues Encountered

- **The generated trajectory was essentially empty.** The plotted orbit was either not visible or a single dot near the origin.
- Possible causes:
  - **PPO model failed to learn anything meaningful** during training (reward stuck around `-83000.93`)
  - **Evaluation inputs were not normalized**, so model received values it had never seen during training
  - **Action outputs were nearly zero** for most steps, resulting in no thrust or change in velocity
- Also encountered the following coding/debug issues:
  - `ValueError: expected sequence of length 4 at dim 1` due to empty `state` tensor
  - `AttributeError: 'Normal' object has no attribute 'squeeze'` when using incorrect return type from the policy
  - Fixed by checking tuple unpacking from the PPO model and correctly sampling actions

## Files Involved

- `ppo_orbit/evaluate_ppo_orbit.py` – evaluation pipeline
- `ppo_best_model.pth` – trained model checkpoint
- `ppo.py` / `model.py` – PPO architecture
- `OrbitEnv` – with initial launch setup at 30° circular velocity
- `plot_trajectory`, `plot_thrust_quiver`, etc. for visualization

## Output Observation

> The plot rendered a blank or nearly invisible orbit.  
> No significant thrust vectors or radial variations were observed.  
> PPO actions might be too small to create orbital motion.

## Reflection

Although the technical stack is working end-to-end, the PPO controller is clearly not producing useful control behavior. This will require:
- Verifying that `normalize_state()` is applied consistently
- Re-evaluating the reward shaping to ensure learning signal exists
- Possibly retraining from scratch with debug visualizations every 100 episodes

## Output Not Achieved

-  No meaningful orbital trajectory produced.
-  No visible thrust vector or radial control.
