# Project Log – Day 11: Imitation Learning Enhancement (V4)

## What I accomplished today

Today, I trained a deeper neural network (V4) for imitation learning using an expanded dataset consisting of 30 expert trajectories with varied initial velocity angles. The key steps included:

- Training `MLPRegressor` with hidden layer sizes `(512, 256, 128, 64)`, `alpha=1e-4`, and `early_stopping=True`.
- Aggregating 30 expert-generated `.npy` datasets from simulations with different launch directions.
- Successfully training the model with ~1.8 million samples; early stopping activated at iteration 20.
- Test MSE reported as **0.179**, indicating accurate local thrust vector prediction.
- Visualizing thrust direction comparison (green: expert, red: predicted).
- Simulating the spacecraft's orbit using the trained V4 model.
- Generating enhanced radial error curves, error histograms, and r(t) plots.
- Comparing the trajectory qualitatively and quantitatively to the expert baseline (V3.1 and expert controller).

## Performance Summary

- **Test MSE**: 1.79e-01
- **Mean radial error**: 5.01e+12 m
- **Standard deviation**: 1.52e+12 m
- **Max deviation**: 7.35e+12 m

## Visual Analysis

- The imitation controller consistently pushed the spacecraft away from the sun but **never stabilized** near the target orbit.
- The `r(t)` curve displayed monotonic outward expansion, confirming lack of corrective behavior.
- The thrust direction quiver plot showed generally good directional alignment with the expert, though slight angular noise and underpowered thrust were observed in central regions.
- Error distribution was heavily skewed toward high deviations.

## Problems Encountered Today

- Although the V4 model successfully mimicked local expert thrust vectors, it failed to achieve **closed-loop orbital stability**.
- The agent **drifted continuously outward**, indicating that the model could not learn corrective thrust behavior from demonstration alone.
- This highlighted a key limitation of behavior cloning: the agent performs well **only within the expert distribution**, and fails to recover once it drifts outside.
- Increasing the dataset to 30 trajectories **did not significantly improve robustness**, and in some cases may have introduced more variance without meaningful diversity.
- I realized that despite accurate thrust imitation, **trajectory-level objectives like orbital convergence and error correction** require either reward-based learning or hybrid imitation+RL strategies.

## Files Generated Today

- `imitation_policy_model_V4.joblib`
- `imitation_traj_V4.npy`
- `enhanced_error_V4.png`
- `error_hist_V4.png`
- `thrust_quiver_V4.png`
- `comparison_radius_V4.png`
- `comparison_error_V4.png`
