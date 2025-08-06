# Project Log – Day 22

## Date
August 6, 2025

## Title
V6.1 Mimic Controller Trained & Closed-loop Tested

## Summary

Today marks a key milestone in the development of the spacecraft thrust controller.

I successfully trained and deployed the **V6.1 Imitation Controller**, which is the first model trained using a **PyTorch-based MLP network** on expert trajectory data.

Key steps completed:

-  Cleaned dataset with duplicate filtering
-  Applied data downsampling to avoid overload
-  Implemented and trained `MLPRegressor` (PyTorch)
-  Saved model as `mimic_model_V6_1.pth`
-  Built a custom loading class `ImitationController_V6_1`
-  Integrated it into `simulate_orbit()` for full closed-loop trajectory
-  Generated full visualizations including:
  - `plot_trajectory()`
  - `plot_radius_vs_time()`
  - `plot_radius_error_with_analysis()`
  - `plot_error_histogram()`
-  Final output saved as `imitation_traj_V6.1_long.npy`

##  Results

-  **Trajectory**: The spacecraft launched out in a straight-line trajectory without successful orbit capture.
-  **Radius vs Time**: Sharp divergence in radial distance, indicating full orbit failure.
-  **Mean Radial Error**: ~6.98e+16 m
-  **Standard Deviation**: ~6.31e+16 m

Clearly, the current model fails to provide sufficient control for orbit insertion.

## Issues Encountered

1. The controller outputs poor thrust values, resulting in total mission failure.
2. The trained model likely suffers from:
   - Poor generalization
   - Weak network capacity
   - Data imbalance
3. The control was tested for **8,000,000 steps**, which exposed long-horizon divergence.

## Power Outage Notice

Today from **1 PM to after 7 PM**, a major power outage delayed work for several hours.
Despite this, I managed to complete the full training + testing pipeline.

## Takeaways

> This is not a failure, but a crucial checkpoint.  
> Imitation Learning is only as strong as the data and model it is built on.  
> A poor mimic is better than a random guess – and it gives PPO a warm start.

## What's Next?

While I won't move to the next step today, I’ve already confirmed the plan:

**Stage: V6.1-Hybrid**

- Load the V6.1 model into a PPO agent as the initial policy
- Fine-tune via reinforcement learning to improve long-term control

This hybrid method is expected to reduce learning time and boost orbit success rates.

---

 *Even when the orbit breaks, I won’t.*  
This is what makes it mine: the pain, the doubt, the will to continue.

