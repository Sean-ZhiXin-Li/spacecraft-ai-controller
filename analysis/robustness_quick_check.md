# Robustness Quick Check

## Setup

- baseline: `dt=100.0`, `r0_over_target=1.00005`, `max_steps=100000`, `thrust_scale=10000`
- trials per perturbation: `5`
- no PPO retraining, no controller redesign, no physics changes

## Results

- `velocity_noise_plus_minus_1pct`: success_rate `0.60`, crossing_rate `0.60`, mean_final_radius_error `2.401e+08`, mean_tail_mean_abs_vr `1.073e+02`; mixed behavior with partial success preservation.
- `action_noise_std_0.001`: success_rate `0.80`, crossing_rate `0.80`, mean_final_radius_error `5.874e+08`, mean_tail_mean_abs_vr `2.043e+02`; mixed behavior with partial success preservation.
- `small_position_perturbation`: success_rate `1.00`, crossing_rate `0.60`, mean_final_radius_error `1.734e+05`, mean_tail_mean_abs_vr `5.612e+01`; all trials preserved strict success.

## Interpretation

- This is a lightweight smoke robustness check, not a statistical robustness proof.
- The controller is most credible as a narrow deterministic insertion solution; perturbation sensitivity remains a core limitation.