# WEEK6 Verification Note (CM Day1)

On Dec 21, I re-ran `python src/quick_compare_v3_v4.py` to confirm the Week6 baseline is stable and reproducible (no code logic changes). For `weak_thrust_far`, both controllers reproduced the same overall behavior and metrics: saturation stayed around ~0.37–0.38 and the summary statistics matched the Week6 baseline scale (total_reward ≈ -2.339e+04, final_r ≈ 9.375e+12, avg_radius_error ≈ 1.875e+12). For `oscillation_noise`, the run completed with a clear warning that OrbitEnv does not support the noise API (thus default behavior), and the baseline metrics remained consistent (saturation_rate ≈ 0.10; higher jitter compared to weak_thrust_far). This matters because it locks a trustworthy ground-truth baseline before introducing multi-orbit and more complex environment variations.

Evidence excerpts:
- [weak_thrust_far] ExpertV3 | steps=2000, total_reward=-2.339e+04, final_r=9.375e+12, avg_radius_error=1.875e+12, avg_jitter=8.145e-07
- [SELF-CHECK] saturation_rate=0.374 | raw_norm_mean=1.892e+00 | clip_norm_mean=7.962e-01
- [weak_thrust_far] ExpertImproved | steps=2000, total_reward=-2.339e+04, final_r=9.375e+12, avg_radius_error=1.875e+12, avg_jitter=9.457e-07
- [SELF-CHECK] saturation_rate=0.378 | raw_norm_mean=1.891e+00 | clip_norm_mean=7.978e-01
- [SELF-CHECK][WARN] 'oscillation_noise' not supported by OrbitEnv (no noise API). Using default.
- [oscillation_noise] ExpertV3 | steps=2000, total_reward=-2.430e+04, final_r=9.375e+12, avg_radius_error=1.875e+12, avg_jitter=3.112e-06
- [SELF-CHECK] saturation_rate=0.100 | raw_norm_mean=5.215e-01 | clip_norm_mean=2.283e-01
- [oscillation_noise] ExpertImproved | steps=2000, total_reward=-2.431e+04, final_r=9.375e+12, avg_radius_error=1.875e+12, avg_jitter=4.506e-06
- [SELF-CHECK] saturation_rate=0.102 | raw_norm_mean=5.215e-01 | clip_norm_mean=2.304e-01
- thrust_vec evidence (printed by self-check): info[thrust_vec] a=[-800. -800.] | 0=[0. 0.] and info[thrust_vec] a=[-3000. -3000.] | 0=[0. 0.]

saturation_rate log file presence: logs/day54/spiral_in/smoke/metrics.json (contains "saturation_rate": 0.23)
