# Controlled Restart Day — 2026-02-01

## 1) What I did (objective)
- Re-attached the correct conda env (spacecraft) and ran the true evaluation entry: src/quick_compare_v3_v4.py.
- Fixed scenario=weak_thrust_far and ran thrust_newton ∈ {200, 800, 2000} for both ExpertV3 and ExpertImproved.

## 2) What I observed (trend)
- Despite thrust_newton increasing, avg_radius_error and total_reward stayed essentially unchanged.
- Effective thrust acceleration (a_eff) remained ~9.36e-03 m/s^2 across thrust settings; clip_norm_mean decreased roughly ~1/thrust.

## 3) What question got answered
- The system is “alive” and responds to control input, but thrust sensitivity is currently canceled by the action scaling/clipping pipeline (thrust * clip_norm ≈ constant).

## 4) What I postponed
- No changes to OrbitEnv physics, no reward refactor, and no new controllers; only validated causal sensitivity and located the scaling bottleneck.
