# Project Log – Day 49

**Date:** 2025-09-30  
**Task:** Validate ExpertController baseline on MultiOrbitEnv (circular / elliptic / transfer)  

---

## Goals
- Integrate `ExpertController` (v3.1 behavior) with `MultiOrbitEnv`.
- Run baseline rollouts across three scenarios:  
  - Circular orbit  
  - Elliptic orbit  
  - Transfer orbit  
- Save results into `logs/day49/test_log_day49.csv`.

---

## Progress
1. Fixed import issues by restructuring `sys.path` and ensuring `controller` is correctly loaded.
2. Adapted Expert → Env interface:  
   - Extracted `(pos, vel)` from `obs/info/env`.  
   - Added fallback logic for `target_radius`.  
   - Wrapped raw thrust vectors into environment-compatible `action_space`.
3. Confirmed rollout pipeline works for all three scenarios.  
4. Verified logging: results saved into CSV with columns `[datetime, scene, controller, preset, seed, steps, total_reward, orbit_error_mean, orbit_error_final, success]`.

---

## Results
- **Circular**: Reward ≈ -1541.7, error stable around 0.25.  
- **Elliptic**: Reward ≈ -1541.7, error stable around 0.25.  
- **Transfer**: Reward ≈ -2769.1, error slightly above 0.25.  
- Saved logs: `logs/day49/test_log_day49.csv`.

---

## Notes
- Currently running with `accel_mode=False` (Expert output treated as force).  
- Orbit error remains flat (0.25) → indicates further alignment needed between Expert’s expected dynamics and environment’s interpretation of thrust.  
- Next step: add targeted debug prints to confirm `action_space` shape and scaling behavior.  

---

## Next Steps
- [ ] Print and analyze `action_space` bounds (`low`, `high`, `shape`).  
- [ ] Verify scaling: check if environment normalizes thrust vectors to [-1,1].  
- [ ] Explore radius synchronization: align `expert.target_radius` with environment’s hidden target.  
- [ ] Re-run trials with thrust gain calibration.

