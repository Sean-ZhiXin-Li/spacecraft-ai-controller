# Project Log – Day 26

## Summary
Today, I updated three key components of the PPO-based spacecraft AI control project:

1. **`ppo.py`**  
   - Set `THRUST_SCALE = 3000`.  
   - Added optional toggle switches for **VF_COEF**, **LR_CRITIC**, and **network size** to allow flexible training experiments.

2. **`rewards_utils.py`**  
   - Applied the updated set of coefficients for the reward function, aiming to balance **radial error**, **velocity error**, **fuel penalty**, and **bonus rewards** for achieving target orbits.

3. **`envs/orbit_env.py`**  
   - Implemented **early stopping on success**: the episode now ends if the spacecraft meets the target orbit stability criteria before reaching the maximum step count.  
   - Added a **terminal reward** for successful orbit insertion to further incentivize stable control behavior.

These modifications are expected to improve **training efficiency** and **policy stability** by:
- Shortening unnecessary simulation steps when success is achieved early.
- Providing stronger positive reinforcement at mission completion.
- Allowing faster iteration with flexible hyperparameter adjustments.

---

## Problems Encountered
- **Trajectory Visualization Delay**: The orbit trajectory plot after training was noticeably slow to generate due to the high number of simulation steps; optimization or step limit adjustments may be necessary in future runs.
- **Reward Scaling Sensitivity**: Initial training runs showed unstable reward curves when the bonus and penalty magnitudes were not balanced, requiring multiple tuning attempts.
- **Parameter Toggle Logic**: Added switches for VF_COEF, LR_CRITIC, and network size, but ensuring that changes do not conflict with other PPO hyperparameters required extra code checks.
- **Early Stop Criteria Tuning**: Choosing the right tolerance for early success detection was tricky—too strict and episodes rarely stop early; too loose and the spacecraft might terminate before actually stabilizing in orbit.

---

## Files Modified
- `ppo.py`
- `rewards_utils.py`
- `envs/orbit_env.py`
