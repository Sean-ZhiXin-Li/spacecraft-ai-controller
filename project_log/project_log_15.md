
# Project Log – Day 15

## Context

Today marks a strategic pivot in the spacecraft control project.

After several days struggling with PPO performance and model instability, I made the decision to temporarily pause the reinforcement learning track and return to a more interpretable, modular approach—focusing again on the **ExpertController** and **ImitationController** pipelines.

This shift was triggered by persistent issues:
- PPO-generated trajectories remained empty or unstable.
- Reward shaping yielded marginal improvements.
- I began to feel disconnected from the logic behind the AI controller, as much of the PPO pipeline was generated externally and required constant debugging with little clarity.

By contrast, the expert rule-based controller—although manually engineered—offers high transparency and intuitive diagnostics. Thus, I decided to return to this baseline to refine and diagnose its behavior in a high-fidelity Voyager-like orbit.

---

## What I Worked On

### 1. **Ran ExpertController in Voyager-scale orbit**
- Target radius was set to `7.5e12` meters.
- Initial position and velocity were carefully set to mimic an inclined injection.
- Orbital trajectory and `r(t)` plot were generated using the controller logic and debug tools.

### 2. **Observed Failure to Maintain Target Orbit**
- The spacecraft initially accelerated outward, overshooting the desired orbit significantly.
- The radial plot `r(t)` showed a parabolic shape: rapid rise, peak, and eventual drop.
- The trajectory visual confirmed an incomplete orbit: the ship flies out, turns back, but fails to settle into a circular orbit.

### 3. **Analyzed ExpertController Logic**
I inspected the following control logic:
```python
thrust_r = -self.radial_gain * np.tanh(radial_error / (0.1 * self.target_radius))
thrust_t = self.tangential_gain * np.tanh(delta_v / v_circular)
thrust = thrust_r * radial_dir + thrust_t * tangential_dir
```

Then applied optional modules:
- **Error feedback scaling:** multiplicative term `(0.5 + error_ratio)`
- **Slowdown near target radius**
- **Turn angle penalty** based on `cos(θ)` between thrust and velocity

However, these modules proved too simplistic or even counterproductive:
- Error ratio scaling caused unstable gain amplification.
- Penalty based on `cos(θ)` lacked smooth transition or saturation control.
- No controller logic handled tangential alignment or orbital injection quality.

---

## Problems Encountered

1. **Orbit did not close:** The spacecraft failed to settle into the desired orbit. Despite turning back, it could not sustain a stable trajectory near the target radius.
2. **`r(t)` peak was far above target:** This indicated excessive initial thrust or poor angular control.
3. **Error feedback was too aggressive:** The `(0.5 + error_ratio)` multiplier overly amplified thrust when error was large.
4. **Angle penalty too weak:** Thrust was applied even when velocity direction was significantly misaligned with tangential orbit.
5. **No orbital injection enforcement:** The controller did not check if velocity was tangential before boosting, leading to inefficient corrections.
6. **Quiver plot missing:** Without thrust vector visualization, debugging exact cause of trajectory deviation was hard.

---

## Reflection

This return to the expert pipeline revealed a deeper understanding of why closed-loop orbit control is non-trivial. Even with a well-engineered reward function, PPO cannot learn what the expert cannot do. If the baseline controller fails to lock on to an orbit, then the imitation and reinforcement learning models trained on it will also struggle.

This day re-emphasized the value of interpretability, trajectory diagnostics, and modular controller design. Instead of purely chasing end-to-end AI solutions, it is essential to master the physical intuition and handcrafted baselines first.

Tomorrow’s direction will likely involve debugging thrust direction logic, adding angular injection checks, and visualizing vector fields to understand thrust application timing.

---
