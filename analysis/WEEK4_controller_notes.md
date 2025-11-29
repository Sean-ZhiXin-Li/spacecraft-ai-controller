## Week4 – Final ExpertControllerImproved (v4.2) baseline

I implemented `expert_controller_improved` with three robustness-related
features, and for the v4.2 baseline I only keep the thrust-direction
smoothing active by default (the scheduler and alignment terms are
implemented but disabled via flags).

I compared the original ExpertController (ExpertV3) and
ExpertControllerImproved on four scenarios:

- weak_thrust_far
- oscillation_noise
- misaligned_entry
- default

Metrics for each run:

- steps
- total_reward
- final radius r(T)
- average radius error |r(t) - r0|
- average thrust direction jitter (1 - cos θ)

Summary of results:

- In all four scenarios, both controllers complete the full 2000-step
  episodes without numerical issues.
- ExpertControllerImproved consistently achieves higher total_reward
  (e.g. -9.521e3 vs -9.569e3), which means a small but systematic
  improvement under the current reward function.
- The average radius error of the improved controller is effectively
  zero under the current metric, while ExpertV3 has a small but non-zero
  error (~4.19e5 on an orbit scale of 9.375e12, i.e. ~10^-8 relative
  error). This indicates that the improved controller keeps the radius
  extremely close to the initial orbit.
- The average thrust jitter of ExpertControllerImproved is slightly
  higher (3.04e-7 vs 2.81e-7) but remains on the same 10^-7 scale, so
  there is no sign of serious oscillations.

Conclusion:

- For the current OrbitEnv and reward design, `expert_controller_improved`
  (v4.2 baseline) is a numerically stable and slightly stronger
  controller than the original ExpertController.
- Future work will use more diagnostic metrics (spiral-in time,
  oscillation amplitude, insertion angle error) and refined failure
  scenarios to better expose robustness differences, and possibly
  re-enable the scheduler/alignment features in a more realistic
  setting.
