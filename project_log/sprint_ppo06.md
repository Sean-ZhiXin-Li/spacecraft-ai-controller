# Day 6

## Title

**Thrust Regime Validation: Does PPO Structure Generalize?**

---

## Objective

Evaluate whether the structural behavior learned by PPO under low thrust conditions persists under a high thrust regime.

Specifically:

> Does PPO maintain an inward-thrust (radial) structure, or does its control policy change under high thrust?

---

## Experimental Setup

### Fixed Parameters

* Initial radius ratio: `r0_over_target = 1.25`
* Environment, reward, and simulation settings: unchanged
* Controllers:

  * `always_on`
  * `gated`
  * `ppo`

### Variable

* Thrust:

  * Low thrust (previous): `1500 N`
  * High thrust (current): `100000 N`

---

## Data Integrity

To ensure valid comparison:

* Only real environment rollouts are used
* No scaling bugs (action → thrust applied exactly once)
* Analysis restricted to the current experiment runs:

  * `run_835152`
  * `run_854705`
  * `run_964254`

---

## Metrics

### Performance Metrics

* `total_reward`
* `avg_radius_error`
* `saturation_rate_mean`

### Structural Metrics (from trajectory)

* `cos_tr = cos(thrust, radial)`
* `cos_tt = cos(thrust, tangential)`

We compute:

* Mean and standard deviation of `cos_tr`
* Mean and standard deviation of `cos_tt`

---

## Results

### Summary Table

| controller | mean_cos_tr | mean_cos_tt | reward   | saturation |
| ---------- | ----------- | ----------- | -------- | ---------- |
| always_on  | -0.9991     | +0.0036     | -1.149e5 | 0.0010     |
| gated      | -0.9987     | +0.0040     | -8.680e4 | 0.00025    |
| PPO        | -0.7620     | +0.6442     | -8.239e4 | 0.00000    |

---

## Observations

### 1. Always_on and Gated Remain Radial

* `cos_tr ≈ -1.0`
* `cos_tt ≈ 0`

These controllers apply almost purely inward (radial) thrust.

---

### 2. PPO Develops a Mixed Direction Strategy

* `cos_tr ≈ -0.76`
* `cos_tt ≈ +0.64`

This indicates:

* Reduced radial alignment
* Strong tangential component

PPO is no longer purely inward.

---

### 3. PPO Achieves Best Reward with Zero Saturation

* Highest reward (least negative)
* `saturation_rate_mean = 0`

This suggests:

* PPO avoids aggressive thrust usage
* Control is achieved through direction, not magnitude

---

## Interpretation

Under the high-thrust regime (100000 N), PPO does not preserve the inward-thrust structure observed under low thrust.

Instead, PPO learns a **distinct control strategy** characterized by:

* Mixed radial and tangential thrust
* Reduced reliance on pure inward force
* Zero saturation behavior

In contrast, both always_on and gated controllers remain structurally unchanged and purely radial.

---

## Key Conclusion

> PPO exhibits a regime-dependent control structure.

* At low thrust: PPO behaves similarly to inward (radial) control
* At high thrust: PPO shifts to a mixed-direction strategy

Therefore:

> PPO does not converge to a fixed heuristic policy, but adapts its control structure based on the physical regime.

---

## Implication

This suggests that PPO is not simply imitating predefined controllers.

Instead, it:

* Internalizes system dynamics
* Adapts directionality to optimize performance
* Trades radial alignment for smoother and more stable trajectories under high thrust

---

## Next Step (Day 7)

Investigate the mechanism behind PPO’s tangential behavior:

* Is PPO implicitly controlling angular momentum?
* Does tangential thrust reduce radial oscillation?
* Is this behavior driven by saturation penalty avoidance?

---

## One-Line Summary

> PPO transitions from inward thrust to a mixed radial–tangential strategy under high thrust, indicating a regime-dependent learned control structure.
