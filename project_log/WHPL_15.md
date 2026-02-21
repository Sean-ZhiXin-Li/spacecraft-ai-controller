# WHPL_15 — Stability Domain Characterization

Date: 2026-02-21
Branch: main
Freeze Rule: No controller modification, no reward change, no parameter tuning

---

# Objective

Day15 transitions the project from experimental logging to structural characterization.

Goal:

1. Convert reward curves into near-optimal stability domains.
2. Compare gated vs always_on across thrust scaling.
3. Identify non-monotonic thrust behavior.

This day focuses strictly on understanding the system, not modifying it.

---

# Dataset

Input file:

analysis/results/whpl14_variant_x_thrust_x_r0.csv

Dimensions:

* thrust ∈ {800N, 2000N}
* controller_variant ∈ {always_on, gated}
* r0_over_target ∈ {1.005, 1.05, 1.20, 1.25}

Key fields used:

* total_reward
* saturation_rate_mean
* avg_radius_error
* final_radius_error

---

# Methodology

## Near-Optimal Stability Definition

For each thrust level:

max_reward = max(total_reward | thrust fixed)

Define near-optimal band:

stable if reward ≥ max_reward − 3000

This defines a relative stability region under each thrust level.

---

# Results

## 800N

Global max reward: -10582.84
Stable threshold: -13582.84

always_on:

* best_r0 = 1.005
* stable_r0 = {1.005}

Gated:

* best_r0 = 1.005
* stable_r0 = {1.005, 1.05}

Interpretation:

Under moderate thrust, gated expands the region of attraction compared to always_on.
The system exhibits a single-peak structure with gradual degradation as r0 increases.

---

## 2000N

Global max reward: +5380.44
Stable threshold: 2380.44

always_on:

* best_r0 = 1.005
* stable_r0 = ∅

Gated:

* best_r0 = 1.005
* stable_r0 = {1.005}

Interpretation:

At high thrust, gated enters a positive reward regime at small initial deviation.
However, the near-optimal stability region compresses dramatically.
always_on completely loses its near-optimal domain.

---

# Saturation Analysis

Observations from saturation_rate_mean plots:

800N:

* always_on saturation increases significantly with r0
* gated remains near zero saturation until largest deviation

2000N:

* always_on saturates earlier (r0 = 1.05)
* gated delays saturation

This suggests a correlation between early saturation and reward collapse.

---

# Structural Conclusions

1. Thrust scaling is non-monotonic.

Increasing thrust from 800N to 2000N:

* Enables higher peak performance (+5380)
* Compresses near-optimal stability region

Higher actuation strength improves local optimality but reduces global robustness.

2. Control structure matters more at high thrust.

Under 2000N:

* gated retains minimal stability
* always_on collapses structurally

3. Evidence of control-induced instability.

Earlier saturation under high thrust correlates with rapid reward degradation.

---

# Engineering Significance

Day15 marks transition from:

"Which controller performs better?"

To:

"How does actuation strength reshape the region of attraction?"

This establishes:

* Explicit stability boundary characterization
* Empirical evidence of non-monotonic thrust scaling
* Structural comparison between control architectures

No controller changes were made during this phase.

---

# Next Step Preview (Day16)

Mechanism analysis:

* Investigate final_radius_error behavior
* Examine inward plunge indicators
* Connect saturation patterns to trajectory dynamics

Objective:
Move from boundary characterization to dynamical mechanism explanation.


