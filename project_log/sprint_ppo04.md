# Day 4

## PPO Stabilization Day

---

# Overview

**Goal of Day 4:**

> Reduce oscillation while keeping PPO learning alive

This marks the transition from:

* Day 3 → *"PPO is alive"*
* Day 4 → *"PPO is controllable"*

---

# Key Changes Implemented

## 1. Reduced Actor Update Aggressiveness

* Lowered `lr_cap`
* Removed extra low-KL acceleration
* Fixed `TRAIN_ITERS = 20` (disabled adaptive increase)

**Purpose:**
Prevent over-aggressive updates that caused instability and large reward drops.

---

## 2. Controlled PPO Update Dynamics

* Maintained KL-based adaptation (single layer)
* Removed stacked adaptation mechanisms

**Effect:**
System shifted from:

* "multi-trigger acceleration"
  → "single, controlled adaptation"

---

## 3. Reduced Advantage Noise (Critical Step)

* Changed:

```python
LAMBDA = 0.97 → 0.90
```

**Purpose:**

* Shorten credit assignment horizon
* Reduce variance in advantage estimation
* Improve policy update stability

---

## 4. Reduced Critic Influence

* Changed:

```python
vf_coef → 0.25
```

**Purpose:**

* Prevent critic from over-shaping policy updates
* Reduce bias/noise in advantage signal

---

# Observed Training Behavior

## Before Stabilization

* Reward pattern:

  * Rapid improvement → catastrophic drop
* Example:

```
-7800 → -13000
```

* Characteristics:

  * High oscillation
  * No stable region
  * Policy unable to retain good behavior

---

## After Stabilization

### 1. Emergence of Stable Region (Attractor)

* Reward band:

```
-7000 ~ -8000
```

* Behavior:

  * PPO remains within region
  * No catastrophic collapse

---

### 2. Oscillation Type Changed

From:

```
large jumps (collapse)
```

To:

```
small local oscillations
```

---

### 3. KL Behavior

* Range:

```
~0.004 – 0.007
```

* Interpretation:

* Not too small → learning active

* Not too large → updates controlled

---

### 4. Exploration Stability

* `std ≈ 0.60+`

* No collapse in entropy

* Policy remains stochastic

---

# Key Insight

> PPO instability was not caused by overly aggressive updates,
> but by **noisy and unreliable advantage signals**.

Fixing stability required:

* Reducing horizon (λ ↓)
* Reducing critic dominance (vf_coef ↓)

---

# Core Finding

> PPO has entered a **stable attractor regime**.

Meaning:

* It can reach a good policy region
* It can remain near that region
* It no longer catastrophically forgets

---

# Trade-offs Observed

| Aspect           | Result                                           |
| ---------------- | ------------------------------------------------ |
| Stability        | ✅ Strong improvement                             |
| Learning speed   | 🟡 Slightly reduced                              |
| Peak performance | ✅ Improved                                       |
| Oscillation      | ✅ Reduced (collapse → drift → local oscillation) |

---

# Final Conclusion (Day 4)

> Reducing actor aggressiveness alone was insufficient.
>
> Stability emerged only after:
>
> * lowering advantage noise (λ ↓)
> * weakening critic influence (vf_coef ↓)

---

# Next Step (Day 5)

Transition from:

```
PPO training
```

To:

```
PPO as a controller in W03 pipeline
```

### Tasks:

1. Integrate PPO into controller interface
2. Run comparison:

```
always_on vs gated vs PPO
```

3. Generate comparable metrics:

* avg_error
* reward
* thrust

---

# Meta Reflection

This marks a transition from:

```
engineering (make PPO work)
```

To:

```
research (understand controller behavior)
```

---

# Summary

> Day 4 successfully transformed PPO from an unstable learner
> into a **controlled and stable learning system**,
> ready to be used as a comparable controller in W03.

---
