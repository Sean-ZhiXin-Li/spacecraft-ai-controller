# Day 9 - Reward Diagnosis and Failure Mode Identification

## Goal

The goal of Day 9 was to diagnose why PPO still failed to achieve stable orbital convergence, even after multiple reward adjustments. Instead of treating PPO as a separate project, I continued using the same W03 pipeline and compared its rollout behavior under the same control-analysis framework.

The main question for today was:

**Why can PPO sometimes move in the correct direction, but still fail to stabilize near the target orbit?**

---

## What I did today

Today I focused on reward diagnosis rather than large-scale pipeline changes.

### 1. Fixed and verified the rollout analysis pipeline

I made sure the rollout pipeline could save and analyze the following control-related signals:

* radius `r`
* radial velocity `v_r`
* thrust magnitude `thrust_norm`
* alignment with radial direction `cos(thrust, radial)`
* alignment with tangential direction `cos(thrust, tangential)`

This allowed me to compare controller behavior at the mechanism level, not only through total reward.

### 2. Identified and corrected a major scaling issue

A critical bug was found in the PPO pipeline: action scaling was being applied twice.

This meant PPO actions were effectively over-amplified before entering the physics step, which produced unrealistic rollout behavior and misleading reward outcomes. After removing the duplicate scaling, PPO behavior became physically interpretable again.

This was one of the most important debugging results of the project so far, because it changed the meaning of all later experiments.

### 3. Tested several reward-shaping directions

I experimented with multiple reward modifications, including:

* positive-only progress shaping
* radial-speed penalties
* directional rewards
* overshoot penalties
* damping terms
* stop-fall / braking terms near the target radius

These experiments were not treated as random tuning. Each one was designed to test a specific failure hypothesis.

---

## Main observations

### 1. PPO is highly sensitive to reward structure

Once the scaling bug was fixed, PPO no longer behaved randomly. Instead, it began to learn consistent but incomplete control strategies.

This means the agent is responsive to reward design, but also extremely sensitive to small reward changes.

### 2. Two dominant failure modes appeared repeatedly

Across today’s runs, PPO repeatedly converged to one of two major failure modes:

* **inward spiral / inward collapse**
* **outward escape / outward drift**

In both cases, the learned policy was not random. It found a locally stable behavior that improved reward under the current shaping, but that behavior was not the desired orbital stabilization policy.

### 3. PPO learned partial control structure, not full stabilization

Some runs showed that PPO could learn useful components of orbital control, such as:

* sustained inward correction
* reduced explosive acceleration after scaling was fixed
* more interpretable thrust behavior
* stronger sensitivity to radial error

However, PPO still failed to learn the full sequence:

**approach target radius -> slow down radial motion -> maintain stable orbital structure**

In other words, PPO could learn a direction, but not a complete stabilization phase.

### 4. Stable convergence is harder than simple correction

Today’s experiments showed that moving toward the target radius is much easier than stopping at the right place.

This is the key difference between:

* **partial orbital control**
* **true stable orbital convergence**

The current reward design can teach PPO to improve part of the trajectory, but it still struggles to encode the full control logic required for stable convergence.

---

## Most important result of Day 9

The most important result today was not a fully successful PPO controller.

It was this:

> **The main failure modes are now identified, and the reward sensitivity of PPO under the corrected W03 pipeline is now much better understood.**

This means the project has moved from trial-and-error tuning to mechanism-level understanding.

---

## Mechanism conclusion

PPO did not fail because it was incapable of learning control. Instead, it repeatedly learned locally rewarding but structurally incomplete behaviors.

After the action-scaling bug was fixed, the system became much more interpretable. From there, the main issue was no longer randomness, but reward-induced policy bias.

Under different reward variants, PPO tended to settle into either inward-collapse or outward-escape solutions, which suggests that stable orbital convergence requires either:

* more carefully staged reward design,
* stronger controller priors,
* or a more explicit separation between approach behavior and stabilization behavior.

---

## Day 9 summary

Today was not a reward-tuning day in the simple sense. It became a reward-diagnosis and failure-mode analysis day.

The biggest progress was:

* fixing a major scaling bug,
* making rollout behavior physically interpretable,
* and identifying the core failure structure of PPO in this orbital control setup.

Even though stable orbit insertion was not achieved yet, the project now has a much stronger experimental story and a much clearer technical conclusion.

---

## Next step

The next step is to stop broad reward guessing and focus on one of two more principled directions:

1. staged reward / staged training, where PPO first learns radius approach and only later learns stabilization;
2. hybrid control design, where PPO is guided by stronger physical priors or expert structure.

This will be a better long-term path than continuing to stack many reward terms into a single undifferentiated objective.
