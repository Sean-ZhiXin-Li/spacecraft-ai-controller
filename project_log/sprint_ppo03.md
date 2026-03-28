# Day 3 Project Log

## Title

PPO Sanity Check Completed: From Broken Signals to a Live Training System

## Date

March 28, 2026

## Core Goal of Today

Today was not about making PPO strong.
Today was about proving that the PPO training system is **alive**, connected to the real environment, and capable of learning.

The success criterion for Day 3 was:

* real rollout through the environment
* non-constant rewards
* changing actions
* changing losses
* no NaN

By the end of today, that goal was achieved.

## What I Did

### 1. Located the actual PPO training loop

At first, there was confusion between the inference controller and the real training code.
I checked the project structure and confirmed that `ppo_orbit/ppo.py` contains the real PPO training loop with `optimizer.step()`.
This established the correct file for Day 3 debugging and sanity validation.

### 2. Added Day 3 sanity logging into the training pipeline

I inserted logging into both rollout and update stages so the system could be observed directly during training.
The added diagnostics included:

* rollout reward
* raw/clipped action statistics
* actor loss
* critic loss
* total loss
* entropy
* KL / clip fraction diagnostics
* NaN checks for action, reward, and loss

This made the PPO pipeline inspectable instead of opaque.

### 3. Fixed rollout / loop placement bugs

While inserting logging, I initially placed several lines outside the correct training loop scope, which caused many variable-definition errors.
I then repaired the structure by restoring:

* correct `while steps_collected < BATCH_STEPS:` placement
* correct variable scope for `reward`, `a_raw_np`, `done`, and `steps_collected`
* correct update logging placement inside the PPO minibatch loop

This restored the executable training structure.

### 4. Ran PPO sanity training and verified that the system is alive

After the logging and structure fixes, I ran short PPO sanity training.
The output showed:

* reward changes across rollout steps
* action mean / std changes over time
* actor and critic losses changing normally
* KL values in a reasonable range
* no NaN failures

This confirmed that PPO is not a dead system and is actually training.

### 5. Identified the next-layer issue: stability, not validity

After passing the basic sanity test, I continued examining later outputs.
The result was clear:

* PPO can sometimes reach much better epochs
* but the reward is still unstable and oscillatory
* the remaining issue is not whether PPO works
* the remaining issue is whether PPO can learn more stably

This means Day 3 is complete, and the next problem belongs to stability tuning rather than system validation.

## Key Technical Fixes / Findings

### Confirmed successful Day 3 conditions

* PPO rollout uses the real environment
* losses update normally
* actions are not constant
* rewards are not constant
* no NaN explosions

### Important mechanism fixes made today

* removed the incorrect `abs(entropy.mean())` usage from the entropy bonus path
* improved KL estimation to a more standard approximation
* reduced curriculum intensity in early training
* raised sigma floor to resist premature action-collapse
* reduced critic weight relative to earlier versions
* limited update aggressiveness by capping training iterations

### Main interpretation

The current PPO system is:

* **valid**
* **trainable**
* **alive**

but still:

* not yet consistently stable
* capable of good epochs, but not yet able to retain them reliably

## Evidence Summary

During testing, PPO produced:

* changing rollout rewards
* meaningful variation in action outputs
* multiple epochs with substantially improved reward compared with weaker epochs
* KL values in a reasonable operating band
* stable training without numerical blow-up

This is enough to conclude that the Day 3 objective was achieved.

## Final Conclusion for Day 3

Day 3 is complete.

The correct conclusion is **not**:

> PPO failed to reach the target.

The correct conclusion is:

> PPO training pipeline is alive and learning under real environment rollout. The unresolved issue is reward stability, not system validity.

## What Changed in My Understanding Today

Before today, the main question was:

* Is PPO broken?

After today, the main question became:

* How can I make PPO learn more stably?

That is a major transition.
It means the project has moved from **debugging existence** to **optimizing behavior**.

## Next Step

Move into the next phase:

* reduce actor update aggressiveness
* stabilize reward behavior
* prepare PPO to become a comparable controller inside the W03 pipeline

## One-Sentence Project Log Summary

Day 3 successfully turned PPO from a questionable training module into a verified live learning system; the next challenge is stability, not existence.
