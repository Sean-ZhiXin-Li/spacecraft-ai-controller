# PPO Failure Mechanism in Orbit Insertion

## Overview

PPO fails in this project primarily because orbit insertion is a **phase-structured control problem**, while the learned PPO policy behaves as a **single continuous reactive controller**.

The failure is not best explained by hyperparameters alone. The dominant issue is that PPO does not recover the required control transitions needed to move from descent to capture and then to lock.

## Required Control Structure

Successful orbit insertion in this environment requires at least three distinct phases:

1. **Crossing**
   - remove enough orbital energy and angular momentum to force the spacecraft across the target radius
   - this requires physically effective descent behavior, not just reduced local error
   - without crossing the target radius, the system never enters the regime where radial feedback can reverse sign and form a closed-loop correction

2. **Capture**
   - once crossing occurs, reverse the control emphasis
   - damp radial motion
   - prevent a simple fly-through or one-sided drift continuation

3. **Lock**
   - once near the target orbit, maintain bounded motion under strict tolerances
   - this is a stabilization phase, not a descent phase

These are not small variations of one feedback law. They are different control regimes with different objectives.

## PPO Behavior

Observed PPO behavior is structurally different from the successful controller.

Instead of learning the insertion phases, PPO tends to:

- reduce local error signals
- suppress radial motion
- collapse toward low-action behavior
- avoid strong state transitions

This produces a policy that can look smooth or stable in short diagnostics, but does not perform the physically necessary insertion sequence.

Key insight:

> **PPO learns to stop moving, not to stabilize motion.**

That distinction matters:

- **stop moving**: reduce instantaneous radial velocity and action magnitude
- **stabilize motion**: actively regulate the system through crossing, capture, and lock

PPO is biased toward the first behavior. Orbit insertion requires the second.

This behavior is not just an empirical artifact. It is a structural consequence of PPO's objective, which favors reducing immediate error and action-related instability rather than deliberately inducing the state transition needed for insertion.

## Error Minimization vs. Phase Transition

PPO behaves like an error-minimizing controller in a local sense.

That leads to:

- minimizing radial activity before first crossing
- treating low `|v_r|` as a good terminal-like condition
- failing to preserve the aggressive descent needed to guarantee crossing

This is consistent with PPO's objective: it is rewarded for locally reducing error signals, not for creating the controlled overshoot and phase transition required to enter capture.

Even when PPO does not fully collapse into zero action, it still lacks the phase switch needed after crossing. So it cannot convert local regulation into full insertion.

In short:

- PPO minimizes motion
- successful insertion requires **managed overshoot plus post-crossing regulation**

## Comparison with the Explicit Controller

The explicit controller succeeds because it encodes the missing structure directly:

1. **DESCENT**
   - full retrograde thrust aligned with velocity
   - goal: guarantee first crossing

2. **CAPTURE**
   - after crossing, apply radial damping and tangential support
   - goal: prevent immediate escape after crossing

3. **LOCK**
   - near the target orbit, switch to fine stabilization
   - goal: satisfy strict success criteria

The explicit controller does not rely on one smooth policy to discover these transitions implicitly. It defines them explicitly.

PPO, by contrast, attempts to use one continuous policy over all regimes. That is the central structural mismatch.

## Why This Is Not Primarily a Hyperparameter Problem

Hyperparameters may affect optimization quality, but they do not resolve the core issue:

- the policy still lacks explicit phase structure
- the task requires qualitatively different actions before and after crossing
- a single reactive policy is not naturally aligned with this insertion sequence

So the main blocker is not:

- learning rate
- batch size
- PPO epochs
- entropy coefficient

The main blocker is:

- **missing control structure**

## Practical Conclusion

PPO fails in orbit insertion because it does not learn the phase transitions required by the task.

The learned behavior suppresses motion and error locally, but does not execute the full insertion sequence:

- crossing
- capture
- lock

This is why the explicit phase controller succeeds where PPO fails.

The implication for learning is clear:

- first transfer the successful control structure
- then optimize within that structure

Trying to recover the structure indirectly through raw PPO alone is the wrong level of attack for this task.

## Broader Insight

This failure illustrates a general limitation of reactive policies:

Tasks that require phase transitions and sustained corrective dynamics cannot be solved by minimizing local error alone.

Instead, they require either:

- explicit control structure
- learning frameworks that preserve phase-dependent behavior
