# PPO Transfer Plan

## Objective

Transfer the successful explicit phase-controller structure into PPO without changing the physics or discarding the existing PPO pipeline.

## Current Evidence

- The explicit phase controller generalizes to a limited but real subset of `(r0, dt, thrust)` settings.
- PPO does not reproduce first crossing under the validated baseline.
- The stable structure now exists outside PPO, which makes direct behavior transfer feasible.

## Recommended Learning Path

### 1. Behavior Cloning First

Use `analysis/phase_controller_dataset/phase_controller_dataset.npz` as the first training source.

Targets:
- clone the explicit controller action
- optionally predict the controller phase as an auxiliary head

Why first:
- PPO currently lacks the control structure needed for descent, capture, and lock
- cloning can provide that structure before policy optimization

### 2. PPO Fine-Tuning Second

Initialize PPO from the cloned policy, then fine-tune under the same strict evaluation setups used by:
- `scripts/orbit_lock_generalization.py`
- `scripts/orbit_lock_benchmark.py`

Fine-tuning objective:
- preserve the first-crossing descent behavior
- improve post-crossing regulation
- reduce radial oscillation after capture

### 3. Optional Reward Shaping Aligned With Phase Transitions

Only after cloning and benchmark validation are in place.

If reward shaping is needed, align it with the explicit controller phases:
- descent: reward progress toward first crossing
- capture: reward damping of `v_r`
- lock: reward sustained low `|r_error|`, low `|v_r|`, and circular tangential speed

This should remain phase-aligned, not global.

## What To Build Next

1. A behavior-cloning training script that consumes the phase-controller dataset.
2. A cloned-policy benchmark using the same representative setups as `orbit_lock_benchmark`.
3. PPO fine-tuning only after the cloned policy reproduces first crossing reliably.

## What Not To Touch Yet

- PPO reward shaping
- PPO hyperparameters
- environment physics
- additional controller hacks

Those should stay frozen until cloning establishes whether the phase structure is learnable directly.

## Shortest Path

1. Use the successful explicit-controller dataset for behavior cloning.
2. Benchmark the cloned policy against explicit and probe controllers.
3. If the cloned policy preserves first crossing, fine-tune with PPO for post-crossing lock quality.
