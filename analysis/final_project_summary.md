# Final Project Summary(2D_First_result)

## Project Goal

Achieve real orbit insertion and orbit lock in the existing orbital environment without changing the physics model.

## Final Outcome

The project now has a working **phase-structured explicit controller** that can:
- reach first crossing in the physically reachable regime
- enter capture and lock phases
- achieve strict success in a narrow but real subset of configurations

PPO does not yet learn this structure on its own.

## Why PPO Fails

PPO fails for structural reasons, not just optimization noise.

Observed failure mode:
- PPO remains a continuous reactive policy
- it does not reliably learn the transition from descent to capture to lock
- in benchmarked setups it either never reaches first crossing or produces uncontrolled radial behavior

Evidence:
- benchmark successes: explicit `2 / 3`, PPO `0 / 3`
- representative successful setup:
  - explicit crosses and succeeds
  - PPO never crosses

In short:
- PPO is missing **control structure**
- not just the right gains

## Why Probe Is Insufficient

The probe controller is useful only as a reachability baseline.

What it does well:
- proves that first crossing is physically possible in the reachable regime
- provides the strongest simple descent behavior through full retrograde thrust

What it cannot do:
- it does not switch behavior after crossing
- it does not stabilize into lock
- it is not a controller for orbit maintenance

So probe is a **descent-only baseline**, not a full insertion controller.

## Why the Explicit Controller Works

The explicit controller works because it is **phase-structured**.

It separates the task into distinct control modes:

1. `DESCENT`
- apply full retrograde thrust aligned with velocity
- remove angular momentum and orbital energy aggressively
- purpose: guarantee first crossing

2. `CAPTURE`
- after first crossing, reverse control emphasis
- damp radial velocity
- support tangential velocity toward circular orbit
- purpose: avoid immediate fly-through or rebound

3. `LOCK`
- once near the target orbit, apply smaller stabilization
- purpose: maintain orbit under strict tolerances

This is the key project result:
- the controller succeeds when the control law is matched to the **phase of insertion**
- not when one continuous policy tries to solve everything at once

## What "Phase-Structured Control" Means

Phase-structured control means the policy does not use one fixed feedback rule everywhere.

Instead, it changes behavior depending on where the spacecraft is in the insertion process:
- before crossing: maximize descent effectiveness
- near and after crossing: damp radial motion and recover tangential support
- near orbit: stabilize

This is fundamentally different from:
- PPO’s single reactive policy
- the probe’s single fixed retrograde behavior

The core lesson is:
- insertion is not one homogeneous control problem
- it is a sequence of qualitatively different subproblems

## Generalization

The explicit controller generalizes, but only in a narrow regime.

From the generalization sweep:
- tested grid: `r0`, `dt`, `thrust_scale`
- successful setups: `5 / 36`

Working region:
- very small initial offset
- moderate to high `dt`
- moderate thrust, not maximal thrust

This means:
- the controller is real, not a single lucky trajectory
- but the valid region is still limited and should be treated carefully

## Benchmark Comparison

Representative benchmark result:
- explicit controller: best overall
- probe: good for descent only
- PPO: fails completely in the same setups

High-level ranking:
1. explicit phase controller
2. probe max retrograde
3. PPO baseline

The explicit controller is now the correct benchmark target for learning transfer.

## Learning Transfer

The first learning-transfer stage has now been run.

Completed path:
1. behavior cloning from the explicit controller
2. short PPO fine-tuning from the cloned policy
3. fixed-baseline comparison against the explicit controller and probe

Result:
- behavior cloning did not reproduce the first crossing on rollout
- short PPO fine-tuning from the cloned policy also did not recover first crossing
- the explicit controller remains the only fully successful controller in the validated comparison

This means the repository should not return to blind PPO retraining. Any later learned controller should keep the explicit phase structure visible, constrained, or directly supervised.

## Phase 3 Hybrid Residual Result

Phase 3 tested hybrid residual learning around the explicit controller.

Completed residual tests:
- zero-residual hybrid exactly preserved explicit-controller success
- alpha sweep showed no change because the residual policy output remained zero
- tiny unconstrained nonzero residual tuning harmed success
- magnitude-only residual preserved the accepted zero-residual checkpoint, but nonzero magnitude bias did not improve the rollout objective

Phase 3 did not prove a useful learned residual. It did show that the safe hybrid design principle is to preserve explicit structure first and reject residual authority unless it improves rollout metrics without losing success.

## Dataset

Two datasets now exist:

1. Original phase-controller dataset
- successful and crossing trajectories
- includes observations, actions, phase labels

2. Balanced phase dataset
- `DESCENT` downsampled
- `CAPTURE` and `LOCK` upsampled
- intended for more stable supervised learning

This balanced dataset is the correct starting point for behavior cloning.

## Final Project Message

The project does not need more blind PPO tuning.

The main result is:
- **orbit insertion requires explicit phase structure**

PPO fails because it lacks that structure.
Probe is insufficient because it only solves descent.
The explicit controller works because it encodes descent, capture, and lock as separate modes.

## Recommended Next Step

Move beyond single-regime 2D validation by testing the explicit phase structure across multiple 2D orbit regimes before moving to 3D or systems integration.

The immediate next target should be multi-orbit / multi-regime generalization:
- keep physics and controller structure explicit
- vary initial radius offset, target radius, thrust, and timestep in controlled regimes
- measure where the phase controller succeeds, where it fails, and which failure modes are structural
- only revisit learned residuals after the successful/failing regimes are mapped clearly
