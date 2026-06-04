# Critical Missing Pieces

## 1. Stable Benchmark Contract

The project needs a formal benchmark contract before adding more phases:

- fixed case list
- fixed terminal controller
- fixed metrics
- fixed CSV schema
- fixed definitions for crossing and recoverability

Without this, future phase comparisons will become fragile.

## 2. Shared Rollout Core

The dynamics, termination, state-machine transitions, and metric calculations should be modularized. This is the biggest engineering gap.

The goal is not refactoring for style. The goal is preventing scientific drift.

## 3. Transfer-Family Dataset

Phase36B needs a dataset that records trajectories by family, not just final rows.

Important fields:

- radius history
- vr history
- vt error history
- crossing step
- closest approach step
- energy proxy
- angular momentum proxy
- Phase34 handoff quality

## 4. State-Space Analysis

The project should map where recoverable states exist in the simplified state space:

- radius error
- radial velocity
- tangential velocity error
- energy proxy
- angular momentum proxy

This would make "recoverability basin" more than a threshold label.

## 5. Robustness Analysis

Before adding 3D physics, test robustness in the 2D model:

- small perturbations to initial velocity
- small perturbations to thrust scale
- small perturbations to target radius
- noise in state estimates

This would reveal whether Phase34 and future families are brittle.

## 6. Trajectory Clustering

Phase36B should cluster trajectories by behavior:

- crosses early
- grazes target radius
- approaches but fails to commit
- overspeeds
- drifts one-sided
- crosses with bad sync
- crosses with good handoff

This is more useful than adding more named variants.

## 7. Control-Theoretic Framing

The project would benefit from a concise control-theoretic framing:

- switching controller
- terminal basin
- handoff condition
- viability/recoverability region
- phase-dependent control law

This would make the work easier for mentors to evaluate.

## What Not To Prioritize Yet

Do not prioritize:

- cosmetic README changes
- more demo figures
- 3D physics
- SPICE
- C++ rewrite
- larger PPO models
- broad future-space roadmap expansion

These can wait until the core 2D transfer-family result is cleaner.

