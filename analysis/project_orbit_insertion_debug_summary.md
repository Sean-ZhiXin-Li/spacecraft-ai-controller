# Project Orbit Insertion Debug Summary

## Scope

This audit treats the PPO shutdown work, anti-shutdown wrapper, explicit controllers, and impulse experiments as one continuous debugging effort across the whole project.

The goal of this pass was to debug the full insertion loop end-to-end instead of continuing to patch local controllers in isolation.

## Changed Files

- `envs/orbit_env.py`
- `scripts/project_orbit_insertion_debug.py`

## What Was Audited

The control loop was checked end-to-end:

- observation construction in `OrbitEnv`
- state normalization in `ppo_orbit/ppo.py`
- controller output meaning
- action clipping and thrust mapping
- environment step physics
- timestep and horizon
- reward-independent success condition
- termination logic
- physical control authority versus required orbit change

## High-Confidence Findings

### 1. The main blocker is physics scale plus horizon mismatch

At the project target radius:

- `target_radius = 7.5e12 m`
- `circular_speed = 4207.17 m/s`
- `orbital_period ≈ 355 years`

The current evaluation horizons are tiny compared with that orbital scale:

- `4000 steps * 2 s = 8000 s`
- `20000 steps * 2 s = 40000 s`
- `60000 steps * 2 s = 120000 s`

Required average radial speed to close the initial radius gap:

- `r0 = 1.01`, `4000 steps`: `9.38e6 m/s`
- `r0 = 1.03`, `4000 steps`: `2.81e7 m/s`
- `r0 = 1.05`, `4000 steps`: `4.69e7 m/s`

Those numbers are not remotely physically reachable.

Even the optimistic constant-acceleration displacement bound with full thrust is far too small:

- max acceleration: `27.70 m/s^2`
- `4000 steps`: max radial displacement bound `8.86e8 m`
- `20000 steps`: max radial displacement bound `2.22e10 m`
- `60000 steps`: max radial displacement bound `1.99e11 m`

Compare those against the actual radius gaps:

- `1% gap = 7.5e10 m`
- `3% gap = 2.25e11 m`
- `5% gap = 3.75e11 m`

So:

- `1%` gap is impossible at `4000` and `20000` steps
- `3%` and `5%` gaps are impossible even at `60000` steps under this optimistic bound

This is the primary reason the project is not inserting.

### 2. The success metric is too loose and does not represent true insertion

Current success tolerance in `OrbitEnv`:

- `tol_r = 1.8e-2`
- radius tolerance = `1.35e11 m`

That means:

- a start at `r0 = 1.01` is already inside the radius tolerance band

The debug run showed exactly that:

- `explicit:orbit_lock_controller`, `r0 = 1.01`
- `success = true` after `52` steps
- `final_radius_error ≈ 7.5e10 m`
- `radius_crossings_total = 0`

So the environment can report “success” without real insertion, without target crossing, and without orbit lock.

This is a project-level blocker because it makes the training/evaluation target inconsistent with the real objective.

### 3. Action-to-physics mapping is not the main problem

The audit found no evidence that the controller output is silently ignored.

Evidence:

- PPO mean action norms around `0.0035` map to about `0.10 m/s^2` average thrust acceleration
- the explicit controller maps to about `0.81–2.23 m/s^2` depending on case
- the debugged discrete impulse now fires and produces an immediate radius change:
  - `impulse_trigger_count = 1`
  - `immediate_delta_radius_after_impulse = -41.96 m`

So action mapping is functioning.

The real problem is that:

- the commanded accelerations are being applied
- but the project is asking for orbit-scale radius movement on an infeasible timescale

## Controlled Tests

The new debug script ran:

- PPO `speed_refine_50`
- PPO `state_vr_nonlinear_100`
- current explicit orbit-lock controller
- max inward full-thrust controller
- max retrograde full-thrust controller

using the actual `OrbitEnv` with:

- `thrust_scale = 20000`
- `dt = 2`
- `use_action_smoothing = False`
- `use_orbit_capture_assist = False`

Important results:

### PPO controllers

At `r0 = 1.05`, `4000` steps:

- PPO mean action norm is only about `0.0036–0.0037`
- average thrust acceleration is about `0.10 m/s^2`
- cumulative delta-v is about `800 m/s`
- `radius_crossings_total = 0`

That is too small to move a `5%` radius error at this project scale over this horizon.

### Explicit controller

At `r0 = 1.05`, `4000` steps:

- mean action norm `≈ 0.0293`
- average thrust acceleration `≈ 0.81 m/s^2`
- cumulative delta-v `≈ 6488 m/s`
- `radius_crossings_total = 0`

The explicit controller is more active than PPO, but still cannot produce insertion under the current horizon and target scale.

### Full-thrust scripted controls

At `r0 = 1.05`:

- max inward thrust:
  - immediately destabilizes and terminates early
  - still no crossing
- max retrograde thrust:
  - runs the full horizon
  - still no crossing even at `60000` steps
  - minimum radius error after `60000` steps is still about `3.749996e11 m`

This is the strongest project-level result:

- even extreme scripted full-thrust control does not cross the target radius from `1.05` under the current setup

So the current failure is not just bad learned control.

## Top 3 Blockers

### Blocker 1

The insertion task is physically infeasible at the current target scale and rollout horizon.

This is the main blocker.

### Blocker 2

The current success condition is not aligned with true insertion.

It is possible to “succeed” while still far from true orbit lock.

### Blocker 3

The project has been debugging the wrong objective locally.

Because the horizon/metric pair is misaligned, controller tweaks mostly change bounded local behavior without any chance of producing real insertion.

This is why:

- PPO patches
- anti-shutdown layers
- explicit oscillators
- phase logic
- impulses

all failed to reach true insertion.

## What Should Be Fixed First

Fix the **task specification**, not PPO, not reward, and not another local controller.

The shortest path is:

1. make the insertion task physically reachable for the chosen horizon
2. tighten success criteria so they cannot be satisfied at the initial state
3. only then revisit controller or PPO training

## What Should NOT Be Touched Yet

Do **not** touch these yet:

- PPO reward shaping
- PPO hyperparameters
- more anti-shutdown logic
- more phase/impulse controller hacks

Those are downstream of the actual blocker.

## Small Fix Applied

One clear project bug was fixed during the audit:

- `OrbitEnv._get_obs()` returns 5 values
- `observation_space` was still declared as shape `(4,)`
- it is now fixed to shape `(5,)`

This does not solve insertion, but it removes a real environment inconsistency.

## Why The Project Is Still Not Inserting

Because the project currently asks the controller to close orbit-scale radius gaps on timescales that are too short for the modeled physics and thrust authority, and then evaluates success with a metric that can already be satisfied near the starting state.

So the dominant failure mode is:

- **physics scale + horizon mismatch**

with a secondary blocker:

- **wrong success/evaluation target**

not:

- broken action mapping

## Exact Next Fix

The exact next fix to attempt is:

**Redefine the insertion task to a physically reachable regime before any retraining.**

Concretely:

- keep the same physics
- keep the same reward for now
- choose an initial radius offset that can plausibly be closed within the chosen horizon
- and tighten the success criterion so the start state is not already “close enough”

If the project wants to keep `r0 = 1.01–1.05`, then the horizon or effective maneuver timescale must increase drastically.

If the project wants to keep the current horizon, then the starting radius offset must be much smaller.

That is the shortest path to real insertion.
