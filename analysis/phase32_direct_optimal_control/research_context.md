# Phase 32 Research Context

## What Previous Layers Failed

- Phase 25 showed existing crossings are near recoverable under relaxed thresholds, with tangential velocity as the dominant blocker.
- Phase 28 showed window-producing and crossing-producing families are mostly disjoint.
- Phase 29 Burn-A family selection did not improve crossings, recoverability, CAPTURE, or success.
- Phase 30 Burn-A endpoint search mapped endpoint manifolds but did not change crossing-state structure.
- Phase 31 bounded global transfer families still did not improve recoverable crossings, CAPTURE, or success.

## Why Heuristics May Be Insufficient

- Local, staged, endpoint, and named transfer-family approaches all encode limited structure.
- They choose behaviors or coarse families, not a continuous state-control trajectory optimized against recoverability.
- A negative result across these layers means the next question is theoretical reachability under the current physics.

## Why Optimal Control Is Necessary

- Direct optimal control can optimize all controls over a finite horizon at once.
- It can target radius, radial velocity, tangential velocity, sync error, and effort simultaneously.
- It provides an upper-bound style baseline rather than another production controller heuristic.

## Architecture Comparison

- Heuristic control: immediate feedback rules.
- Staged transfer: fixed Burn A, coast, Burn B, handoff.
- Family search: bounded named transfer templates.
- Global transfer: architecture-level burn/coast schedules.
- Continuous optimization: direct solve over a full control sequence under the same dynamics.

## Optimal Under Current Physics

- State is `[x, y, vx, vy]`.
- Control is bounded normalized thrust `[u_x, u_y]`.
- Dynamics reuse the project gravitational acceleration and thrust scaling.
- No recoverability, CAPTURE, LOCK, reward, or physics threshold is changed.
- CasADi availability in this run: `unavailable, fallback used: No module named 'casadi'`.