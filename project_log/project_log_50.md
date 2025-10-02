# Project Log — Day 50

**Date:** 2025-10-02  
**Owner:** Sean  
**Component:** `ExpertController v3.1` evaluation  
**Scope:** Add `transfer_2phase` scenario, introduce failure logging & plots, harden env/controller I/O.

---

## Summary

- Evaluated `ExpertController v3.1` across **four scenes**: `circular`, `elliptic`, `transfer`, and **`transfer_2phase`** (two-phase Hohmann-style: raise apogee → circularize).
- Added a **failure logging pipeline** (CSV + plots) with clear reasons to make issues searchable and comparable across runs.
- Implemented a **robust runner** that tolerates Gym/Gymnasium API differences and controller method-name drift.
- Began a **Velocity Backfill Adapter** so controllers needing `(pos, vel)` can still run when observations provide only position.

**Status:** Runner + logging complete. Controller invocation edge cases remain (signature/packing of `__call__`), so end-to-end for all scenes is **partially blocked** pending a small shim or consistent I/O contract.

---

## Changes in Day 50

### 1) New scenario
- **`transfer_2phase`** added to the test matrix.
  - Reset options add semantic hints:  
    `two_phase=True, phase1="raise_apogee", phase2="circularize"`  
  - Goal: closer to real multi-burn transfers; clearer failure modes vs single-phase `transfer`.

### 2) Failure logging & plots
- CSV: `logs/day50/test_log_day50.csv`
  - Columns:  
    `ts, preset, scene, seed, max_steps, steps, terminated, truncated, reward_sum, orbit_error_final, failure_reason, controller_target_radius`
- Failure reasons (priority):
  1. `Large final error` (`final_err > err_threshold`)
  2. `Did not converge` (no termination by step limit)
  3. `Unstable trajectory` (episode return below threshold)
  4. `Missing error metric` (no usable error)
- Plots (on failure only): `logs/day50/figs/<scene>_seed<seed>_err.png`

### 3) Runner resilience
- **Env creation:** pass only `scene` to `MultiOrbitEnv.__init__` (fallback to attribute), push other hints via `reset(options=...)`.
- **Gym/Gymnasium compatibility:** handles 5-return and legacy 4-return step APIs.
- **Controller inputs:** `target_radius` discovered (from `info/env/options`) with **CLI override** `--target_radius`.

### 4) Velocity Backfill (in progress)
- If observations lack velocity:
  - First step: use `info['velocity']` when present; otherwise zeros.
  - Next steps: finite differences with `dt` (`info/env` → fallback `1.0`).
- Tries multiple call shapes when invoking the controller:
  - `(pos, vel)`; `((pos, vel))`; `concat([pos, vel])`; `{"pos": pos, "vel": vel}`
- Avoids passing unknown kwargs (e.g., `info`) to `__call__` when not supported.

---

## How to run (current)

```bash
# Tip: pass the Day49 target radius explicitly for now
python test_day_50.py --steps 2000 --seed 50 --preset voyager1 --target_radius 9375000297472.0
```

_Default thresholds:_  
- `--err_threshold=0.1`  
- `--unstable_reward_threshold=-50.0`  
- Logs under `logs/day50/`

---

## 📎 Artifacts

```
logs/
  day50/
    test_log_day50.csv
    figs/
      circular_seed<seed>_err.png
      elliptic_seed<seed>_err.png
      transfer_seed<seed>_err.png
      transfer_2phase_seed<seed>_err.png
```

_Optional next:_ `logs/day50/failure_cases.md` (see scaffold below).

---

## Issues & current understanding

1) **Env ctor signature mismatch**  
   - `MultiOrbitEnv.__init__` didn’t accept `preset/reset_options`.  
   - **Mitigation:** set `scene` only; use `reset(options=...)` for the rest.

2) **Controller requires `target_radius`**  
   - Added discovery + `--target_radius` to ensure reproducibility across scenes.

3) **Controller action interface drift**  
   - No `.act()`. Uses `__call__` with varying arity/packing.  
   - Some runs report “missing `vel`” or reject the `info` kwarg.  
   - **Mitigation (WIP):** adapter now prefers `(pos, vel)` and tries tuple/concat/dict packing if needed.

4) **Observations may lack velocity**  
   - Backfilled via finite differences; accuracy depends on reliable `dt`.

---

## 📓 Quick debug plan (next session)

- Print once after the first reset:
  - `obs0` type/shape/keys  
  - `info0` keys  
  - `ExpertController.__call__` signature (class & instance)
- Lock the **single canonical** call shape the controller expects, then prune adapter branches.

---

## / 🟡 Checklist

- [x] Add `transfer_2phase` scenario & reset hints  
- [x] CSV logging with `failure_reason`  
- [x] Error plots on failures  
- [x] CLI: `steps/seed/preset/thresholds/logdir/target_radius`  
- [x] Gym/Gymnasium step API compatibility  
- [🟡] Finalize controller invocation shim  
- [🟡] Author `failure_cases.md` with plots + CSV snippets

---

## `failure_cases.md` scaffold (to create next)

```md
# Failure Cases — Day 50

This note documents representative failures with plots and log snippets.

## transfer
- **Symptom:** Overshoot during transfer burn → large final error.
- **Evidence:** `figs/transfer_seed50_err.png`
- **CSV row:** seed=50, steps=..., orbit_error_final=..., failure_reason=Large final error

## transfer_2phase
- **Symptom:** Phase 2 circularization incomplete → long eccentric orbit.
- **Evidence:** `figs/transfer_2phase_seed50_err.png`
- **CSV row:** seed=50, steps=..., orbit_error_final=..., failure_reason=Did not converge
```

---

## Notes & lessons

- Push scenario configuration into `reset(options)` to avoid ctor brittleness.  
- Make controller I/O **explicit**: document whether it expects `(pos, vel)` or a packed payload.  
- A small, consistent adapter/shim prevents runner breakage as APIs evolve.

---

## Suggested commit message

```
[Day50] Add transfer_2phase; failure logging & error plots; env/controller I/O hardening; WIP velocity backfill adapter
```
