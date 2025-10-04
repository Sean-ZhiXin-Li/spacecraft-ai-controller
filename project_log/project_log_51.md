# Day 51 — Stabilize Replay & Quickrun
**Theme:** *Stabilize replay + quickrun pipeline for fast validation*  
**Date:** 2025-10-04 (PDT)  
**Seed:** 51 · **Preset:** `voyager1` · **Discovered `target_radius`:** `7.5e12`

---

## 🎯 Objectives
- **Deterministic & replayable** rollouts: fixed seed, record JSONL trajectory, strict drift checks.
- **Fast validation**: a 1–2 minute `quickrun` for local/CI smoke.

---

## ✅ What I Implemented Today
1. **Replay pipeline**
   - `scripts/record_replay.py`: records `(t, obs, action, reward, done)` with fixed seed.
   - `scripts/play_replay.py`: replays recorded actions and compares final state & return.
   - Normalizes env signatures (`reset/step` 4-tuple/5-tuple) and supports multiple `OrbitEnv` ctor shapes.
   - **Controller call matches my ExpertController signature**: `ctrl(t, pos, vel)`.

2. **Quick smoke**
   - `scripts/quickrun.py`: short rollout with performance thresholds; supports `--strict`.

3. **Env compatibility patch**
   - `envs/multi_orbit_env.py`: `__init__` now accepts either **scenario string** *or* **SimConfig/dict**.
   - Merges `mu/dt/max_steps/target_radius/seed` from config into `preset_overrides` (backward-compatible).

---

## 🧪 Commands & Key Outputs

### 1) Record reference trajectory
```powershell
python scripts/record_replay.py --seed 51 --steps 2000 --preset voyager1 --out replays/day51_v1.jsonl
```
**Output**
```
[record] wrote 2000 steps -> .\replays\day51_v1.jsonl; return=-24302.721296; target_radius=7.5e+12
```

### 2) Replay & drift check (strict thresholds)
```powershell
python scripts/play_replay.py --seed 51 --preset voyager1 --replay replays/day51_v1.jsonl --pos_scale 7.5e12 --vel_scale 3.0e4 --strict
```
**Metrics**
```
pos_l2 = 5.6096e-09    (≤ 5e-4)
vel_l2 = 4.5573e-07    (≤ 5e-4)
ret_abs = 0.0          (≤ 5e-4)
Status: PASS
```

### 3) Quickrun (strict)
```powershell
python scripts/quickrun.py --seed 51 --steps 1200 --preset voyager1 --strict
```
**Metrics**
```
wall_time_s ≈ 0.1075
steps_run = 1200  → steps_per_sec ≈ 11162.58   (≥ 400)
ep_return ≈ -15003.1185
Status: PASS
```

### 4) Pytest smoke
Initial run showed one failure in `Tests/test_env_smoke.py` due to `MultiOrbitEnv(cfg)` accepting a `SimConfig` object.  
Applied patch to `envs/multi_orbit_env.py` (support config objects/dicts).  
**Post-patch:** smoke tests are green locally.

---

## 📁 Artifacts
- `replays/day51_v1.jsonl` — reference trajectory (JSONL, seed=51).
- Console logs for replay/quickrun (stored in CI logs when integrated).

---

## 🔧 Notes & Decisions
- **ExpertController interface** is enforced as `__call__(t, pos, vel)`; adapter builds `pos, vel` from `obs`.
- **`target_radius` discovery**: prefer `info/env` fields; fallback to preset (`voyager1 → 7.5e12`); final value printed in logs.
- **Performance**: quickrun far exceeds thresholds on local machine; thresholds set for CI robustness.
- Directory creation: created `replays/` explicitly today; future improvement is to let `write_jsonl` auto-create the parent dir.

---

## 📌 Commit Messages
- `Stabilize replay + quickrun pipeline for fast validation (Day51)`
- *(or split)*
  - `feat(replay): deterministic record & replay with auto target_radius`
  - `chore(smoke): quickrun using ExpertController(t,pos,vel) + perf thresholds`
  - `fix(env): MultiOrbitEnv accepts SimConfig/dict and merges overrides`

---

## ➡️ Next (Day 52 suggestions)
- Add a **rolling hash** over `(obs, action)` per step to detect silent drift in CI.
- Emit a compact `artifacts/metrics.json` for replay & quickrun and surface in CI summary.
- Run **Expert vs IL vs PPO** on the **same replayed action sequence** for apples-to-apples return/robustness comparisons.
- Optional: switch `t` to physical time if `info['time']`/`info['dt']` is available; keep step-index fallback.
