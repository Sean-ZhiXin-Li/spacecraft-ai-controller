# ProjectLog 32 (DAY32)

## Goals
- Add an **Expert upper-bound** reference curve alongside the agent for both **return (`ret`)** and **radial error (`r_err`)**.
- Automatically export the **worst 3 tasks** for later A/B replay and inspection.
- **Do not modify** the original `ExpertController`; fix issues only in the glue layer (unit/scale alignment).

---

## What Got Done Today

### 1) A/B Comparison & Plots
- Integrated `run_ab_compare` into `eval_battery.py`, producing two reference curves:
  - `ab/day32/plots/refcurve_ret.png`
  - `ab/day32/plots/refcurve_r_err.png`
- Exported per-task specs for the **worst 3** (by `ret`) to:
  - `ab/day32/worst/task_specs/*.json`

### 2) Worst-Task Replay Script
- Added `script/replay_worst.py`:
  - Replays Agent (`greedy_energy_rt`) vs **Expert** on the exported tasks.
  - Outputs:
    - CSV: `ab/day32/replay/replay_worst_ab.csv`
    - Figures: `ab/day32/replay/figs/ab_ret.png`, `ab_r_err.png`, `ab_fuel.png`
  - Prints per-task deltas: **Δret**, **Δr_err**, **Δfuel**.

### 3) Glue-Layer Only Fixes (Expert stays unchanged)
- Implemented **`ExpertAdapter`** to feed **physical** `(pos [m], vel [m/s])` to the Expert:
  - Prefer `env.get_raw_rv()` / `env.pos/env.vel`; otherwise **de-normalize** `obs[:4]` via `mu=G*M` and `rt=task["target_radius"]`.
  - Optional **“stop-in-band with hysteresis”** (based on `rerr_thr/verr_thr`) to save fuel, without touching Expert internals.
- Added a **thrust-scale mapping toggle** so the environment’s `thrust_limit` aligns with the Expert’s acceleration cap:
  - `EXPERT_THRUST_MAP="identity"` → passthrough (exactly your old feel).
  - `EXPERT_THRUST_MAP="linear"` → map env `thrust_limit ∈ [100,150]` to Expert accel cap `∈ [0.05,0.45]` (closer to the baseline’s scale, typically less fuel).  
  - Current default: **`linear`**.

---

## Current Results (Snapshot)

From today’s run of `eval_battery.py`:

| Controller        | SR    | ret    | r_err     | v_err     | align | fuel(all) |
|-------------------|-------|--------|-----------|-----------|-------|-----------|
| `zero`            | 0.000 | 790.3  | 1.220e-01 | 7.067e-02 | 0.995 | 0.0       |
| `greedy_energy_rt`| 0.560 | 1493.9 | 2.802e-02 | 3.168e-02 | 0.990 | 1,650,408.9 |

Observations:
- On **`r_err`**, the **Expert curve** now **beats or matches** the agent on hard tasks (left side) — stability looks good.
- On **`ret`**, the Expert is still **lower** mainly due to **higher fuel** (not control failure).

---

## Bugs/Issues Resolved

- **`KeyError: 'target_radius'`**  
  Unified task dictionaries so **both** env and expert receive the keys they expect (`target_radius` for env, `rt` for expert).

- **Signature & shape mismatch** (`__call__()` args, `(8,) vs (2,)`)  
  Fixed by using `ExpertAdapter` to extract `(x,y,vx,vy)` and **convert to physical units** before calling the Expert.

- **Incorrect units in replay**  
  Early replays fed normalized obs to the Expert, causing huge `Δr_err` and `Δfuel`. Syncing the adapter + thrust mapping between `eval_battery.py` and `script/replay_worst.py` solved this.

---

## Artifacts Produced

- Battery CSV: `results/battery_day31.csv`
- A/B reference plots:
  - `ab/day32/plots/refcurve_ret.png`
  - `ab/day32/plots/refcurve_r_err.png`
- Worst-task specs: `ab/day32/worst/task_specs/*.json`
- Worst-task replay:
  - `ab/day32/replay/replay_worst_ab.csv`
  - `ab/day32/replay/figs/ab_ret.png`
  - `ab/day32/replay/figs/ab_r_err.png`
  - `ab/day32/replay/figs/ab_fuel.png`

---

## Repro Commands

```bash
# 1) Evaluate full battery and generate upper-bound curves
python eval_battery.py

# 2) Replay the exported worst tasks (Agent vs Expert)
python script/replay_worst.py
