# Project Log — Day 33 (2025-08-17)

**Scope:** Align the replay pipeline with `eval_battery.py`, fix observation denormalization and thrust-to-acceleration mapping, and produce side-by-side Agent vs. Expert diagnostics (CSV + plots + console summary).

---

## Objectives
- Feed *physical* state `(r, v)` into the Expert (auto de-normalize from `(mu, rt)` when needed).
- Enforce action shaping around the Expert:
  - hard acceleration clamp `||a|| ≤ a_cap`,
  - firing gate (`||a|| < fire_frac * a_cap → 0`),
  - coast-in-band with hysteresis.
- Make thrust mapping consistent: `a_raw = thrust(N) / mass(kg)`, then clamp to `[a_lo, a_hi]`.
- Emit a comparison CSV and three bar charts; print deltas in console.

---

## Changes Made
- **New `ExpertAdapter`** (English-only comments):
  - Extracts physical `(r, v)` from the env (or de-normalizes from obs via `(mu, rt)`).
  - Hysteresis “coast window” (configurable `band_in`, `band_out`).
  - **Hard clamp** on action magnitude.
  - **Firing gate** to suppress micro-burns.
  - Tracks `thrust_on_frac = on_steps / total_steps` for diagnostics.
- **Thrust-to-accel mapping:** `a_raw = thrust / mass`; default `a_lo = 0.0` (prevents tiny-thrust tasks from being floored up).
- **CLI switches:** `--a_lo/--a_hi/--fire_frac/--band_in/--band_out/--radial_gain/--tangential_gain/--damping_gain/--bang_bang`.

---

## Run Configuration (final run of the day)
- `a_lo=0.0, a_hi=0.45, fire_frac=0.45`
- `band_in=1.3, band_out=1.6`
- `bang_bang=False`

**Repro command:**

---

## Key Console Evidence
- Denormalization is active: `"[AB] denorm via mu,rt; v_ref=..."`
- Mapping is correct: for all tasks, `a_raw == a_cap` and `(floor_hit=False, ceil_hit=False)`.
- Firing ratios:
  - `fixed_3`: `thrust_on_frac=1.000 (30000/30000)` — continuous tiny thrust (very low `a_cap`).
  - `random_15`: `thrust_on_frac=0.002 (52/30000)`.
  - `random_7`: `thrust_on_frac=0.106 (175/1655)`.

---

## Results (Expert − Agent)

| name       | agent_success | expert_success | Δret      | Δr_err  | Δfuel     |
|------------|----------------|----------------|-----------|---------|-----------|
| fixed_3    | 1              | 0              | −2.69e+03 | 0.273   | −2.39e+06 |
| random_15  | 0              | 0              | −2.36e+03 | −0.0238 | 7.94e+06  |
| random_7   | 1              | 1              | −1.77e+03 | 0.00143 | 8.65e+06  |

**Figures (saved by the script):**
- `ab/day33/replay/figs/ab_fuel.png`
- `ab/day33/replay/figs/ab_r_err.png`
- `ab/day33/replay/figs/ab_ret.png`

![Δfuel](ab/day33/replay/figs/ab_fuel.png)
![Δr_err](ab/day33/replay/figs/ab_r_err.png)
![Δret](ab/day33/replay/figs/ab_ret.png)

---

## Interpretation
- **Unit and observation issues are resolved.** The Expert now acts on physical state; thrust is correctly mapped to acceleration without unintended floors/ceilings.
- **`fixed_3`** (ultra-low thrust, `a_cap ≈ 5.56e−4 m/s²`): expected continuous light thrust. Importantly, **Δfuel is negative** (the Expert now uses less fuel than the Agent), confirming the earlier fuel spike was a tooling artifact.
- **`random_15` / `random_7`**: very low firing ratios, but **positive Δfuel** persists, which indicates the environment likely integrates *thrust magnitude* (not just “on/off”). Fewer burns help, yet large per-burn magnitude still raises total fuel.

---

## Artifacts
- CSV: `ab/day33/replay/replay_compare.csv`
- Plots: `ab/day33/replay/figs/ab_ret.png`, `ab/day33/replay/figs/ab_r_err.png`, `ab/day33/replay/figs/ab_fuel.png`
- Script: `script/replay_worst.py` (English comments)

---
