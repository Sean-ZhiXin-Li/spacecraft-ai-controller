# Day 31 — Project Log (Stage B)

## Goal
Push **SR ≥ 0.50** under stricter Stage B gates while holding or reducing fuel use (primary metric: `fuel_succ(median)`).

## Stage B Setup
- **Success gates:** `rerr_thr=0.015`, `verr_thr=0.030`, `align_thr=0.97`, `stable_steps=160`
- **Time scale:** `dt=6000 s`, `max_steps=30000` (~5.7 years/episode)
- **Authority:** `thrust_scale_range=(100, 150)`
- **Eval battery:** 5 fixed tasks + 20 random tasks
- **Eval plumbing:** after each `env.reset(...)`, pass task to controller via `set_task(...)`; rollout uses the initial observation (no double reset).
- **Output:** `results/battery_day31.csv`

## Controller Updates (v1.4 → v1.4b)
- Keep v1.3 features: strong near-target fuel saving, direction-safety guard, hysteresis deadzone, proximity-based throttle.
- **Far & hard relaxation (v1.4):** if `r_norm > 1.10` **and** (`target_radius > 4e12` or `e > 0.25`), mildly relax throttle caps.
- **Tangential clamp relax (v1.4b):** in the same far&hard region, relax `t_clip` to speed up velocity convergence.
- **Direction-safety gating:** apply only **near target** or when the task is **not hard**; allow tangential corrections in far&hard cases.
- **Deadzone alignment:** tighten deadzone to sit just above Stage B gates to avoid “coasting outside the band.”

## Final Baseline Hyperparameters (v1.4b-trim)
- Gains: `k_e=0.9`, `k_rp=0.10`, `k_rd=0.60`
- Tangential clamp: `t_clip=0.41` (far&hard: `×1.12`)
- Proximity caps: `a_max_lo=0.048`, `a_max_hi=0.43` (far&hard caps: `×(1.08, 1.16)`)
- Deadzone: `dead_r_in=0.020`, `dead_r_out=0.017`, `dead_v_in=0.040`, `dead_v_out=0.033`
- Desired v_t limits: `v_des_min=0.82`, `v_des_max=1.19`
- Action space: clipped to `[-1, 1]^2` (Cartesian)

## Results (key checkpoints)

| Configuration                           | SR   | fuel_succ(median) | Notes                                   |
|----------------------------------------|:----:|-------------------:|-----------------------------------------|
| v1.4 initial (Stage B)                 | 0.44 |          2.43e6    | Tight gates exposed under-correction     |
| v1.4b + dir-safety gating              | 0.48 |              ↓     | First lift on hard far-eccentric tasks   |
| + Deadzone aligned to Stage B          | 0.52 |        **2.14e6**  | Big gain; fewer “coast-outside” episodes |
| **Final v1.4b-trim (current baseline)**| **0.56** | **2.08e6**     | Stable buffer above target + lower fuel  |

Additional aggregates for the final baseline:
- `fuel(all) ≈ 1.65e6`, `ret ≈ 1494`, `r_err ≈ 2.80e-2`, `v_err ≈ 3.17e-2`, `align ≈ 0.99`
- Episode endings: `max_steps ≈ 44%`, `violations = 0%`

## Hard-Case Profile (recurring)
High eccentricity and mid-to-far targets remain the toughest:
- `random_11`: `rt=3.69e12`, `e=0.32`, `mass=1000`, `thrust≈139.5`
- `random_6`:  `rt=4.90e12`, `e=0.29`, `mass=1000`, `thrust≈101.9`
- `random_1`:  `rt=2.76e12`, `e=0.35`, `mass=720`,  `thrust≈125.1`

## What Worked
- Tightening the deadzone to sit just above Stage B gates (reduced premature coasting).
- Enabling tangential authority only where it matters (far&hard), while keeping near-target fuel logic intact.
- Slightly stronger radial P/D to settle within `stable_steps=160`.

## Deferred (not executed today)
- **Expert upper bound** controller as a high-SR reference curve.
- **Auto-export “worst 3 tasks”** JSON for targeted A/B and curriculum building.

## Artifact
- CSV: `results/battery_day31.csv` (Day 31 archive)
