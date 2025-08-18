# Project Log — Day 34

**Goal.** Reduce fuel on worst tasks via CLI-only tuning (no code changes), with focus on `random_15`.

**Best Sets (Δfuel = Agent − Expert; lower is better).**
- **lower_min** — `a_hi=1.7e-4`, `fire_frac=0.16`, `band=(1.18, 2.12)`  
  - **Total Δfuel = -2.600e6** (best)
  - `random_15 ≈ +1.98e6`, `random_7 ≈ -2.16e6`
- **lower_min_pulse15** — same bands/cap, `fire_frac=0.15`  
  - **Total Δfuel = -2.600e6** (ties best)
- **lower_min_tight** — `band_out=2.10`  
  - **Total Δfuel = -2.586e6`

**Trend.**
- Lower thrust cap + moderate duty (0.16) consistently reduced fuel without harming success.
- Slightly tighter band_out (2.10) began to shave margin with minimal gain; 2.12 is a good sweet spot.
- Shorter pulses (0.15) can match best fuel, but 0.16 is a safer default for robustness.

**Decision.**
- **Adopt `lower_min` as the Day 34 baseline**: `a_hi=1.7e-4`, `fire_frac=0.16`, `band=(1.18, 2.12)`.
- Keep **pulse15** as an optional toggle for A/B checks; revert to 0.16 if tracking error grows.

**Artifacts.**
- Per-set results in `ab/day34/day34_<label>/replay/...`
- Combined summary: `ab/day34/summary_day34_all.csv` (TOTAL row per set)
