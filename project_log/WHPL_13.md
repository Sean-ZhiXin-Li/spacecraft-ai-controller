# WHPL_13 — Variant Tracking (always_on vs gated) @ 200N Hard

**Date:** 2026-02-19
**Repo:** `spacecraft_ai_project`
**Branch:** (local run context)
**Primary script:** `src/quick_compare_v3_v4.py`
**Controller:** `controller/expert_controller_improved.py` (`ExpertController`)
**Scenario:** `weak_thrust_far`
**Goal:** Track and compare **controller variants** (`always_on` vs `gated`) under identical physics + difficulty, and write **append-only** ablation rows with `dedup_key`.

---

## 0) Freeze Rules (non-negotiable)

* **No new physics.**
* **No reward edits.**
* **No new controller class.** Only parameter/variant toggles through `CONTROLLER_VARIANT`.
* **One-factor change:** controller variant only (`always_on` vs `gated`).
* **Append-only logging:** results go to `analysis/results/ablation_thrust_x_difficulty.csv`.

---

## 1) Experiment Design

### Independent variables

* `THRUST_NEWTON = 200`
* `DIFFICULTY_TAG = Hard`
* `CONTROLLER_VARIANT ∈ { always_on, gated }`
* `SEED ∈ {0, 1, 2}` (attempted)

### Controlled variables

* Same scenario (`weak_thrust_far`)
* Same target radius (`target_r = 7.5e12`)
* Same controller class + file path
* Same run length (`steps=2000`)

### Outputs tracked

* `total_reward`
* `avg_radius_error`, `final_radius_error`
* `saturation_rate_mean`
* `thrust_intent_norm_mean`, `action_norm_mean`
* `avg_jitter` (direction variance)
* `dedup_key` + `csv_status (append/skip)`

---

## 2) Commands Run

### Baseline gated (single-run evidence)

```powershell
$env:THRUST_NEWTON="200"
$env:DIFFICULTY_TAG="Hard"
python src/quick_compare_v3_v4.py
```

### Variant comparison at SEED=0

```powershell
$env:THRUST_NEWTON="200"
$env:DIFFICULTY_TAG="Hard"
$env:SEED="0"

$env:CONTROLLER_VARIANT="always_on"
python src/quick_compare_v3_v4.py

$env:CONTROLLER_VARIANT="gated"
python src/quick_compare_v3_v4.py
```

### Attempted multi-seed replication (always_on)

```powershell
$env:CONTROLLER_VARIANT="always_on"
$env:SEED="1"; python src/quick_compare_v3_v4.py
$env:SEED="2"; python src/quick_compare_v3_v4.py
```

### Attempted multi-seed replication (gated)

```powershell
$env:CONTROLLER_VARIANT="gated"
$env:SEED="1"; python src/quick_compare_v3_v4.py
$env:SEED="2"; python src/quick_compare_v3_v4.py
```

---

## 3) Code Inspection Evidence

### Controller has explicit gating + D-term scaling hooks

The following signals are produced and printed by `whpl09_debug`:

* `rel` (relative radial error)
* `g_r` (error-band gate)
* `d_scale` (D-term sign gate)
* `thrust_r_pd` (PD injection)

Confirmed via grep:

* `g_r = clip((rel - r_on)/(r_full - r_on))`
* `d_scale = 0.3 when radial_error>0 and radial_velocity<0`
* `thrust_r_pd = g_r * clip(p_term + d_term) * cap`

---

## 4) Results

### 4.1 SEED=0: always_on

Key summary (final line):

* `total_reward = -1.9846e+04` (≈ **-19846.3231**)
* `avg_radius_error ≈ 1.875e+12`
* `final_r ≈ 9.375e+12` → `final_radius_error ≈ 1.875e+12`
* `saturation_rate_mean = 0.33475`
* `avg_jitter ≈ 1.451e-09`

CSV row written (append):

* `controller_variant = always_on`
* `dedup_key = 40556d24`
* `csv_status = append`

### 4.2 SEED=0: gated

Key summary (final line):

* `total_reward = -2.0260e+04` (≈ **-20260.1816**)
* `avg_radius_error ≈ 1.875e+12`
* `final_r ≈ 9.375e+12` → `final_radius_error ≈ 1.875e+12`
* `saturation_rate_mean = 0.096`
* `avg_jitter ≈ 1.604e-09`

CSV row written (append):

* `controller_variant = gated`
* `dedup_key = 2e225f00`
* `csv_status = append`

---

## 5) Interpretation (what the data says)

### 5.1 Variant effect is real and large (on saturation)

* `always_on` produces **much higher saturation** (0.33475) than `gated` (0.096).
* Reward difference is modest relative to the scale, but consistent:

  * Δreward ≈ **(+413.86)** in favor of `always_on` (less negative).

### 5.2 Radius error is unchanged in this regime

Both variants show:

* `avg_radius_error ≈ 1.875e+12`
* `final_radius_error ≈ 1.875e+12`

This indicates **200N + Hard + weak_thrust_far** is still outside the reachable control regime: the controller shapes thrust and saturation, but cannot reduce radius error meaningfully in 2000 steps.

### 5.3 Mechanistic explanation (consistent with logs)

* `always_on`: PD injection is effectively always contributing; it increases thrust intent magnitude and drives more frequent clipping/saturation.
* `gated`: error-band gate + D-term sign gate reduce PD injection, lowering saturation substantially.

---

## 6) Multi-seed Attempt: What happened

### Observation

Runs with `SEED=1` and `SEED=2` produced **identical trajectories and identical summary statistics** for each variant, and CSV dedup logic returned:

* `csv_status = skip`

### Working hypothesis

**SEED is not currently wired into the environment reset / RNG path**, or the scenario is deterministic so seed has no effect.

### Consequence

* Multi-seed robustness cannot be claimed yet.
* The current evidence is still valuable: the **variant effect is structural** under a deterministic pipeline.

---

## 7) Deliverables Produced

* `analysis/results/ablation_thrust_x_difficulty.csv` appended two new rows:

  * (`always_on`, 200N, Hard)
  * (`gated`, 200N, Hard)
* `analysis/SESSION6_metrics_upgrade.json` written (self-check metrics dump).

---

## 8) Next Actions (thin-but-hard, no scope creep)

1. **Wire SEED into env.reset** and print `seed_effective` in `[SELF-CHECK]`.

   * Target: different seeds must produce detectable differences (if env has any stochasticity).
2. Keep the same test point (200N, Hard) and re-run `SEED ∈ {0,1,2}` to validate.
3. If the environment is intentionally deterministic, explicitly document:

   * `SEED has no effect under weak_thrust_far`.

---
