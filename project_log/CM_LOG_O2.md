# ProjectLog — CM_DAY2

## What I did today (CM Day 2)

* Ran `src/quick_compare_v3_v4.py` to compare **ExpertV3** (`controller/expert_controller.py`) vs **ExpertImproved** (`controller/expert_controller_improved.py`).
* Verified the **action interface mode switch** via:

  * `ACTION_IF_MODE=raw`
  * `ACTION_IF_MODE=prescale`
* Tested multiple scenarios:

  * `weak_thrust_far` (with `thrust_newton=800`)
  * `oscillation_noise` (warning: not supported by current env)
  * `misaligned_entry`
  * `default`

## Key observations

* **Mode switch works** (the run was executed under both `raw` and `prescale`), so the config path is not a no-op.
* **ExpertV3 vs ExpertImproved produced nearly identical metrics** on the tested scenarios:

  * Final radius stayed around `9.375e12` with target `7.5e12`.
  * Average radius error remained ~`1.875e12` across runs.
  * Total reward and jitter were very close.
* `oscillation_noise` scenario triggered:

  * `[WARN] 'oscillation_noise' not supported by OrbitEnv (no noise API). Using default.`
  * Meaning: this scenario currently behaves like the default setting and does **not** inject noise.

## Interpretation (so I don’t fool myself)

* Today’s result is still valuable: it establishes a **baseline** showing that my “Improved” expert currently **does not diverge** from V3 in these cases.
* Next step is **not** “rewrite the expert again,” but to:

  1. create scenarios that actually stress the differences, and/or
  2. implement a minimal noise API / perturbation hook in OrbitEnv so `oscillation_noise` is real.

## Artifacts

* Command(s) run:

  * `python src/quick_compare_v3_v4.py`
  * `ACTION_IF_MODE=raw`
  * `ACTION_IF_MODE=prescale`

## Summary

Baseline established: action-mode switch verified; ExpertImproved currently matches ExpertV3 on tested scenarios; env lacks a noise hook for `oscillation_noise`.
