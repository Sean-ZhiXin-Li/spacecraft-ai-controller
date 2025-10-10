# Day 54 — Spiral-In Smoke Test & Weak-Point Scan

**Date:** 2025-10-10  
**Scope:** Minimal smoke runs for the `spiral_in` family to validate the pipeline and surface weak points.  
**Link-back:** Continues from Day 53 (`transfer_2phase`).

---

## 1) Objective
- Run a smoke test (`steps=200`, seed=54) for `spiral_in`.
- Confirm artifacts (`replay.npz`, `meta.json`) and array shapes are sane.
- Focus on identifying weak points, not solving them.

---

## 2) Commands
Executed PowerShell script:

```powershell
$seed   = 54
$replay = "logs/day54/spiral_in/smoke"
$recOut = Join-Path $replay "replay_smoke.jsonl"

New-Item -ItemType Directory -Force -Path $replay | Out-Null

& python -X dev -u @(
  "tools/replay_recorder.py",
  "--env-factory","envs.multi_orbit_env:MultiOrbitEnv",
  "--policy","controller.expert_controller:policy",
  "--scenario","spiral_in",
  "--seed",$seed,
  "--steps",200,
  "--out",$replay,
  "--extra-kv","family=spiral_in"
) *>&1 | Tee-Object -FilePath $recOut
```

Artifacts generated:  
- `logs/day54/spiral_in/smoke/replay.npz`  
- `logs/day54/spiral_in/smoke/meta.json`  

---

## 3) Sanity Check Output
Python check (`tools/quickcheck_day54.py`) results:

```
keys: ['obs', 'actions', 'rewards', 'dones']
obs.shape: (201, 4)
obs[-1]: [-1.80991188e+06  9.37499505e+12 -4.74039795e+03 -6.43018115e+03]

meta.json:
{
  "steps_recorded": 200,
  "seed": 54,
  "env_factory": "envs.multi_orbit_env:MultiOrbitEnv",
  "policy": "controller.expert_controller:policy"
}
```

Observations:
- Shapes are consistent with steps (201 obs entries for 200 steps).  
- Final state values are finite, no NaN/Inf.  
- Meta fields correctly reflect run parameters.  

---

## 4) Findings — Weak Points
(To be filled after trajectory/reward inspection)

- Convergence speed: ...  
- Trajectory oscillation: ...  
- Thrust scaling: ...  
- Entry precision: ...  
- Numerical stability: ...  
- Logging gaps: ...  

