# Day 52 — Stabilize Replay & Quickrun + Batch-run elliptic_strong Family

## Theme
Stabilize replay + quickrun pipeline for fast validation

## Objectives
- **Replay stability**: Ensure baseline replays can be deterministically reproduced, checking L2/L∞ drift metrics.  
- **Quick validation**: Create a lightweight quickrun pipeline that can fail fast when stability degrades.  
- **Coverage**: Batch-run the `elliptic_strong` family and log stability results in CSV format.  

## Progress
- Implemented a **PowerShell-friendly version** of `tools/replay_recorder.py`, adding `--scenario` and `--extra-kv` flags to avoid JSON parsing issues.  
- Added a new **`replay_player.py`** with consistent CLI design, supporting replay drift validation on positional and velocity slices.  
- Verified recording + replay workflow:  
  1. Record baseline trajectory (`replay_recorder.py`).  
  2. Replay deterministically (`replay_player.py`) and check drift metrics.  
  3. Prepare for batch execution across `elliptic_strong` variants.  
- Confirmed log output path structure under `logs/day52/elliptic_strong/`.  

## Deliverables
- `tools/replay_recorder.py` — extended recorder with scenario/extra-kv support.  
- `tools/replay_player.py` — deterministic replay with drift metrics and strict thresholds.  
- Baseline outputs stored under `logs/day52/elliptic_strong/`.  

## Next Steps
- Implement `scripts/day52_batch.py` to automate elliptic_strong family runs.  
- Add `scripts/quickrun.py` with a smoke-test suite for CI validation.  
- Aggregate results into `logs/day52/elliptic_strong/stability.csv`.  
