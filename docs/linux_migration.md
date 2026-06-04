# Linux Migration Guide

This repository is being prepared for a Linux-first workflow. The current
research results remain simulator-scoped: this guide changes environment and
execution setup only, not physics, controllers, benchmark rules, or result
interpretation.

## Environment Strategy

Use the Linux CPU baseline first:

```bash
git clone https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller.git
cd spacecraft-ai-controller
conda env create -f conda_envs/spacecraft_linux.yml
conda activate spacecraft_linux
```

The CPU baseline is intentional. Validate imports and deterministic smoke
checks before adding a machine-specific CUDA installation.

The files `conda_envs/spacecraft.yml` and `conda_envs/orbittools.yml` are
Windows environment snapshots. Keep them as provenance, but do not use them to
create the Linux environment because they contain Windows packages and local
prefixes.

## Smoke Test Checklist

Run from the repository root:

```bash
export MPLBACKEND=Agg
python -m pytest -q Tests/test_env_smoke.py Tests/test_quickrun_smoke.py
python scripts/quickrun.py --steps 1200 --preset voyager1
python -m py_compile train/preprocess_merge_dataset.py
python -c "from pathlib import Path; files=sorted((Path('data') / 'dataset').glob('expert_dataset_*.npy')); print(f'expert_dataset_count={len(files)}')"
```

Expected outcomes:

- Pytest exits with status `0`.
- `scripts/quickrun.py` prints a `[quickrun]` metrics dictionary and exits with
  status `0`.
- Compilation exits with status `0`.
- Canonical dataset discovery prints `expert_dataset_count=30`.

These expected outcomes must be verified on the target Linux machine. They are
not claimed as Linux-tested until that run is complete.

Optional dependency smoke test:

```bash
python analysis/smoke_full_system.py
```

This checks SPICE, CasADi/IPOPT, OSQP, do-mpc, and the orbit initialization
fixture. Treat a CasADi/IPOPT failure as an environment issue to diagnose, not
as permission to change Phase32 conclusions.

## Main Entry Points

Current demo:

```bash
python main.py
```

Phase34 post-cross synchronization benchmark:

```bash
python scripts/explicit_controller_phase34_post_cross_sync.py
```

Phase36B transfer-family benchmark:

```bash
python scripts/explicit_controller_phase36b_transfer_family_benchmark.py
```

Phase36C non-crossing diagnosis:

```bash
python scripts/phase36c_non_crossing_geometry_diagnosis.py
```

Historical PPO entry point:

```bash
python ppo_orbit/ppo.py
```

Do not start with PPO training during migration validation. Confirm the smoke
tests and explicit-controller imports first.

## Known Migration Risks

Must verify on Linux:

- CasADi can load its IPOPT plugin in the created environment.
- SPICE imports successfully through `spiceypy`.
- Headless Matplotlib runs with `MPLBACKEND=Agg`.
- The canonical expert dataset tree is available at `data/dataset/`.
- Any optional GPU installation preserves the CPU-baseline behavior.

Known historical issues that are not migration blockers:

- `train/preprocess_merge_dataset.py` keeps its derived output under
  `data/data/preprocessed/` to avoid downstream churn.
- `scripts/inspect_dataset.py` and `script/inspect_dataset.py` reference an old
  preprocessing filename and should be reviewed separately.
- Historical analysis files may contain Windows absolute paths as recorded
  provenance. Do not rewrite those outputs during migration.

## Maintaining the Environment File

Do not replace the Linux manifest with a platform-locked export. When the
Linux environment changes intentionally, prefer:

```bash
conda env export --from-history -n spacecraft_linux > conda_envs/spacecraft_linux.yml
```

Then review the file manually so that required pip-only packages and comments
remain documented.
