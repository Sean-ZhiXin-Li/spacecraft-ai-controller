# Phase Controller Dataset

## Outputs

- `E:/spacecraft_ai_project/analysis/phase_controller_dataset/phase_controller_dataset.npz`
- `E:/spacecraft_ai_project/analysis/phase_controller_dataset/phase_controller_dataset_metadata.json`

## Dataset Contents

- observation count: `265527`
- action count: `265527`
- setup count: `5`
- phase counts: `{'DESCENT': 264405, 'CAPTURE': 101, 'LOCK': 1021}`

## Included Setups

- setup `0`: `r0=1.00002`, `dt=50`, `thrust=10000`, `steps=75877`
- setup `1`: `r0=1.00002`, `dt=50`, `thrust=20000`, `steps=29860`
- setup `2`: `r0=1.00002`, `dt=100`, `thrust=10000`, `steps=15072`
- setup `3`: `r0=1.00005`, `dt=50`, `thrust=20000`, `steps=96260`
- setup `4`: `r0=1.00005`, `dt=100`, `thrust=10000`, `steps=48458`

## Purpose

- This dataset is intended for behavior cloning of the explicit phase controller before PPO fine-tuning.
- Phase labels are included so cloning and phase-conditioned reward shaping can share the same supervision source.