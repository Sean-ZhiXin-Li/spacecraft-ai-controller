# 🚀 AI-Controlled Spacecraft Orbital Simulator

An open-source project exploring **AI-driven thrust control for orbital dynamics**.  
The system integrates a custom 2-D orbital environment, multiple controllers (expert, imitation, PPO),  
and a baseline benchmarking pipeline designed for long-term reproducibility and research.

---

## 📌 Project Overview
- **Goal:** Develop and benchmark AI controllers for spacecraft thrust control under realistic orbital dynamics.  
- **Features:**
  - Custom `OrbitEnv` (Gymnasium-compatible) with solar-scale orbits.
  - Multiple controllers: heuristic, expert, imitation (MLP), PPO.
  - Baseline evaluation harness with fuel-aware metrics and reproducibility.
  - Detailed project logbook (Day 1 → Day 39).

---

## 🏗️ Repository Structure

```text
spacecraft_ai_project/
│
├── simulator/         # Orbit physics environment (OrbitEnv, integrators)
├── controller/        # Expert, imitation, PPO controllers
├── data/              # Expert datasets (.npy, .csv)
├── ppo_orbit/         # PPO agent code
├── tools/             # Quickrun, summarizer, plotting utilities
├── project_log/       # Daily project logs (Day1–Day39)
├── ab/                # Experiment results (task specs, csv, figs)
├── results/           # Final CSV summaries and baseline results
├── LICENSE            # License file
└── README.md
```

---

## ⚙️ Installation

### Requirements
- Python 3.10+
- Recommended: create a new virtual environment (`venv` or `conda`)

### Install dependencies
```bash
pip install numpy matplotlib torch scikit-learn gymnasium
```

(Modules like `os`, `json`, `glob`, `random`, `dataclasses`, and `typing` are part of the Python standard library.)

---

## 🚀 Quick Start

Run a baseline evaluation with expert controllers:

```bash
python tools/day39_quickrun.py \
  --tasks_dir ab/day36/task_specs_fast \
  --out_dir ab/day39 \
  --controllers elliptic_strong transfer_2phase spiral_in \
  --limit 64
```

**Outputs:**
- CSV: `ab/day39/csv/summary.csv`
- Figures:
  - `ab/day39/figs/day39_r_err_by_controller.png`
  - `ab/day39/figs/day39_return_by_controller.png`

---

## 📊 Results & Figures

### Example Outputs
```markdown
![Baseline Error Comparison](ab/day39/figs/day39_r_err_by_controller.png)
![Baseline Return Comparison](ab/day39/figs/day39_return_by_controller.png)
![Training Curve](plots/training_curve.png)
![Example Trajectory](plots/example_trajectory.png)
```

(Replace paths with actual available figures in your repo.)

---

## 📖 Project History

This project has been developed over a 40-day logbook, gradually evolving from basic orbit simulation to complex multi-task controllers with expert and baseline comparisons. Key milestones:

### Phase 1 – Foundations (Day 1–10)
- Built the first **2-D Orbit Simulator (OrbitEnv)** with gravitational physics.
- Implemented early **Expert Controllers** (radial/tangential thrust logic).
- Collected expert datasets (`expert_dataset_*.npy`) and trained first **Imitation Models** (MLPRegressor).
- Initial closed-loop imitation tests showed thrust prediction accuracy but unstable orbit capture.

### Phase 2 – Imitation & PPO Experiments (Day 11–19)
- **V4–V5 Imitation Controllers** trained on ~30 expert trajectories; still failed long-horizon orbit capture.
- First **PPO runs** with reward shaping attempted (Day 12–14) but collapsed to empty or unstable trajectories.
- Pivot back to interpretable **Expert v3.1** → first **stable circular orbit** at Voyager-scale (Day 16).
- **V5 Imitation Long Run** escaped orbit; **V6 prep** exposed memory/scale issues → switched to smaller networks and better dataset hygiene.

### Phase 3 – Hybrid & PPO Refinement (Day 20–29)
- Fixed environment/tooling issues (IDE imports, dataset generation).
- Trained **V6.1 Imitation (PyTorch)**, closed-loop tested → straight-line escape, highlighting IL limits.
- Introduced **Hybrid idea**: load imitation/expert into PPO.
- Designed a **smooth reward function** (radius error, velocity error, angular misalignment, fuel penalty, Gaussian bonus).
- PPO training stabilized with KL-adaptive updates, expert warm-starts, and entropy scheduling, but rewards plateaued at sub-optimal levels.
- Prepared to extend to **multi-orbit curriculum**.

### Phase 4 – Baseline Phase A/B (Day 30–35)
- Introduced **fuel-aware GreedyEnergyRT baseline**, aligning timescale with solar-scale orbits.
- **Stage A (Day 30):** SR=0.64, median fuel ≈1.8e6.  
- **Stage B (Day 31):** stricter gates (rerr_thr=0.015, verr_thr=0.030, align_thr=0.97) → SR=0.56 with lower fuel.
- Added **Expert Upper Bound** comparison (Day 32–34) with replay pipelines (`script/replay_worst.py`).
- Day 35: baseline phase consolidated → Zero (lower bound), Greedy (mid-baseline), Expert (upper bound).

### Phase 5 – Complex Environment & Quickruns (Day 36–39)
- Generated **fast task bundles** (circular, elliptic, transfer).
- Verified expert families (`elliptic_strong`, `transfer_2phase`, `spiral_in`) on large radii:
  - Circular: consistently solved.
  - Elliptic/transfer: failure cases remain (eccentricity damping & phase strategies needed).
- Built automated **summary, replay, and quickrun pipelines** (Day 38–39) for rapid validation:
  - `elliptic_strong`: most stable, ~100% SR.
  - `transfer_2phase`: accurate but ~85–90% SR.
  - `spiral_in`: weaker, ~70–80% SR.

---

## 🔮 Next Steps
- Hybrid **Imitation + PPO** initialization for faster RL convergence.
- Curriculum training with multi-orbit and higher eccentricity tasks.
- Robustness experiments under fuel faults and attitude noise.
- Extended evaluation pipeline (agg stats, hardest-task curriculum).

---

## 📜 License
This project is licensed under the terms of the [MIT License](LICENSE).
