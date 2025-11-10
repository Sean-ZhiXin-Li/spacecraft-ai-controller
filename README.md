# AI-Controlled Spacecraft Orbital Simulator

An open-source project exploring **AI-driven thrust control for orbital dynamics**.
The system integrates a custom 2-D orbital environment, multiple controllers (expert, imitation, PPO),
and a baseline benchmarking pipeline designed for long-term reproducibility and research.

---

## Project Overview
- **Goal:** Develop and benchmark AI controllers for spacecraft thrust control under realistic orbital dynamics.
- **Features:**
  - Custom `OrbitEnv` (Gymnasium-compatible) with solar-scale orbits.
  - Multiple controllers: heuristic, expert, imitation (MLP), PPO.
  - Baseline evaluation harness with fuel-aware metrics and reproducibility.
  - Detailed project logbook (Day 1 → Day 39).

---

## Repository Structure

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

## Installation

### Requirements
- Python 3.10+
- Recommended: create a new virtual environment (`venv` or `conda`)

### Install dependencies
```bash
pip install numpy matplotlib torch scikit-learn gymnasium
```

(Modules like `os`, `json`, `glob`, `random`, `dataclasses`, and `typing` are part of the Python standard library.)

---

## Quick Start

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

## Results & Figures

### Example Outputs
```markdown
![Baseline Error Comparison](ab/day39/figs/day39_r_err_by_controller.png)
![Baseline Return Comparison](ab/day39/figs/day39_return_by_controller.png)
![Training Curve](plots/training_curve.png)
![Example Trajectory](plots/example_trajectory.png)
```

(Replace paths with actual available figures in your repo.)

---

## Project History

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

## Naming Convention (since NEW_WEEK_0)

- All research stages are named as `NEW_WEEK_X` (e.g., NEW_WEEK_0, NEW_WEEK_1, NEW_WEEK_2),
  representing distinct development phases of the spacecraft AI controller project.
- Each stage includes:
  - `/docs/NEW_WEEK_X_*.md` — formal report
  - `/analysis/NEW_WEEK_X_*.md` — mapping or supporting tables
  - `/logs/new_week_X/` — experimental data (may contain DayXX runs)
- "DayXX" is used only to tag experimental runs inside `/logs/` or `ProjectLog.md`.
  It does **not** represent a new research stage.

---

## 🚀 Next Steps

### Immediate Objectives (Phase 1–2)
- **Hybrid Imitation + PPO Initialization**
  Combine expert imitation pretraining with PPO fine-tuning to accelerate convergence and improve thrust decision stability.

- **Curriculum Training across Multi-Orbit Tasks**
  Gradually increase environment difficulty — from near-circular to highly eccentric orbits — to enhance controller generalization.

- **Robustness under Faults & Sensor Noise**
  Conduct Monte Carlo experiments introducing thrust degradation, random fuel loss, and attitude input noise to test fault tolerance.

- **Expanded Evaluation Pipeline**
  Add aggregated performance metrics (fuel efficiency, orbital error, convergence speed) and build a “hardest-task” benchmark suite.

---

### Planned Extensions (Phase 3–5)
- **3D Orbit Dynamics Integration**
  Extend the current 2D environment to full 3D orbital mechanics with pitch–yaw thrust control and realistic inclination parameters.

- **Multi-Agent Formation & Rendezvous Control**
  Implement coordinated control among multiple spacecraft for formation-keeping and autonomous docking tasks.

- **Advanced Propulsion Models**
  Introduce switchable thruster physics (chemical vs. electric), thrust decay models, and future extensions toward photon or fusion drives.

- **Reproducible Research Suite**
  Add modular logging (CSV/JSONL), trajectory visualization, and `metrics_core` integration for open research reproducibility.

---

## Project Vision / Long-Term Goal

> *“Trajectory may drift, but the mission continues.”*

This project aims to evolve into an open, physics-based simulation and control framework for **autonomous spacecraft propulsion and orbit transfer** — a platform that bridges modern AI decision-making with realistic astrodynamics.

### Long-Term Research Direction
- **AI × Propulsion Control**
  Develop a generalizable controller capable of reasoning about multi-mode thrusters (chemical, electric, photonic) and optimizing long-duration orbit transfers under uncertainty.

- **From Single Spacecraft to Distributed Systems**
  Scale the current architecture toward *multi-satellite formation flying* and *autonomous rendezvous* scenarios, inspired by the Stanford **Space Rendezvous Laboratory (SLAB)** Distributed Space Systems project.

- **Toward Real-World Autonomy**
  Integrate robust fault tolerance, dynamic resource management, and reinforcement/imitation hybrid learning to enable spacecraft to adapt, survive, and complete missions beyond direct Earth contact.

- **Educational & Open Science Goals**
  Serve as an accessible testbed for students and researchers exploring astrodynamics, guidance and control, and reinforcement learning — emphasizing transparency, reproducibility, and modular design.

### Ultimate Objective
To contribute a reproducible foundation for **AI-driven space exploration**, where intelligent control systems can autonomously navigate complex gravitational environments, manage propulsion resources efficiently, and maintain stable trajectories even in the face of uncertainty.

---

## References & Inspirations

This project integrates insights from **global research institutions and aerospace innovators**, bridging traditional orbital mechanics with modern intelligent control.

### Stanford University
- **Space Rendezvous Laboratory (SLAB)**
  Research on distributed space systems, formation flying, and autonomous rendezvous control.
  → [https://slab.stanford.edu](https://slab.stanford.edu)
- **CAESAR – Center for AEroSpace Autonomy Research**
  Exploring AI-enhanced spacecraft autonomy, landing, and decision-making.
  → [https://caesar.stanford.edu](https://caesar.stanford.edu)

### ESA (European Space Agency)
- **Advanced Concepts Team (ACT)** — *Artificial Intelligence for Guidance, Navigation, and Control (GNC)*
  Proposed AI-based onboard autonomy and evolutionary trajectory optimization.
  *Izzo, D. et al., “A Survey on Artificial Intelligence Trends in Spacecraft Guidance Dynamics and Control,” arXiv, 2018.*
  → Inspired this project’s design philosophy of **onboard autonomy and intelligent orbit transfer**.

### NASA / JPL
- **Autonomous Spacecraft Attitude Control Using Deep Reinforcement Learning**
  *Elkins, J.G. et al., NASA Technical Reports, 2020.*
  Provided the framework for AI-based fault-tolerant spacecraft control.
  → Inspired the *Hybrid Imitation + PPO* design and robustness testing pipeline.

### Propulsion Systems & Physical Modeling
- **NFAero – New Frontier Aerospace**
  *Mjölnir full-flow staged combustion engine* (2023).
  Inspired the **nonlinear thrust decay modeling** and **hybrid propulsion switch** simulation.
  → [https://www.nfaero.com](https://www.nfaero.com)
- **Electric Propulsion Research (ScienceDirect, 2023)**
  Provided low-thrust continuous model reference for the electric propulsion mode.
- **Photonic & Fusion Propulsion Studies (2023–2025)**
  Inspired long-term expansion modules (`model_photon.py`, `model_fusion.py`) for speculative deep-space drives.

### Academic & Open Science Ecosystem
- **MIT Space Systems Lab** – Open-source astrodynamics frameworks and formation control modeling.
- **KAIST Propulsion Research Group** – Neural network–based Hall-effect thruster performance modeling, reference for *thrust uncertainty simulation*.
- **OpenAI Gym / Stable-Baselines3** – Reinforcement learning modularization standards adopted for reproducibility and controller benchmarking.

---

> *This repository embodies a cross-disciplinary exploration — combining physics, control theory, and artificial intelligence —
>  to prototype the next generation of intelligent spacecraft propulsion systems.*

 *Last Updated:* November 2025
 **Maintainer:** [Sean-ZhiXin-Li](https://github.com/Sean-ZhiXin-Li)

---

## License
This project is licensed under the terms of the [MIT License](LICENSE).
