# AI-Controlled Spacecraft Orbital Simulator

*A reproducible research framework for developing, benchmarking, and scaling AI-based spacecraft control systems under realistic orbital dynamics.*

---

## Project Overview

This project presents a **physics-grounded control framework for spacecraft orbital dynamics**, integrating classical control strategies with modern machine learning methods.

Rather than focusing solely on learning-based approaches, this work emphasizes:

* **Control structure design** (e.g., gated control mechanisms)
* **Physics-consistent evaluation**
* **Systematic comparison between rule-based and learning-based controllers**

The goal is to build a **research-grade experimental platform** for studying autonomous spacecraft control under constrained thrust and realistic orbital conditions.

---

## Core Contribution

This project introduces a **gated control mechanism for orbital insertion**, 
which improves stability under constrained thrust conditions 
and reveals structural differences between rule-based and learning-based controllers.

Key contribution:

- Demonstrates that control structure (gating) can outperform naive continuous thrust
- Provides a reproducible framework for comparing control strategies under identical physics
- Bridges classical control intuition with modern RL approaches

---

## Objectives

* Develop control strategies for orbital insertion and stabilization
* Benchmark rule-based and learning-based controllers under identical physics
* Build a reproducible experimentation pipeline
* Analyze control behavior through trajectory-level diagnostics

---

## Key Features

* Custom **Gymnasium-compatible orbital environment (`OrbitEnv`)**
* Multiple controller paradigms:

  * Rule-based / heuristic controllers
  * Expert-designed controllers
  * Imitation learning (MLP)
  * Reinforcement learning (PPO)
* Unified evaluation pipeline with:

  * Orbit accuracy metrics
  * Stability analysis
  * Trajectory logging (`traj.npz`)
* Automated experiment management and visualization

---

## Demo: Orbital Insertion Performance

## Demo: Orbital Control Behavior (PPO Controller)

### Best Behavior-Based Result (PPO)

The best recovered PPO controller demonstrates **stable long-horizon behavior**, 
but does not yet achieve true closed-loop orbit lock.

#### Radius vs Time

![radius](analysis/figs/radius_vs_time.png)

#### Radial Velocity vs Time

![vr](analysis/figs/vr_vs_time.png)

### Observations

* The spacecraft maintains stable motion for **20,000 steps**
* Radius remains consistently above the target orbit (biased trajectory)
* Radial velocity quickly converges to a small value but **does not oscillate around zero**
* The controller reduces radial motion but does not form a full closed-loop system

### Metrics

- Survival: 20,000 steps
- Final radius error: ~3.75e11
- Average radius error: ~3.75e11 :contentReference[oaicite:0]{index=0}

### Configuration

```text
Controller: PPO (speed_refine_50)
Checkpoint: ppo_orbit/speed_refine_50/ppo_epoch_300.pth
Initial condition: r0 ≈ 1.05 × target radius
```
### Run the demo locally

```bash
python scripts/recover_ppo_rollout.py --checkpoint ppo_orbit/speed_refine_50/ppo_epoch_300.pth --runs-root analysis/runs --thrust-scale 20000 --r0-over-target 1.05 --max-steps 20000
```

This will:

* Automatically select the best run from `analysis/runs/`
* Generate trajectory plots in `analysis/demo_best/`

---

## Repository Structure

```
spacecraft_ai_project/
│
├── simulator/         # Orbital physics and environment (OrbitEnv)
├── controller/        # Expert, imitation, and RL controllers
├── data/              # Training datasets (expert trajectories)
├── ppo_orbit/         # PPO training implementation
├── tools/             # Utilities (quickrun, plotting, summarization)
├── project_log/       # Research logs and experiment notes
├── analysis/          # Runs, trajectories, and evaluation outputs
├── results/           # Aggregated experiment results
├── LICENSE
└── README.md
```

---

## Installation

### Requirements

* Python 3.10+
* Recommended: virtual environment (venv or conda)

### Install Dependencies

```bash
pip install numpy matplotlib torch scikit-learn gymnasium
```

---

## Experimental Pipeline

This project follows a structured research workflow:

1. **Environment Setup** — fixed physics and normalization
2. **Controller Design** — rule-based vs learning-based
3. **Batch Experiments** — parameter sweeps (thrust, initial radius)
4. **Trajectory Logging** — saved as `traj.npz`
5. **Metric Extraction** — reward, stability, error
6. **Visualization & Analysis** — automated plotting tools

This ensures **fair comparison and reproducibility** across experiments.

---

## Current Progress

### Working Components

* Stable expert and gated controllers
* Reproducible evaluation pipeline
* Automated trajectory analysis and visualization
* PPO integration with training and checkpointing

### Limitations

* RL policies show:

  * Energy inefficiency
  * Suboptimal thrust alignment
* Transfer maneuvers remain challenging
* Reward design is not fully aligned with physical optimality

---

## Key Insights

* Gated control achieves **strong stability under constrained thrust**
* RL models require **physics-aligned reward shaping**
* Energy efficiency is a primary bottleneck
* Learned policies struggle with long-horizon transfer tasks

---

## Research Direction

* Hybrid control (Imitation → PPO)
* Curriculum learning for orbital complexity
* Energy-aware reward design
* Robustness under noise and system uncertainty

---

## Vision

This project aims to evolve into a **research platform for autonomous spacecraft control**, combining:

* Orbital physics simulation
* Control theory
* Machine learning
* Autonomous decision systems

Long-term direction:

> Simulation → Control → Learning → Autonomous Space Systems

---

## License

MIT License

---

## Acknowledgment

Inspired by research in:

* Autonomous space systems
* Reinforcement learning for control
* AI-driven robotics and dynamics

---

## Philosophy

> *Trajectory may drift, but control adapts.*

This project explores how intelligent systems can operate reliably under uncertainty in complex physical environments.
