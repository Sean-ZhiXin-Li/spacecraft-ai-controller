# AI-Controlled Spacecraft Orbital Simulator

*A reproducible research framework for developing, benchmarking, and scaling AI-based spacecraft control systems under realistic orbital dynamics.*

---

## Project Overview

This project focuses on building a **physics-grounded, AI-driven control framework** for spacecraft orbit management. It combines simulation, classical control, and modern machine learning into a unified experimental platform.

### Objectives

* Develop AI controllers for orbital thrust and trajectory control
* Benchmark different control strategies under consistent conditions
* Build a reproducible pipeline for research and experimentation
* Bridge simulation results toward real-world embedded systems

### Key Features

* Custom **Gymnasium-compatible orbital environment (`OrbitEnv`)**
* Support for multiple controllers:

  * Heuristic / rule-based
  * Expert-designed controllers
  * Imitation learning (MLP)
  * Reinforcement learning (PPO)
* Standardized evaluation pipeline with:

  * Fuel-aware metrics
  * Orbit accuracy metrics
  * Reproducibility guarantees
* Modular structure for rapid experimentation

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
├── project_log/       # Daily research logs and notes
├── ab/                # Experiment outputs and benchmarks
├── results/           # Final summarized results
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

## Quick Start

Run a baseline evaluation using expert controllers:

```bash
python tools/day39_quickrun.py \
  --tasks_dir ab/day36/task_specs_fast \
  --out_dir ab/day39 \
  --controllers elliptic_strong transfer_2phase spiral_in \
  --limit 64
```

### Output

* CSV summary: `ab/day39/csv/summary.csv`
* Figures:

  * Orbit error comparison
  * Return comparison

---

## Current Progress (Updated)

The project has evolved into a **multi-stage AI control research pipeline**, integrating simulation fidelity, controller diversity, and evaluation robustness.

### What is Working Well

* Stable expert controllers for circular and elliptic orbits
* Reproducible evaluation pipeline (quickrun + summaries)
* Multi-task benchmarking across orbit types
* PPO framework integrated and functional

### Current Limitations

* RL policies still show:

  * Energy inefficiency
  * Thrust misalignment
* Transfer tasks remain harder than circular stabilization
* Reward design is not yet fully aligned with physical efficiency

---

## Research Progress Timeline

### Phase 1 — Foundations

* Built 2D orbital simulator
* Implemented basic expert controllers

### Phase 2 — Learning Models

* Imitation learning (MLP)
* PPO baseline setup

### Phase 3 — RL Refinement

* Improved reward shaping
* Stabilized PPO training

### Phase 4 — Benchmark System

* Established baseline tiers:

  * Zero
  * Greedy
  * Expert

### Phase 5 — Scalable Evaluation

* Task bundles (circular, elliptic, transfer)
* Automated evaluation pipeline

### Phase 6+ — Expansion

* Metrics system
* Energy-based analysis
* Multi-orbit validation

---

## Key Insights

* Expert controllers achieve **high stability** in simple orbits
* RL models require **better physical alignment** in rewards
* Energy efficiency is currently the main bottleneck
* Transfer maneuvers expose weaknesses in learned policies

---

## Next Steps

### Short-Term

* Hybrid training (Imitation → PPO)
* Curriculum learning across orbit complexity
* Improved reward shaping (energy-aware)
* Robustness testing (noise, faults)

### Mid-Term

* 3D orbital dynamics
* Multi-agent coordination (formation flying)
* Advanced propulsion modeling

### Long-Term

* Integration with embedded systems (Arduino / ROS2)
* Real-time control deployment
* Autonomous spacecraft decision-making

---

## Vision

This project aims to become a **complete research framework for autonomous spacecraft control**, combining:

* Physics-based simulation
* Machine learning
* Control theory
* Real-world system integration

The long-term goal is to move from:

> Simulation → Intelligence → Autonomy → Real Deployment

---

## Research Direction

* AI-driven propulsion optimization
* Fault-tolerant autonomous navigation
* Distributed spacecraft systems
* Long-duration mission autonomy

---

## Philosophy

> *Trajectory may drift, but the mission continues.*

This project is not only about solving orbital control problems, but also about exploring how **machine intelligence can persist and operate independently in space environments**.

---

## License

MIT License

---

## Acknowledgment

This work is inspired by ongoing research in:

* Autonomous space systems
* Reinforcement learning for control
* Distributed intelligence in extreme environments

It represents an ongoing effort to push AI beyond simulation into real-world space applications.
