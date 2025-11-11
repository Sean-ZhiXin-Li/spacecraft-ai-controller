# AI-Controlled Spacecraft Orbital Simulator

*A reproducible research framework reflecting the technical and philosophical direction of the Spacecraft AI Controller and Tech Foundations projects.*

---

## Project Overview

* **Goal:** Develop and benchmark AI controllers for spacecraft thrust control under realistic orbital dynamics.
* **Features:**

  * Custom `OrbitEnv` (Gymnasium-compatible) with solar-scale orbits.
  * Multiple controllers: heuristic, expert, imitation (MLP), PPO.
  * Baseline evaluation harness with fuel-aware metrics and reproducibility.
  * Integrated research continuum connecting simulation, control, and hardware experiments.

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
├── project_log/       # Daily project logs (Day1–Day55)
├── ab/                # Experiment results (task specs, csv, figs)
├── results/           # Final CSV summaries and baseline results
├── LICENSE            # License file
└── README.md
```

---

## Installation

### Requirements

* Python 3.10+
* Recommended: create a new virtual environment (venv or conda)

### Install dependencies

```bash
pip install numpy matplotlib torch scikit-learn gymnasium
```

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

* CSV: `ab/day39/csv/summary.csv`
* Figures:

  * `ab/day39/figs/day39_r_err_by_controller.png`
  * `ab/day39/figs/day39_return_by_controller.png`

---

## Project History

This project has evolved from simple orbital physics simulation to multi-phase AI control research, combining expert systems, imitation learning, and reinforcement learning within a reproducible workflow.

### Phase 1 – Foundations (Day 1–10)

Built the first 2D `OrbitEnv` with gravitational physics and expert controllers. Generated imitation datasets and ran closed-loop training tests.

### Phase 2 – Imitation & PPO Experiments (Day 11–19)

Trained imitation models and PPO baselines, stabilized reward shaping, and implemented energy-aligned penalties for realistic thrust behavior.

### Phase 3 – Hybrid & PPO Refinement (Day 20–29)

Introduced hybrid PPO initialization using expert warm starts. Improved convergence through KL-adaptive updates and entropy scheduling.

### Phase 4 – Baseline Phase A/B (Day 30–35)

Established reproducible benchmarks: Zero (lower bound), Greedy (mid-baseline), Expert (upper bound). Achieved stable circular and elliptic orbits.

### Phase 5 – Complex Environment & Quickruns (Day 36–39)

Implemented task bundles across circular, elliptic, and transfer orbits. Automated replay and summary pipelines for scalable evaluation.

### Phase 6–10 – Research Expansion

* Multi-orbit environment integration and fault simulation.
* Quantitative metrics (`metrics_core.py`) and energy-momentum analyses (`energy_view.py`).
* Validation of Expert v3.1 controllers across diverse orbital families.

---

## I.5 Research Continuum — From Simulation to Concept Architecture

The current 2D simulation environment represents the foundation of a broader vision. Each iteration, from imitation learning to PPO and curriculum-based control, builds reproducible autonomy for future orbital intelligence.

This continuum extends into the **Tech Foundations** repository, where embedded experimentation is underway. Integrating AI controllers with embedded systems (Arduino and ROS2) bridges simulation and hardware, transforming algorithms into tangible, real-time architectures. Each simulation cycle thus becomes a step toward practical, autonomous spacecraft systems.

---

## Results & Figures

| Scenario           | Controller  | Reward  | Final Orbit Error | Δv₁ (m/s) | Notes                               |
| ------------------ | ----------- | ------- | ----------------- | --------- | ----------------------------------- |
| Circular           | Expert v3.1 | −1541.7 | 0.25              | 0         | Stable baseline                     |
| Elliptic           | Expert v3.1 | −1541.7 | 0.25              | 0         | Stable circular-like performance    |
| Transfer (2-Phase) | Expert v3.1 | −2769.1 | 0.25↑             | ≈563      | Δv₁ matches Hohmann transfer theory |
| Spiral-In          | Expert v3.1 | −1503   | 0.31              | –         | Energy deviation +38.9%             |

### Interpretation

* Expert controllers achieved 100% stability in circular and elliptic orbits.
* Transfer and spiral-in scenarios revealed energy-thrust inefficiency and angular misalignment.
* Week 0 metrics indicated ~23% thrust saturation, sufficient but not excessive.
* Energy convergence efficiency η ≈ 2.8%, motivating future reward redesign.

---

## Next Steps

### Immediate Objectives (Phase 1–2)

* **Hybrid Imitation + PPO Initialization**  Combine expert imitation pretraining with PPO fine-tuning for faster convergence.
* **Curriculum Training across Multi-Orbit Tasks**  Gradually expand environment complexity to improve generalization.
* **Robustness under Faults & Sensor Noise**  Monte Carlo testing with thrust degradation and attitude noise.
* **Expanded Evaluation Pipeline**  Aggregate performance metrics and build hardest-task benchmark suite.

### Planned Extensions (Phase 3–5)

* **3D Orbit Dynamics Integration**  Extend to full 3D mechanics with inclination parameters.
* **Multi-Agent Formation & Rendezvous Control**  Implement coordination among spacecraft for formation-keeping and docking.
* **Advanced Propulsion Models**  Add switchable propulsion backends (chemical, electric, photonic, fusion).
* **Reproducible Research Suite**  Modular logging, visualization, and metrics integration for open research reproducibility.

---

## Project Vision / Long-Term Goal

> *“Trajectory may drift, but the mission continues.”*

This project aspires to evolve into a physics-grounded framework for **autonomous spacecraft propulsion and orbit transfer**, bridging AI decision-making with real-world astrodynamics.

### Long-Term Research Direction

* **AI × Propulsion Control**  Develop controllers that optimize multi-mode thrusters and long-duration orbit transfers under uncertainty.
* **From Single Spacecraft to Distributed Systems**  Expand toward formation flying and distributed autonomy inspired by Stanford’s SLAB.
* **Toward Real-World Autonomy**  Integrate reinforcement learning with fault tolerance for resilience beyond Earth contact.
* **Educational & Open Science Goals**  Provide a transparent testbed for astrodynamics and AI control research.

---

## Epilogue — Toward the Continuum of Exploration

The frontier of exploration will be defined not by distance but by persistence, by how many autonomous agents can endure, exchange data, self-repair, and evolve collectively.

What began as a 2D simulation has grown into a conceptual framework for distributed cognition in space. The next step is the realization of embedded, real-time implementations through the **Tech Foundations** initiative, integrating AI controllers with hardware and ROS2 platforms.

This vision synthesizes reinforcement learning, astrodynamics, and control theory into a unified narrative of machine cognition in space. The mission is not to reach perfection but to make exploration sustainable, across generations, worlds, and the silence beyond Earth.

> *Every trajectory fades, but intelligence endures, carried by the machines we send and the curiosity that sent them.*

---

## Related Document

For a detailed philosophical and technical vision behind this project, see  
 [Inspiration Log — From Autonomous Propulsion to Cosmic Intelligence](project_log/inspiration_2025_11_11.md)


## References & Inspirations

* **Stanford University** — Space Rendezvous Laboratory (SLAB) and CAESAR, research on distributed autonomy and AI-guided control.
* **NASA JPL** — Reinforcement learning and fault-tolerant navigation for deep-space probes.
* **ESA ACT** — AI-driven guidance, navigation, and control (GNC) for autonomous exploration.
* **MIT Space Systems Lab** — Human-in-the-loop autonomy.
* **KAIST Propulsion Group** — Intelligent propulsion modeling and degradation prediction.
* **NFAero** — Nonlinear thrust decay and hybrid propulsion concepts.
* **Breakthrough Starshot & NIAC Concepts** — Photon sails, magnetic sails, nuclear and fusion propulsion as frontier models.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgment

Inspired by Stanford Engineering’s broader aspiration of enabling intelligence beyond Earth, this repository represents a continuous journey toward understanding how AI can extend the reach of spacecraft long after contact is lost.

> *Every trajectory fades, but intelligence endures, carried by the machines we send and the curiosity that sent them.*
