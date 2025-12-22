# Engineering & Reproducibility Status

## Overview

This project has established a **reproducible, auditable, research-grade engineering setup**.
All experiments are executed under controlled environments, verified toolchains, and validated end-to-end system checks.

The current state is not merely "working," but **designed for long-term experimental integrity and comparison**.

---

## 1. Environment Strategy

### 1.1 Primary Research Environment

* **Environment manager**: Conda
* **Environment name**: `spacecraft`
* **Python version**: 3.12.12
* **Interpreter path**: `E:\\conda3\\envs\\spacecraft\\python.exe`

This environment is the **single source of truth** for:

* Training and evaluation
* Reinforcement learning experiments
* Control and optimization baselines
* Experiment logging and analysis
* SPICE-based validation

The project explicitly avoids mixing:

* System Python
* Windows Store Python
* Project-level `.venv`
* Editor-generated virtual environments

This constraint ensures that **all experimental results are traceable to a single, well-defined runtime context**.

---

### 1.2 Auxiliary Orbit Tooling Environment

* **Environment name**: `orbittools`
* **Purpose**: Orbital initialization, sanity checks, and analytical reference generation
* **Key libraries**: `poliastro`, `astropy`
* **Python version**: 3.10 (chosen for dependency compatibility)

Due to current compatibility constraints between `poliastro` and Python 3.12, orbital tooling is intentionally isolated.
Data generated in this environment (e.g., orbital initial states) is passed to the main project via **explicit data artifacts (JSON)**, not via shared runtime state.

This separation preserves **numerical correctness** without sacrificing the evolution of the main research environment.

---

## 2. Environment Freezing and Reproducibility

Complete Conda environment snapshots have been exported:

```text
conda_envs/
├── spacecraft.yml
└── orbittools.yml
```

* Exported using `conda env export --no-builds`
* Locks dependency relationships rather than platform-specific build artifacts
* Enables environment reconstruction on new machines or systems

This guarantees that experiments can be **reproduced independently of the original development setup**.

---

## 3. Experiment Tracking and Auditability (Weights & Biases)

### Current Status

* Weights & Biases (W&B) is installed **only** in the primary `spacecraft` environment
* Authentication is complete and persisted securely via system `_netrc`
* No credentials are stored in code or the repository
* Real training runs have been verified:

  * `wandb.init`
  * `wandb.log`
  * `wandb.finish`

W&B is used strictly as an **experiment recorder**, not as a control dependency.

### Design Principles

* **Minimal intrusion**: logging does not alter training logic
* **Audit-friendly**: every run records configuration, metrics, and timelines
* **Reversible**: logging can be enabled or disabled without refactoring code

This ensures experiments are **comparable, reviewable, and non-ad hoc**.

---

## 4. Control and Optimization Baselines

The project explicitly avoids relying on reinforcement learning as a single-point solution.

The following components are installed and verified in the main environment:

* **CasADi** — symbolic modeling for optimal control
* **IPOPT** — nonlinear optimization solver (verified via minimal NLP solve)
* **OSQP** — quadratic programming solver
* **do-mpc** — rapid MPC prototyping framework

All components have been tested for importability and basic runtime correctness.

This enables future **PPO vs MPC** or **learning vs optimization** comparisons under identical dynamics.

---

## 5. Orbital Initialization and Numerical Consistency

Orbital initial states are generated using `poliastro` and `astropy` in the `orbittools` environment.

**Workflow**:

1. Generate orbit (e.g., circular orbit at 1 AU)
2. Export position and velocity in SI units to JSON
3. Load JSON into the main `spacecraft` environment

This decouples **orbit definition** from **training runtime**, while keeping initialization physically grounded and verifiable.

---

## 6. SPICE-Based Validation

* `spiceypy` and the CSPICE toolkit are installed in the primary environment
* Toolkit availability and version have been verified at runtime
* SPICE serves as a **fact-based cross-check layer** for ephemerides and physical parameters

This introduces an external, industry-standard reference into the simulation stack.

---

## 7. Full-System Smoke Test

A full-system smoke test script has been implemented and executed successfully.

Verified components include:

* SPICE toolkit availability
* CasADi + IPOPT optimization solve
* OSQP import and readiness
* do-mpc import and version check
* Orbit initialization data path (JSON)
* Unified execution under the `spacecraft` environment

The smoke test provides a **repeatable system health check** for future development, CI integration, and regression detection.

---

## Conclusion

The project now operates under a **research-grade engineering baseline**:

* Single-source runtime environment
* Explicit dependency freezing
* Auditable experiment logging
* Independent physical validation tools
* End-to-end system verification

Experimental results produced under this setup are **reproducible, comparable, and defensible**.

---

## Optional Reproduction Workflow

```text
git clone <repo>
conda env create -f conda_envs/spacecraft.yml
conda activate spacecraft
python analysis/smoke_full_system.py
```

---


