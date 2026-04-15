# AI-Controlled Spacecraft Orbital Simulator

Physics-grounded spacecraft control research for orbit insertion under thrust limits.

This repository compares three control families under the same orbital environment:

- PPO policies
- simple probe baselines
- an explicit phase-structured controller

The current best verified result is not PPO. It is a hand-designed **phase-structured explicit controller** that performs:

1. `DESCENT`
2. `CAPTURE`
3. `LOCK`

in sequence.

## Current Project Result

The project now has a **working explicit insertion controller** that reaches a real target-radius crossing and achieves strict success in a narrow but real reachable regime.

What is supported by the repository outputs:

- PPO baseline does not reach the first crossing on the validated baseline.
- A pure retrograde probe can force one crossing, but does not stabilize afterward.
- The explicit phase controller is the only controller here that both:
  - reaches the first crossing
  - satisfies the project's strict success criterion in successful setups

What is **not** currently supported:

- repeated sustained orbit-lock cycling across the target radius
- successful PPO transfer of the explicit phase structure

Those statements are grounded in:

- [analysis/final_project_summary.md](analysis/final_project_summary.md)
- [analysis/orbit_lock_benchmark.md](analysis/orbit_lock_benchmark.md)
- [analysis/orbit_lock_generalization.md](analysis/orbit_lock_generalization.md)
- [analysis/ppo_transfer_results.md](analysis/ppo_transfer_results.md)

## Key Insight

Orbit insertion in this system is not a single continuous control problem.

It is a **phase-structured process** requiring:

- energy removal (descent)
- post-crossing capture
- near-orbit stabilization

Controllers that do not explicitly represent this structure fail,
even if they have sufficient capacity (e.g., PPO).

## Demo

Running the project entry now generates a real orbital demo animation for the best successful explicit-controller setup.

```bash
python main.py
```

This produces:

- [analysis/demo/orbit_demo.gif](analysis/demo/orbit_demo.gif)
- [analysis/demo/orbit_demo_full.png](analysis/demo/orbit_demo_full.png)
- [analysis/demo/orbit_demo_trajectory.png](analysis/demo/orbit_demo_trajectory.png)
- [analysis/demo/orbit_demo_summary.json](analysis/demo/orbit_demo_summary.json)

Animated demo:

![Orbit demo](analysis/demo/orbit_demo.gif)

Full-orbit reference for global geometric context:

![Orbit full reference](analysis/demo/orbit_demo_full.png)

Zoomed insertion-event view for local control behavior around crossing, capture, and lock:

![Orbit trajectory](analysis/demo/orbit_demo_trajectory.png)

The demo uses the validated explicit-controller setup:

- `r0_over_target = 1.00005`
- `dt = 100`
- `max_steps = 100000`
- `thrust_scale = 10000`

## Why PPO Fails

PPO fails here for structural reasons, not just weak tuning.

Observed failure pattern:

- PPO behaves as a single continuous reactive policy.
- It does not reliably produce the control-phase transition needed for:
  - descent
  - post-crossing capture
  - final lock
- In the validated comparisons, it does not reach the first crossing on the main baseline.

Evidence:

- benchmark successes: explicit `2 / 3`, probe `0 / 3`, PPO `0 / 3`
- transfer result: BC and short PPO fine-tuning still fail to recover first crossing

See:

- [analysis/orbit_lock_benchmark.md](analysis/orbit_lock_benchmark.md)
- [analysis/ppo_transfer_results.md](analysis/ppo_transfer_results.md)

## Why Probe Is Not Enough

The `probe_max_retrograde` controller is useful because it proves physical reachability in the reachable regime, but it is still only a descent baseline.

It can:

- remove orbital energy aggressively
- force one crossing in reachable conditions

It cannot:

- switch behavior after crossing
- capture and stabilize the orbit

So it is a reachability tool, not a full insertion controller.

## Why the Explicit Controller Works

The explicit controller works because it is **phase-structured**.

It does not use one feedback law for the entire insertion process. Instead it switches between:

1. `DESCENT`
   - full retrograde thrust aligned with velocity
   - objective: guarantee first crossing
2. `CAPTURE`
   - damp radial motion after crossing
   - restore tangential support
3. `LOCK`
   - apply smaller near-orbit stabilization

This project's main lesson is:

> orbit insertion is not one homogeneous control problem; it is a sequence of control phases

## Benchmark And Generalization

### Final benchmark comparison

Representative benchmark result summary:

![Success comparison](analysis/figs/final_project/success_comparison.png)

This comparison plot shows strict success counts across the three representative benchmark setups used in the final benchmark pass.

More detail:

- [analysis/orbit_lock_benchmark.md](analysis/orbit_lock_benchmark.md)
- [analysis/figs/orbit_lock_benchmark/aggregate_results.json](analysis/figs/orbit_lock_benchmark/aggregate_results.json)

### Representative controller comparison on a successful setup

The following plot is a controller-comparison diagnostic on one representative successful setup, not a complete characterization of the full physical system.

![Radius vs time](analysis/figs/final_project/radius_vs_time.png)

Related radial-velocity diagnostic:

- [analysis/figs/final_project/final_project_plot_summary.json](analysis/figs/final_project/final_project_plot_summary.json)
- [analysis/figs/final_project/v_r_vs_time.png](analysis/figs/final_project/v_r_vs_time.png)

These 2D diagnostics are used to illustrate controller behavior and compare control regimes. They are useful validation views, but they are not a full physical system characterization by themselves.

### Generalization takeaway

The explicit controller generalizes, but only in a narrow reachable regime.

Verified result:

- success in `5 / 36` tested setups
- working region is limited to:
  - very small initial radius offsets
  - moderate or large `dt`
  - moderate thrust, not maximal thrust

See:

- [analysis/orbit_lock_generalization.md](analysis/orbit_lock_generalization.md)
- [analysis/figs/orbit_lock_generalization/aggregate_results.json](analysis/figs/orbit_lock_generalization/aggregate_results.json)

## Learning Transfer Status

The repository already includes the first learning-transfer stage:

1. behavior cloning from the explicit phase controller
2. short PPO fine-tuning from the cloned policy

Current status:

- explicit controller: crossing `1`, success `true`
- behavior cloning: crossing `0`, success `false`
- PPO fine-tuned from BC: crossing `0`, success `false`

So the explicit structure is solved, but the learned transfer is not solved yet.

Relevant files:

- [analysis/phase_controller_dataset.md](analysis/phase_controller_dataset.md)
- [analysis/ppo_transfer_results.md](analysis/ppo_transfer_results.md)
- [analysis/phase_controller_dataset/phase_controller_dataset_balanced.npz](analysis/phase_controller_dataset/phase_controller_dataset_balanced.npz)

## Repository Structure

```text
spacecraft_ai_project/
├── analysis/               # summaries, figures, demo assets, datasets
├── controller/             # explicit, probe, PPO-related controllers
├── envs/                   # orbital environment and task definitions
├── models/                 # saved BC and PPO-transfer models
├── ppo_orbit/              # PPO implementation
├── scripts/                # evaluation, sweeps, plotting, demo generation
├── main.py                 # current demo entry point
└── README.md
```

## Installation

Recommended:

- Python 3.10+
- Conda environment setup first

The primary recommended setup is the Conda environment below. If you prefer a lightweight manual install, the minimal package list is:

```bash
pip install numpy matplotlib torch pillow gymnasium scikit-learn
```

## Environment Setup (Conda)

This project is developed and tested with Conda.

Create the environment:

```bash
conda env create -f environment.yml
conda activate spacecraft
```

## Reproducibility

### Generate the demo

```bash
python main.py
```

Equivalent direct script:

```bash
python scripts/generate_orbit_demo.py
```

### Re-run the orbit-lock validation

```bash
python scripts/orbit_lock_validation.py
```

### Re-run the generalization sweep

```bash
python scripts/orbit_lock_generalization.py
```

### Re-run the benchmark comparison

```bash
python scripts/orbit_lock_benchmark.py
```

### Re-run the transfer evaluation

```bash
python scripts/train_behavior_cloning.py
python scripts/eval_bc_policy.py --policy-kind bc --checkpoint models/bc_policy.pth
python scripts/eval_bc_policy.py --policy-kind ppo --checkpoint models/ppo_bc_finetuned.pth
```

## Recommended Reading Order

If you want the shortest evidence trail, read:

1. [analysis/final_project_summary.md](analysis/final_project_summary.md)  -(final presentation summary — scoped to the 2D insertion setting)
2. [analysis/orbit_lock_benchmark.md](analysis/orbit_lock_benchmark.md)
3. [analysis/orbit_lock_generalization.md](analysis/orbit_lock_generalization.md)
4. [analysis/ppo_transfer_results.md](analysis/ppo_transfer_results.md)

## Future Directions

This project currently focuses on a simplified 2D orbital insertion setting to isolate control structure and learning behavior.

Future extensions aim to move toward more realistic and higher-performance systems:

- **3D Orbital Dynamics**
  Extending the simulator from planar motion to full 3D orbital mechanics, including inclination and out-of-plane control.

- **Higher-Performance Simulation (C++)**
  Re-implementing the simulation core in C++ to support larger-scale experiments and faster rollout for control and learning.

- **Learning Structured Controllers**
  Improving reinforcement learning approaches (e.g., PPO) by incorporating phase-aware or memory-based architectures, to better capture sequential control structure.

- **Robust Control Under Constraints**
  Studying controller performance under stricter thrust limits, perturbations, and uncertainty in system dynamics.

This direction aligns with the broader goal of developing AI-driven control systems for complex physical environments, particularly in orbital and aerospace applications.

## License

MIT License. See [LICENSE](LICENSE).

