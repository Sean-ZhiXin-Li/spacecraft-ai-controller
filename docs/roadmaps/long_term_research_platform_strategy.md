# 3-5 Year Technical Strategy For `spacecraft-ai-controller`

## Executive Summary

This repository should not be rebuilt. It already contains the seed of a serious autonomous spacecraft control research platform: a 2D simulator, staged explicit controllers, PPO/behavior cloning experiments, benchmark artifacts, replay tools, metrics tools, documentation, CI, and a unusually careful evidence trail.

The next 3-5 years should be about turning the current phase-driven research repo into a modular research platform while preserving the phase history. The right direction is not “clean slate architecture.” The right direction is controlled extraction:

```text
phase scripts -> reusable platform modules -> benchmarked research systems
```

The current repository has three major strengths:

1. It preserves experimental failures instead of hiding them.
2. It has already discovered that benchmark semantics matter more than headline success counts.
3. It contains many emerging subsystem seeds: simulator, controller, benchmark, metrics, replay, dataset, learning, documentation, CI.

Its major weakness is that these subsystems are not yet clean architectural layers. The current implementation is still organized around chronological experiments. That was correct for the paper, but it will become a liability over several years.

The long-term strategy should be:

```text
Year 1: stabilize the 2D research platform
Year 2: modularize benchmark/controller/evaluator infrastructure
Year 3: add optimization, uncertainty, and robust autonomy
Year 4: expand simulator fidelity: 3D, attitude, perturbations
Year 5: build mission-level autonomy and multi-agent/long-horizon research
```

The core rule: every new capability must enter through benchmarks, metrics, and reproducibility contracts. Do not add realism faster than the evaluation system can explain failures.

---

# Part 1: Current Architecture Evaluation

## 1.1 What Architecture Already Exists

The repository already has a real architecture, but it is implicit rather than formal.

### Simulator And Environment Layer

Current files:

- `envs/orbit_env.py`
- `envs/multi_orbit_env.py`
- `envs/orbit_presets.py`
- `envs/task_sampler.py`
- `simulator/physics.py`
- `simulator/simulate_orbit.py`
- `simulator/types.py`
- `simulator/tasks.py`
- `simulator/config.py`
- `simulator/orbit_analysis.py`
- `simulator/visualize.py`

There are really two simulator paths:

1. `envs/`: Gym/Gymnasium-style environment used by PPO, smoke tests, and older control loops.
2. `simulator/`: a lighter physics/utilities layer, closer to reusable simulation primitives.

`envs/orbit_env.py` is currently a large mixed-responsibility environment. It owns physics integration, reward shaping, state representation, action smoothing, orbit-capture assist, termination logic, info fields, success checks, and runtime configuration. It is functional, but too central.

`envs/multi_orbit_env.py` is a wrapper around presets and scenario handling. It is a seed of a scenario/benchmark interface, but it is not yet the platform-level benchmark API.

`simulator/physics.py` is much cleaner: circular speed, circular state generation, and one-step dynamics. This is the shape future core physics code should move toward.

### Controller Layer

Current files:

- `controller/orbit_lock_controller.py`
- `controller/stable_orbit_controller.py`
- `controller/expert_controller.py`
- `controller/expert_controller_improved.py`
- `controller/ppo_controller.py`
- `controller/combined_controller.py`
- `controller/imitation_controller.py`
- `controller/imitation_controller_V6_1.py`
- `controller/factory.py`
- `baselines/zero_thrust.py`
- `baselines/greedy_radial.py`
- `baselines/greedy_energy_rt.py`

The actual controller architecture evolved over time:

```text
simple expert/heuristic -> PPO/IL -> explicit staged controllers -> transfer family scripts -> Phase34 post-cross synchronization
```

The strongest current controller logic is not centralized in `controller/`. It lives inside phase scripts such as:

- `scripts/explicit_controller_phase34_post_cross_sync.py`
- `scripts/explicit_controller_phase36b_transfer_family_benchmark.py`
- `scripts/explicit_controller_phase37a_radial_commit_timing.py`
- `scripts/explicit_controller_phase37b_weak_tangential_subset.py`

That is understandable historically, but long-term it is a problem. The most scientifically important controller behavior is embedded inside benchmark scripts instead of reusable controller modules.

`controller/orbit_lock_controller.py` is important because it defines the CAPTURE/LOCK style terminal behavior. But Phase34’s post-cross synchronization behavior is not cleanly exposed as a reusable controller class. It is implemented in phase scripts.

### Experiment And Benchmark Layer

Current files:

- `scripts/explicit_controller_phase*_*.py`
- `scripts/check_phase_results.py`
- `docs/benchmark_contract.md`
- `docs/planner_search_benchmark_manifest.md`
- `analysis/artifact_manifest.md`
- `.github/workflows/python-app.yml`

The benchmark architecture is one of the strongest parts of the repository. It is not yet formalized as code, but it is formalized in documents and regression checks.

The key benchmark contract is the 24-case reduced benchmark:

```text
r0_over_target: 0.98, 1.00, 1.02
initial_velocity_angle_deg: 150, 165, 170, 175
thrust_scale: 8000, 10000
```

The strongest engineering pattern is:

```text
run experiment -> write CSV -> write Markdown summary -> preserve artifact -> check protected aggregate counts
```

This is exactly the pattern that should become a first-class experiment manager later.

### Metrics Layer

Current files:

- `tools/metrics/metrics_core.py`
- `tools/metrics/compute_metrics.py`
- `tools/metrics/energy_view.py`
- `tools/metrics/npz_inspector_v2.py`
- metric logic inside phase scripts
- `docs/logging_schema_v2.md`

The metrics layer exists but is split:

- Older metrics tools analyze replay/NPZ outputs.
- Phase scripts compute crossing/recoverability metrics inline.
- `docs/logging_schema_v2.md` proposes a better passive observability schema.

The project has already learned that metrics are scientific infrastructure, not just reporting. Crossing, recoverability, closest approach, overspeed, instability, CAPTURE, LOCK, fuel, and effort should become a stable metrics API.

### Replay And Dataset Layer

Current files:

- `tools/replay_recorder.py`
- `tools/replay_player.py`
- `utils/replay_io.py`
- `data/generate_dataset.py`
- `data/thrust_dataset.py`
- `train/preprocess_merge_dataset.py`
- `train/train_mimic.py`
- `scripts/build_phase_controller_dataset.py`
- `analysis/phase_controller_dataset/*`

There are three dataset eras:

1. Raw expert trajectory datasets under `data/`.
2. PPO/IL model artifacts and checkpoints.
3. Phase-controller datasets under `analysis/phase_controller_dataset/`.

The dataset system is not yet versioned. Some files are huge, including large merged expert arrays. This will become a major bottleneck if not formalized.

### Learning Layer

Current files:

- `ppo_orbit/ppo.py`
- `ppo_orbit/model.py`
- `ppo_orbit/rewards_utils.py`
- `scripts/train_behavior_cloning.py`
- `scripts/eval_bc_policy.py`
- `train/train_mimic.py`
- `main_imitation.py`
- `controller/ppo_controller.py`
- `controller/imitation_controller*.py`
- `models/`

The learning layer is historically important, but not currently the scientific core. PPO and imitation learning were diagnostic and did not provide the main positive result.

Architecturally, this layer should stay, but be demoted from “main solver” to “baseline and learning-assisted planning infrastructure.” It should eventually use the same benchmark/evaluator interfaces as explicit controllers.

### Documentation And Evidence Layer

Current files:

- `README.md`
- `docs/research_direction.md`
- `analysis/artifact_manifest.md`
- `docs/benchmark_contract.md`
- `docs/logging_schema_v2.md`
- `docs/research_workspace/*`
- `project_log/*`
- `analysis/phase*_*/summary.md`

This is unusually strong. The repository has a living research memory. That is a real asset.

The problem is volume and navigation. There are many historical logs, generated print artifacts, submission artifacts, and old result files. A future collaborator could easily confuse:

```text
current evidence
historical evidence
diagnostic evidence
submission artifacts
print artifacts
learning artifacts
demo artifacts
```

The solution is not deletion. The solution is classification, manifests, and stable navigation.

### CI And Reproducibility Layer

Current files:

- `.github/workflows/python-app.yml`
- `Tests/test_env_smoke.py`
- `Tests/test_quickrun_smoke.py`
- `Tests/compat_shims.py`
- `scripts/check_phase_results.py`
- `conda_envs/spacecraft_linux.yml`
- `conda_envs/spacecraft.yml`
- `conda_envs/orbittools.yml`

CI currently runs:

- environment smoke tests
- quickrun smoke
- controller import/action smoke
- CSV-only protected result guard

This is appropriate for the current state. But as the platform grows, CI needs tiers:

```text
fast smoke
unit tests
benchmark contract tests
nightly benchmark subset
artifact/schema validation
optional heavy regression
```

---

## 1.2 Which Modules Are Beginning To Emerge Naturally

The natural long-term modules are already visible:

| Emerging Module | Current Evidence |
|---|---|
| `simulation` | `envs/`, `simulator/`, repeated `env_step` functions in phase scripts |
| `controllers` | `controller/`, `baselines/`, controller logic embedded in phase scripts |
| `benchmarks` | benchmark docs, phase scripts, 24-case manifest |
| `evaluation` | `scripts/check_phase_results.py`, phase aggregators |
| `metrics` | phase metrics, `tools/metrics`, logging schema |
| `experiments` | phase scripts and analysis directories |
| `datasets` | `data/`, `analysis/phase_controller_dataset`, replay tools |
| `replays` | `tools/replay_recorder.py`, `tools/replay_player.py`, `utils/replay_io.py` |
| `learning` | `ppo_orbit/`, `train/`, `scripts/train_behavior_cloning.py` |
| `visualization` | plotting inside scripts, `tools/plots/`, generated figures |
| `documentation/evidence` | artifact manifest, benchmark contract, project logs |

These should become formal package boundaries gradually.

---

## 1.3 Which Parts Are Too Tightly Coupled

### Phase Scripts Mix Too Many Concerns

The Phase34/36/37 scripts combine:

- physics constants
- case definitions
- controller implementation
- rollout loop
- metric computation
- failure labeling
- CSV writing
- Markdown generation
- plotting
- regression interpretation

That was productive for research velocity. It is not sustainable for five years.

Long-term target:

```text
controller definitions -> reusable
rollout engine -> reusable
metrics -> reusable
benchmark spec -> declarative
report generation -> reusable
phase script -> thin orchestration
```

### `envs/orbit_env.py` Owns Too Much

It owns:

- dynamics
- reward
- termination
- success
- smoothing
- capture assist
- diagnostics
- info schema
- state serialization

This should not be split immediately. But over time, it should delegate:

```text
dynamics -> simulator dynamics module
reward -> reward module
termination -> termination module
metrics/info -> observability module
scenario reset -> scenario/task module
```

### Learning Code Is Coupled To One Environment Representation

`ppo_orbit/ppo.py` imports `OrbitEnv` directly and uses local normalization constants. This is fine for early experiments, but later learning baselines should depend on interfaces:

```text
EnvFactory
ObservationNormalizer
Policy
Evaluator
BenchmarkSpec
```

### Metrics Are Coupled To Historical CSV Schemas

Current phase outputs are evidence artifacts, so they should not be rewritten. But future metrics need a stable schema layer. Otherwise every new phase will create its own slightly different interpretation of crossing, recoverability, closest approach, and success.

---

## 1.4 Files Accumulating Too Many Responsibilities

High-risk files:

| File | Why It Is Heavy |
|---|---|
| `envs/orbit_env.py` | environment, dynamics, reward, action smoothing, capture assist, termination, diagnostics |
| `scripts/explicit_controller_phase34_post_cross_sync.py` | core terminal-controller result plus benchmark runner, metrics, plots, summary |
| `scripts/explicit_controller_phase36b_transfer_family_benchmark.py` | transfer family implementation plus benchmark/evaluator/reporting |
| `scripts/explicit_controller_phase37a_radial_commit_timing.py` | controller variant logic plus benchmark/evaluator/reporting |
| `scripts/explicit_controller_phase37b_weak_tangential_subset.py` | subset diagnostic plus regression logic |
| `ppo_orbit/ppo.py` | PPO algorithm, normalization, env management, logging, evaluation |
| `ppo_orbit/rewards_utils.py` | many reward regimes and diagnostic values |
| `README.md` | public narrative, demo, architecture history, limitations, future vision |

These files do not need immediate aggressive refactoring. They need extraction pressure: new code should avoid adding more responsibilities to them.

---

# Part 2: Future Architecture

The future architecture should be modular, but not over-abstracted. The project is still a research platform, not production flight software.

A good target architecture:

```text
spacecraft_ai/
  sim/
  scenarios/
  controllers/
  planners/
  benchmarks/
  evaluation/
  metrics/
  logging/
  datasets/
  replays/
  learning/
  visualization/
  experiments/
  reports/
```

The existing directories can remain during transition. New modules can be introduced gradually under `spacecraft_ai/` or `src/spacecraft_ai/`.

## 2.1 Simulator Layer

### Responsibility

Own the physics integration and state transition.

It should answer:

```text
given state, action, dt, dynamics model -> next state
```

It should not own:

- reward
- controller logic
- benchmark definitions
- plotting
- paper-specific metrics
- learning algorithms

### Current Starting Points

- `simulator/physics.py`
- `simulator/types.py`
- `envs/orbit_env.py`
- repeated `env_step` functions inside phase scripts

### Future Shape

```text
spacecraft_ai/sim/
  state.py
  dynamics_2d.py
  dynamics_3d.py
  integrators.py
  forces.py
  termination.py
  units.py
```

### Key Interfaces

```python
class DynamicsModel:
    def step(self, state: State, action: Action, dt: float) -> State:
        ...
```

```python
@dataclass
class State2D:
    x: float
    y: float
    vx: float
    vy: float
    mass: float
```

```python
@dataclass
class Action2D:
    ax_cmd: float
    ay_cmd: float
```

### Design Rule

The simulator layer should be boring. It should not know whether a run is Phase34, PPO, MPC, or a benchmark.

### Future Plug-In Points

- 2D central gravity
- 3D two-body gravity
- J2 perturbations
- drag
- solar radiation pressure
- mass depletion
- actuator limits
- sensor noise
- multi-body gravity

---

## 2.2 Scenario And Task Layer

### Responsibility

Own initial conditions, target orbit definitions, randomized task generation, and scenario metadata.

### Current Starting Points

- `envs/orbit_presets.py`
- `envs/task_sampler.py`
- `docs/planner_search_benchmark_manifest.md`
- benchmark grids embedded in phase scripts

### Future Shape

```text
spacecraft_ai/scenarios/
  task.py
  scenario_registry.py
  initial_conditions.py
  target_orbits.py
  samplers.py
```

### Key Concepts

A task should be an immutable object:

```python
@dataclass(frozen=True)
class OrbitInsertionTask:
    task_id: str
    initial_state: State
    target_orbit: TargetOrbit
    thrust_scale: float
    max_steps: int
    tags: tuple[str, ...]
```

### What Should Never Depend On It

The task layer should not depend on a specific controller. Controllers consume tasks; tasks do not know controllers.

### Future Research Use

- fixed benchmark grids
- randomized initial conditions
- held-out benchmark splits
- mission families
- degraded/faulted initial states
- transfer tasks
- rendezvous tasks

---

## 2.3 Controller Layer

### Responsibility

Define policies that map observation/state/history to action.

### Current Starting Points

- `controller/`
- `baselines/`
- Phase34/36/37 embedded controllers
- `ppo_orbit/`
- `scripts/eval_bc_policy.py`

### Future Shape

```text
spacecraft_ai/controllers/
  base.py
  explicit/
    phase34_terminal.py
    transfer_families.py
    orbit_lock.py
  baselines/
    zero.py
    greedy_radial.py
    retrograde.py
  learned/
    bc_policy.py
    ppo_policy.py
  hybrid/
    guarded_policy.py
    fallback_controller.py
```

### Key Interface

```python
class Controller:
    def reset(self, task: OrbitInsertionTask) -> None:
        ...

    def act(self, obs: Observation) -> Action:
        ...

    def diagnostics(self) -> dict:
        ...
```

### Important Design Rule

Controllers should not write CSV files, generate plots, or decide benchmark success. They may expose internal phase labels and diagnostics.

### Future Plug-In Points

- explicit staged controllers
- MPC controllers
- direct optimization warm-start controllers
- behavior-cloned policies
- PPO/SAC policies
- hybrid controllers with fallback
- fault-aware controllers
- uncertainty-aware controllers

---

## 2.4 Planner Layer

### Responsibility

Own long-horizon intent and trajectory-level decisions. A planner is not the same as a low-level controller.

### Current Starting Points

- `scripts/explicit_controller_phase21_orbital_transfer_planner.py`
- `scripts/explicit_controller_phase22_two_burn_transfer.py`
- `scripts/explicit_controller_phase31_global_transfer_solver.py`
- `scripts/phase32_direct_optimal_control.py`
- Phase36 transfer family code

### Future Shape

```text
spacecraft_ai/planners/
  base.py
  transfer_planner.py
  burn_coast_planner.py
  trajectory_optimizer.py
  mpc.py
  handoff.py
```

### Key Boundary

Planner decides:

```text
what trajectory family / phase / target corridor should be attempted
```

Controller decides:

```text
what action should be applied now
```

### What Should Never Depend On It

The simulator should not depend on planner code. Metrics should not depend on planner internals except optional phase labels.

### Future Research Use

- transfer window creation
- crossing generation
- MPC-lite
- direct collocation
- hybrid planning plus explicit terminal control
- learning-assisted planner proposal

---

## 2.5 Benchmark Layer

### Responsibility

Define reproducible evaluation suites.

### Current Starting Points

- `docs/benchmark_contract.md`
- `docs/planner_search_benchmark_manifest.md`
- hardcoded case grids in phase scripts
- `scripts/check_phase_results.py`

### Future Shape

```text
spacecraft_ai/benchmarks/
  spec.py
  registry.py
  suites/
    crossing24_v1.yaml
    heldout_grid_v1.yaml
    randomized2d_v1.yaml
    robustness2d_v1.yaml
```

### Benchmark Spec Should Include

- benchmark ID
- version
- task list or sampler
- random seeds
- simulator version
- metrics schema
- acceptance criteria
- protected regression cases
- expected artifact paths

### Example

```yaml
benchmark_id: crossing24_v1
version: 1.0
tasks:
  r0_over_target: [0.98, 1.00, 1.02]
  initial_velocity_angle_deg: [150, 165, 170, 175]
  thrust_scale: [8000, 10000]
terminal_controller: phase34_radius_priority
metrics_schema: crossing_recoverability_v1
```

### Design Rule

Benchmark definitions should be declarative. Benchmark execution should be code.

---

## 2.6 Evaluation Layer

### Responsibility

Run controllers/planners on benchmarks and produce structured results.

### Current Starting Points

- phase scripts
- `scripts/check_phase_results.py`
- `eval_battery.py`
- `tools/testing/stress_battery.py`

### Future Shape

```text
spacecraft_ai/evaluation/
  runner.py
  rollout.py
  result.py
  aggregation.py
  regression.py
  comparison.py
```

### Key Interfaces

```python
class EvaluationRunner:
    def run(self, controller, benchmark_spec) -> EvaluationResult:
        ...
```

```python
@dataclass
class RolloutResult:
    task_id: str
    controller_id: str
    metrics: dict
    trajectory_ref: str | None
    diagnostics: dict
```

### What Should Never Depend On It

Controllers should not depend on evaluator internals. They should only expose actions and diagnostics.

---

## 2.7 Metrics Engine

### Responsibility

Compute scientific metrics from rollouts.

### Current Starting Points

- `tools/metrics/`
- phase script metric functions
- `docs/logging_schema_v2.md`

### Future Shape

```text
spacecraft_ai/metrics/
  crossing.py
  recoverability.py
  safety.py
  effort.py
  orbital_elements.py
  phase_metrics.py
  schema.py
```

### Stable Metrics

The following should become first-class metrics:

- `crossing_occurs`
- `first_crossing_step`
- `radius_crossings_total`
- `recoverable_crossing`
- `phase34_compatible_crossing`
- `closest_approach_step`
- `min_abs_radius_error_ratio`
- `crossing_vr_ratio`
- `crossing_vt_error_ratio`
- `crossing_sync_error`
- `best_post_cross_distance`
- `overspeed`
- `instability`
- `termination_reason`
- `cumulative_delta_v_proxy`
- `cumulative_control_effort`
- `radial_effort_proxy`
- `tangential_effort_proxy`
- `specific_energy_*`
- `angular_momentum_*`

### Design Rule

Metrics compute. They do not decide what is scientifically important unless a benchmark contract names them as primary criteria.

---

## 2.8 Logging Layer

### Responsibility

Write stable, versioned experiment records.

### Current Starting Points

- CSVs in `analysis/`
- `docs/logging_schema_v2.md`
- replay metadata
- W&B notes in docs

### Future Shape

```text
spacecraft_ai/logging/
  schema.py
  writers.py
  manifest.py
  provenance.py
```

### Requirements

Every experiment should record:

- git commit
- environment
- benchmark version
- controller version
- simulator version
- task ID
- random seed
- metrics schema version
- artifact directory
- source script
- notes/caveats

### Design Rule

Historical CSVs remain historical. New schema versions get new files or new directories.

---

## 2.9 Replay System

### Responsibility

Store and replay trajectories.

### Current Starting Points

- `tools/replay_recorder.py`
- `tools/replay_player.py`
- `utils/replay_io.py`
- `replays/`
- `analysis/demo/*`

### Future Shape

```text
spacecraft_ai/replays/
  recorder.py
  player.py
  format.py
  validation.py
```

### Replay Format

Use a stable artifact format:

```text
replay.npz
meta.json
metrics.json
events.jsonl
```

### Future Use

- debug controller failure
- visual inspection
- dataset generation
- reproducibility
- learned-policy training data
- paper figures

---

## 2.10 Dataset Manager

### Responsibility

Track datasets, not just files.

### Current Starting Points

- `data/`
- `analysis/phase_controller_dataset/`
- `train/preprocess_merge_dataset.py`
- `scripts/build_phase_controller_dataset.py`

### Future Shape

```text
spacecraft_ai/datasets/
  manifest.py
  builders.py
  splits.py
  loaders.py
```

### Dataset Manifest

Each dataset should have:

- dataset ID
- source benchmark/run
- controller source
- generated date
- git commit
- schema
- size
- train/val/test split
- known caveats

### Design Rule

No learning paper or result should cite a dataset path alone. It should cite a dataset manifest.

---

## 2.11 Visualization Layer

### Responsibility

Generate standard plots from standard result objects.

### Current Starting Points

- plots embedded in phase scripts
- `tools/plots/`
- `simulator/visualize.py`
- many `analysis/*/*.png`

### Future Shape

```text
spacecraft_ai/visualization/
  trajectory.py
  metric_plots.py
  benchmark_summary.py
  report_figures.py
```

### Design Rule

Plotting should depend on results and replays, not on controller internals.

---

## 2.12 Experiment Manager

### Responsibility

Provide a thin orchestration layer for experiments.

### Current Starting Points

- phase scripts
- docs/phase design files
- analysis output directories

### Future Shape

```text
experiments/
  phase34_reproduce.py
  phase40_heldout_benchmark.py
  gen1_crossing_platform.py
  gen2_uncertainty_eval.py
```

or:

```text
spacecraft_ai/experiments/
  run_experiment.py
  registry.py
```

### Experiment Contract

An experiment should specify:

- objective
- benchmark
- controller/planner
- metrics schema
- output directory
- expected guard checks
- whether it may overwrite anything

### Design Rule

Experiment scripts should be short. Most logic should live in platform modules.

---

# Part 3: Long-Term Research Roadmap

Instead of Phase38/39/40, think in generations.

## Generation 1: 2D Recoverability Platform

Timeline: 0-12 months

### Engineering Objective

Turn the current phase-script evidence base into a reusable 2D research platform.

### Scientific Objective

Understand crossing generation and recoverability in simplified 2D orbital insertion under controlled benchmarks.

### Repository Evolution

- Preserve all historical phase artifacts.
- Add reusable modules for:
  - benchmark specs
  - rollout runner
  - metrics
  - observability logging
  - replay format
- Keep Phase34 as fixed terminal-controller reference.
- Convert the 24-case benchmark into a versioned benchmark file.
- Add held-out deterministic and randomized 2D benchmarks.

### Benchmark Evolution

From:

```text
24-case reduced benchmark
```

To:

```text
crossing24_v1
heldout_grid_2d_v1
randomized_2d_v1
threshold_sensitivity_v1
```

### New Controller Capabilities

- regression-safe upstream search
- explicit terminal controller as reusable module
- guarded controller composition
- controller diagnostics interface

### Simulator Evolution

- keep 2D central gravity
- remove simulator metric ambiguity
- add passive energy/angular-momentum observability
- no 3D yet unless 2D benchmark tooling is stable

### Expected Publications

Possible paper directions:

1. Recoverability-aware benchmark design for orbital insertion.
2. Failure taxonomy for 2D thrust-limited orbital control.
3. Regression-safe controller search under preserved terminal recoverability.

---

## Generation 2: Optimization And Hybrid Control Platform

Timeline: 12-24 months

### Engineering Objective

Introduce trajectory optimization and MPC as first-class planner/controller modules, not one-off scripts.

### Scientific Objective

Compare explicit staged controllers, direct optimization, MPC-lite, and learning-assisted controllers under the same recoverability-aware benchmark.

### Repository Evolution

Add:

```text
spacecraft_ai/planners/trajectory_optimizer.py
spacecraft_ai/planners/mpc.py
spacecraft_ai/controllers/hybrid/
spacecraft_ai/benchmarks/optimization_suites/
```

Standardize optimization outputs:

- planned trajectory
- achieved trajectory
- constraint violations
- solver status
- compute time
- fuel/effort
- recoverability handoff quality

### Benchmark Evolution

Add:

```text
optimization_2d_v1
mpc_lite_2d_v1
robustness_2d_v1
```

### New Controller Capabilities

- direct-shooting planner as reusable baseline
- MPC-lite controller
- fallback from planner to Phase34 terminal controller
- hybrid explicit/optimization controller
- infeasibility labeling

### Simulator Evolution

Still mostly 2D, but with cleaner dynamics modules.

Add optional:

- mass depletion proxy
- actuator saturation
- control delay
- sensor noise

### Expected Publications

1. Hybrid explicit/optimization architecture for recoverable orbital insertion.
2. Recoverability-constrained MPC in a simplified orbital-control sandbox.
3. Failure-aware planner/controller handoff.

---

## Generation 3: Robust Autonomy And Uncertainty

Timeline: 24-36 months

### Engineering Objective

Add uncertainty, faults, and robustness infrastructure.

### Scientific Objective

Study whether controllers can detect loss of recoverability and choose safe degradation modes.

### Repository Evolution

Add:

```text
spacecraft_ai/uncertainty/
spacecraft_ai/faults/
spacecraft_ai/estimation/
spacecraft_ai/safety/
```

### Benchmark Evolution

Add:

```text
uncertain_initial_state_2d_v1
sensor_noise_2d_v1
actuator_fault_2d_v1
delayed_control_2d_v1
recovery_after_disturbance_v1
```

### New Controller Capabilities

- fault detection
- degraded-mode control
- abort/safe-coast policies
- uncertainty-aware recoverability prediction
- ensemble or belief-state planner
- controller confidence reporting

### Simulator Evolution

Add:

- sensor noise
- state estimation
- actuation delay
- thrust degradation
- partial actuator failure
- stochastic perturbations

### Expected Publications

1. Recoverability-aware fault handling for orbital control.
2. Failure detection and safe degradation in autonomous orbital insertion.
3. Uncertainty-sensitive benchmark design for spacecraft autonomy.

---

## Generation 4: Higher-Fidelity Orbital Dynamics

Timeline: 36-48 months

### Engineering Objective

Expand simulation fidelity without breaking the 2D benchmark lineage.

### Scientific Objective

Test whether the recoverability architecture transfers from simplified 2D insertion to more realistic orbital settings.

### Repository Evolution

Add:

```text
spacecraft_ai/sim/dynamics_3d.py
spacecraft_ai/sim/attitude.py
spacecraft_ai/sim/perturbations.py
spacecraft_ai/scenarios/orbit3d.py
spacecraft_ai/benchmarks/3d/
```

### Benchmark Evolution

Add:

```text
two_body_3d_insertion_v1
j2_perturbed_orbit_v1
drag_low_orbit_v1
attitude_coupled_control_v1
```

### New Controller Capabilities

- 3D guidance
- out-of-plane correction
- attitude-aware thrust pointing
- coupled translation/attitude control
- energy/angular momentum/h-plane diagnostics

### Simulator Evolution

Add only in stages:

1. 3D point mass.
2. 3D with orbital elements.
3. J2 perturbation.
4. drag for low orbit.
5. attitude and thrust pointing.
6. mass/fuel model.

Do not add all at once.

### Expected Publications

1. Transfer of recoverability-aware insertion from 2D to 3D.
2. Coupled guidance and attitude constraints in autonomous orbital insertion.
3. Benchmarking failure modes across increasing orbital fidelity.

---

## Generation 5: Mission-Level Autonomy

Timeline: 48-60 months

### Engineering Objective

Move from single insertion task to mission autonomy architecture.

### Scientific Objective

Study how a spacecraft autonomy stack reasons about goals, constraints, uncertainty, faults, and recoverability over long horizons.

### Repository Evolution

Add:

```text
spacecraft_ai/mission/
spacecraft_ai/autonomy/
spacecraft_ai/world_model/
spacecraft_ai/decision/
```

### Benchmark Evolution

Add:

```text
multi_stage_mission_v1
rendezvous_v1
stationkeeping_v1
faulted_mission_recovery_v1
multi_agent_probe_v1
```

### New Controller Capabilities

- mission planner
- task sequencing
- goal switching
- safety veto
- autonomous abort
- target replan
- multi-agent coordination prototype

### Simulator Evolution

Possibly add:

- multiple spacecraft
- relative motion
- rendezvous dynamics
- communication delay
- resource constraints
- long-horizon mission timelines

### Expected Publications

1. Recoverability-aware autonomy stack for multi-stage spacecraft missions.
2. Safety veto and failure labeling in autonomous mission planning.
3. Multi-agent recoverability and non-correlated failure design.

---

# Part 4: Future Bottlenecks

## 4.1 Technical Debt Bottlenecks

### Bottleneck: Phase Scripts Become Unmaintainable

Risk: Every new experiment copies the previous phase script and mutates metrics, controller logic, and output schemas.

Prevention:

- Add reusable rollout runner.
- Add reusable metrics engine.
- Keep new experiment scripts thin.
- Keep old phase scripts frozen as historical reproduction scripts.

### Bottleneck: Multiple Definitions Of Success

Risk: `success`, `CAPTURE`, `LOCK`, `crossing`, `recoverable`, and `Phase34-compatible` drift across experiments.

Prevention:

- Version metric schemas.
- Require benchmark contracts to define primary metrics.
- Add tests for metric functions.
- Keep historical terms documented.

### Bottleneck: Simulator Behavior Drifts Invisibly

Risk: changes to `envs/orbit_env.py` or constants silently alter results.

Prevention:

- Version simulator configurations.
- Add simulator golden tests.
- Record simulator version/hash in experiment metadata.
- Separate dynamics from reward and termination.

### Bottleneck: Data Files Become Unmanageable

Risk: large `.npy`, `.pth`, `.csv`, and generated figures bloat the repo.

Prevention:

- Add dataset manifests.
- Move large generated artifacts to releases or external storage where appropriate.
- Keep small canonical evidence CSVs in git.
- Use `.gitignore` rules for transient generated artifacts.

---

## 4.2 Architecture Bottlenecks

### Bottleneck: No Stable Controller API

Risk: every evaluator needs custom controller calling logic.

Prevention:

- Define `Controller.reset(task)` and `Controller.act(obs)`.
- Add adapters for old controllers.
- Use controller IDs and config serialization.

### Bottleneck: Planner/Controller Confusion

Risk: long-horizon transfer logic and low-level action control remain tangled.

Prevention:

- Introduce planner interface.
- Let planners output phase targets/corridors.
- Let controllers output actions.
- Log handoff explicitly.

### Bottleneck: Metrics Embedded In Controllers

Risk: controllers start optimizing or reporting their own success.

Prevention:

- Controllers may expose diagnostics.
- Evaluator owns metrics.
- Benchmark owns acceptance criteria.

---

## 4.3 Documentation Bottlenecks

### Bottleneck: Too Much Research History

The project log is valuable, but difficult to navigate.

Prevention:

- Maintain an evidence index.
- Maintain current vs historical result map.
- Use manifests for phase artifacts.
- Add “read this first” guide for new collaborators.

### Bottleneck: Old Claims Remain In README

Prevention:

- Keep README focused on current platform.
- Move historical narrative to docs.
- Add status labels:
  - current reference
  - historical
  - diagnostic
  - deprecated
  - speculative

---

## 4.4 Benchmark Bottlenecks

### Bottleneck: 24 Cases Become Overfit

Prevention:

- Add held-out deterministic grid.
- Add seeded randomized benchmark.
- Keep original 24 cases as regression guard, not full truth.

### Bottleneck: Threshold Gaming

Prevention:

- Run threshold sensitivity, but never use it to redefine headline success after the fact.
- Version threshold definitions.
- Report crossing/recoverability/closest approach separately.

### Bottleneck: Learning Baselines Not Comparable

Prevention:

- Evaluate all controllers through the same benchmark runner.
- Report failure labels, not only reward.
- Use fixed splits and manifests.

---

## 4.5 Reproducibility Bottlenecks

### Bottleneck: Environment Drift

Prevention:

- Keep conda environment files.
- Add lock or export snapshots for major releases.
- CI should test Linux CPU baseline.
- Heavy optional dependencies should be isolated.

### Bottleneck: Artifacts Without Provenance

Prevention:

- Every generated artifact directory gets `manifest.json`.
- Include command, git commit, environment, benchmark, seed, controller ID.

---

## 4.6 Scaling Bottlenecks

### Bottleneck: Python Rollouts Become Too Slow

Prevention:

- First optimize architecture and vectorization.
- Then consider JAX/Numba/C++ for dynamics only.
- Do not rewrite the research stack prematurely.

### Bottleneck: Too Many Experiments To Compare

Prevention:

- Use experiment registry.
- Use benchmark result database or standardized CSV/Parquet.
- Use dashboards only after schemas stabilize.

---

# Part 5: Major Future Capabilities

These should not all be implemented now. They are natural future capabilities.

## 5.1 Attitude Dynamics

Logical because real thrust direction depends on attitude. Add only after 3D translation is stable.

Research questions:

- How does thrust pointing delay affect recoverability?
- Can terminal insertion survive attitude constraints?
- How should guidance hand off to attitude control?

## 5.2 3D Orbital Dynamics

Natural next fidelity step after 2D benchmark maturity.

Research questions:

- Does post-cross synchronization generalize to 3D orbital element alignment?
- What is the 3D equivalent of crossing vs recoverability?
- How do inclination and plane error enter recoverability?

## 5.3 Multi-Body Gravity

Not immediate. Add after 3D two-body and perturbations.

Research questions:

- Does recoverability remain a useful concept near weak-stability boundaries?
- How does autonomy reason when target states are time-dependent?

## 5.4 Trajectory Optimization

Already foreshadowed by Phase32.

Capabilities:

- direct shooting
- direct collocation
- constrained optimization
- warm-started explicit controllers
- offline reference generation

## 5.5 MPC

Natural after optimization.

Capabilities:

- receding-horizon transfer correction
- constraint handling
- terminal recoverability cost
- fallback when infeasible

## 5.6 Fault Detection And Recovery

Strong fit with the project philosophy.

Capabilities:

- actuator degradation
- stuck thruster
- sensor corruption
- delayed commands
- safe-coast mode
- abort mode
- recoverability monitor

## 5.7 Mission Planning

Eventually needed for multi-stage autonomy.

Capabilities:

- task sequencing
- target selection
- resource allocation
- autonomous abort/retry
- recovery route planning

## 5.8 Learning-Assisted Planning

Better fit than pure PPO.

Capabilities:

- learned proposal distributions for optimizer
- learned failure classifier
- learned recoverability estimator
- imitation of explicit/optimization plans
- policy as fallback, not sole authority

## 5.9 Autonomy Stack

Long-term stack:

```text
mission manager
  -> planner
    -> guidance
      -> controller
        -> actuator model
  -> monitor
  -> fault manager
  -> recoverability estimator
  -> safety veto
```

This builds directly on the current “crossing is not insertion” lesson.

---

# Part 6: Repository Evolution

## 6.1 Directory Organization

Do not move everything immediately. Add new structure gradually.

Recommended target:

```text
spacecraft_ai_project/
  spacecraft_ai/
    sim/
    scenarios/
    controllers/
    planners/
    benchmarks/
    evaluation/
    metrics/
    logging/
    datasets/
    replays/
    learning/
    visualization/
  experiments/
    legacy_phases/
    gen1_2d_recoverability/
    gen2_hybrid_optimization/
  scripts/
    legacy/
    reproduce/
  analysis/
    phase*/
    gen*/
  docs/
    architecture/
    benchmarks/
    evidence/
    research_workspace/
  tests/
    unit/
    integration/
    regression/
  data/
    manifests/
  models/
    manifests/
```

Existing directories can remain. The key is to stop adding new platform code into one-off phase scripts.

## 6.2 Package Structure

Update `pyproject.toml` eventually to package:

```text
spacecraft_ai
envs
simulator
controller
ppo_orbit
```

But do this incrementally. First create the package with new modules. Then migrate imports.

## 6.3 Reusable APIs

Minimum stable APIs:

```python
Controller
Planner
DynamicsModel
BenchmarkSpec
TaskSpec
EvaluationRunner
MetricBundle
Replay
DatasetManifest
```

Keep APIs small. Avoid abstract class overengineering until there are at least two real implementations.

## 6.4 Coding Standards

Adopt:

- type hints for new platform modules
- dataclasses for state/task/result specs
- pure functions for metrics
- no hidden writes in metrics/controllers
- deterministic seeds for benchmark code
- explicit artifact output directories
- no overwriting historical results

## 6.5 Documentation Strategy

Maintain these document classes:

| Doc Type | Purpose |
|---|---|
| README | platform overview and current status |
| Architecture docs | module boundaries and APIs |
| Benchmark docs | benchmark definitions and versions |
| Evidence docs | artifact manifests and claim maps |
| Research logs | chronological reasoning |
| Experiment reports | generated summaries |
| Roadmaps | future direction |

The most important documentation improvement is a current/historical index.

## 6.6 Benchmark Versioning

Use semantic benchmark IDs:

```text
crossing24_v1
heldout_grid_2d_v1
randomized_2d_v1
uncertainty_2d_v1
optimization_2d_v1
two_body_3d_v1
```

Never silently mutate a benchmark. If cases or thresholds change, create a new version.

## 6.7 Dataset Versioning

Each dataset gets:

```text
dataset.npz
manifest.json
schema.md
splits.json
README.md
```

Dataset ID examples:

```text
phase_controller_dataset_v1
explicit_rollouts_crossing24_v1
optimization_references_2d_v1
fault_recovery_rollouts_v1
```

## 6.8 Experiment Versioning

Each experiment output directory should include:

```text
results.csv
summary.md
manifest.json
plots/
replays/
```

Manifest should include:

- command
- git commit
- environment
- benchmark ID
- controller ID
- metric schema
- seed
- date
- notes

## 6.9 Release Strategy

Use releases for major research platform states:

```text
v0.1 crossing-is-not-insertion evidence release
v0.2 2D benchmark platform
v0.3 observability and held-out benchmark
v0.4 hybrid optimization baseline
v1.0 stable 2D recoverability platform
```

Do not call anything flight-ready.

---

# Part 7: GitHub Roadmap

## GitHub Projects

Use several long-lived project boards:

1. **Platform Core**
2. **Benchmarks And Metrics**
3. **Controllers And Planners**
4. **Learning Baselines**
5. **Simulation Fidelity**
6. **Documentation And Evidence**
7. **Reproducibility And CI**

## Multi-Year Milestones

### Milestone 1: 2D Platform Stabilization

Epics:

- Extract benchmark specs
- Extract metrics engine
- Add observability schema
- Add replay format
- Add held-out benchmark

Representative issues:

- Create `BenchmarkSpec` for crossing24_v1
- Implement passive metrics bundle
- Add replay manifest format
- Add held-out deterministic grid
- Add randomized 2D benchmark seed manifest

### Milestone 2: Regression-Safe Controller Search

Epics:

- Reusable Phase34 terminal controller
- Upstream planner search
- Regression guard expansion
- Failure taxonomy

Issues:

- Extract Phase34 post-cross controller class
- Add protected regression case suite
- Add failure label unit tests
- Add search cancellation criteria
- Add effort metrics to search outputs

### Milestone 3: Hybrid Optimization

Epics:

- Direct shooting module
- MPC-lite module
- Optimization benchmark
- Solver diagnostics

Issues:

- Implement optimizer result schema
- Add direct-shooting baseline runner
- Add MPC-lite controller interface
- Record solver infeasibility labels
- Compare explicit vs optimization vs hybrid

### Milestone 4: Robustness And Faults

Epics:

- Noise models
- Fault injection
- Recovery benchmark
- Safety monitor

Issues:

- Add sensor noise scenario
- Add thrust degradation model
- Add recoverability monitor
- Add safe-coast fallback controller
- Add faulted benchmark suite

### Milestone 5: 3D And Attitude

Epics:

- 3D state model
- 3D dynamics
- orbital element metrics
- attitude dynamics
- thrust pointing constraints

Issues:

- Add `State3D`
- Add 3D two-body dynamics
- Add orbital element diagnostics
- Add attitude state prototype
- Add coupled thrust-direction test

### Milestone 6: Mission Autonomy

Epics:

- Mission task model
- Mission planner
- multi-stage benchmarks
- autonomy monitor
- multi-agent prototype

Issues:

- Define mission task schema
- Add task sequencing benchmark
- Add abort/retry planner
- Add autonomy event log
- Add two-spacecraft relative-motion prototype

---

# Part 8: Research Philosophy

## 8.1 Inferred Philosophy

The project’s underlying philosophy is:

```text
Do not trust apparent progress.
A milestone is not success unless the next dynamical regime is recoverable.
Failures are data.
Benchmarks must expose false success.
Explicit structure matters.
Learning is useful, but not magic.
Survival and recoverability are more important than reward.
```

That is a strong research philosophy.

The paper’s core phrase, “crossing is not insertion,” is really a broader engineering principle:

```text
event achievement is not operational viability
```

That principle can scale beyond 2D orbital insertion into autonomy, fault handling, mission planning, and multi-agent systems.

## 8.2 Strengths

### Strong Failure Discipline

The repo keeps negative results. This is rare and valuable.

### Strong Metric Skepticism

It distinguishes:

- crossing
- recoverability
- closest approach
- overspeed
- instability
- simulator success

This is the right mindset for autonomy research.

### Strong Architecture Instinct

Even before formal architecture, the research has converged toward staged control:

```text
transfer -> crossing -> synchronization -> capture -> lock
```

That is a real architecture insight.

### Strong Reproducibility Orientation

The artifact manifest, benchmark contract, conda environments, CI, and regression guard show serious reproducibility intent.

## 8.3 Weaknesses

### Too Much Chronological Coupling

The repo is organized by research history more than platform boundaries. That is natural, but it will not scale.

### Too Much Logic In Scripts

The scripts are scientifically valuable but architecturally overloaded.

### Learning Infrastructure Is Messy

The PPO/IL history is large, artifact-heavy, and not clearly separated from current claims.

### Simulator Fidelity Is Still Low

The project is not close to real spacecraft autonomy. It should keep saying that.

### Benchmark Is Small

The 24-case benchmark is useful but too small for long-term generalization claims.

## 8.4 Hidden Assumptions

1. The 2D recoverability concept will transfer to higher fidelity.
2. Explicit staged controllers will remain interpretable enough as complexity grows.
3. Failure labels are stable across benchmark expansions.
4. Phase34-style terminal recovery is a good architectural primitive.
5. Learning will become useful after explicit structure is better defined.

These are plausible, not proven.

## 8.5 Risks

### Risk: The Project Becomes A Pile Of Experiments

Mitigation: introduce platform modules now.

### Risk: The Project Adds Realism Too Fast

Mitigation: benchmark and observability must precede simulator complexity.

### Risk: Negative Results Become Buried

Mitigation: maintain artifact manifest and evidence map.

### Risk: Learning Baselines Consume Too Much Time

Mitigation: use learning as a diagnostic or assistant until it beats explicit baselines under the same benchmark.

### Risk: The Project Overclaims

Mitigation: keep the current honesty standard permanently.

## 8.6 Opportunities

### Opportunity: Recoverability As A General Autonomy Concept

This could become the central research identity.

### Opportunity: Failure-Aware Benchmarking

The repository can become a benchmark platform, not only a controller repo.

### Opportunity: Hybrid Explicit/Optimization/Learning Control

The project is naturally positioned for hybrid methods.

### Opportunity: Autonomy Safety Architecture

The idea of refusing false progress is directly relevant to fault recovery, mission autonomy, and safety veto design.

---

# Final Strategic Recommendation

The next 3-5 years should be guided by one architectural principle:

```text
Preserve the research history, but stop making history the architecture.
```

The current phase scripts and artifacts should remain as evidence. New work should gradually extract reusable platform layers:

```text
simulator
task/scenario
controller
planner
benchmark
evaluator
metrics
logging
dataset
replay
visualization
experiment manager
```

The first major engineering objective should not be 3D, MPC, or more PPO. It should be a stable 2D platform with versioned benchmarks, reusable metrics, replayable results, and protected regression cases.

The highest-value path is:

1. Formalize the 2D benchmark/evaluation/metrics stack.
2. Extract Phase34 terminal behavior into reusable controller architecture.
3. Add held-out and randomized 2D benchmarks.
4. Add optimization/MPC as comparable planners.
5. Add uncertainty and fault recovery.
6. Only then expand to 3D, attitude, perturbations, and mission autonomy.

The project’s long-term identity should not be “AI controls spacecraft.” That is too broad and too easy to overclaim.

A better identity is:

```text
Recoverability-aware autonomous spacecraft control:
architectures, benchmarks, and failure-aware evaluation for orbital autonomy.
```

That identity respects the current evidence and gives the repository enough room to grow for five years.
