# Hardware, Vision, Sensor, And Robotics Expansion Strategy

## Purpose

This document extends `docs/roadmaps/long_term_research_platform_strategy.md`.

The original long-term strategy treats `spacecraft-ai-controller` as the seed of a recoverability-aware autonomous spacecraft control research platform. This second strategy expands that direction toward hardware-aware autonomy, vision, sensing, contact-rich robotics, embedded constraints, and sim-to-real thinking.

This document does not claim that the current repository already supports hardware, perception, robotic insertion, or flight-like autonomy. It defines a gradual path for adding those ideas without replacing the existing spacecraft-control foundation.

The core principle is:

```text
Do not turn the repository into a robotics repo.
Turn it into an autonomy research platform whose spacecraft-control lessons can transfer into robotics, perception, sensing, and hardware-constrained execution.
```

The existing project identity should remain:

```text
Recoverability-aware autonomous control under physical constraints.
```

The expansion adds:

```text
perception -> estimation -> sensing -> contact -> hardware constraints -> real-world execution risk
```

---

# 1. Why Hardware, Vision, And Sensors Belong In This Project

## 1.1 The Current Lesson Generalizes

The current paper's central lesson is:

```text
Crossing is not insertion.
```

In the current 2D orbital-control setting, a target-radius crossing is only a geometric event. It does not guarantee recoverability, synchronization, CAPTURE, LOCK, or survival.

That lesson is not limited to orbital control. It generalizes directly to perception and robotics:

```text
Perception success is not task success.
```

Examples:

| Perception Or Intermediate Event | Why It Is Not Enough |
|---|---|
| Detecting a docking target | The spacecraft may still have wrong relative velocity, attitude, or timing. |
| Estimating a target pose | Pose may be noisy, stale, biased, or dynamically unrecoverable. |
| Seeing the hole for plug insertion | The robot may still be misaligned, tilted, or blocked by contact/friction. |
| Reaching a contact point | Contact may be unstable, off-axis, or unrecoverable without retreat. |
| Computing a valid plan | The physical system may fail under latency, actuation limits, or sensor drift. |
| Entering a nominal state | The system may not be able to maintain it under disturbances. |

The broader principle is:

```text
An observed milestone is not the same as physical recoverability.
```

This is exactly why hardware, vision, and sensors belong in the long-term platform. Real autonomy does not act on perfect simulator state. It acts on partial, noisy, delayed, sometimes wrong measurements. It must decide when the current estimate is trustworthy enough to continue and when the system should slow down, retreat, re-observe, switch feedback mode, or abort.

## 1.2 Why Spacecraft Autonomy Needs Sensors

Any realistic spacecraft autonomy stack eventually depends on sensing:

- Cameras for target recognition, docking, terrain-relative navigation, inspection, and rendezvous.
- Star trackers for attitude determination.
- IMUs for angular rates and inertial propagation.
- Sun sensors, horizon sensors, and magnetometers for coarse attitude awareness.
- LiDAR or depth sensors for relative navigation and docking.
- Radar for range/range-rate in some rendezvous settings.
- Force/torque sensing or contact inference for docking, capture, berthing, robotic servicing, and assembly.
- Onboard compute for running perception, estimation, planning, and control under latency and power limits.

Even if the repository starts with a simplified 2D simulator, the long-term autonomy problem is sensor-mediated:

```text
world -> sensors -> estimates -> decisions -> actions -> physical response -> new sensor data
```

The current project already studies the difference between a geometric event and a recoverable dynamic state. Hardware expands that distinction:

```text
detected event -> estimated state -> control decision -> physical outcome -> recoverability
```

## 1.3 Why Robotics Belongs Without Replacing Spacecraft Control

Robotic plug insertion is not spacecraft orbital insertion. The physics are different. The hardware is different. The time scales are different. But the autonomy structure is similar:

- approach a target
- align to a narrow valid region
- cross a critical boundary
- handle post-boundary dynamics
- detect failure
- correct or retreat
- reach a locked/stable final state

In spacecraft:

```text
approach orbit -> crossing -> post-cross synchronization -> recoverability basin -> CAPTURE/LOCK
```

In plug insertion:

```text
approach socket -> align plug -> contact/entry -> post-contact correction -> seated insertion/lock
```

The repository should use robotics as a transfer laboratory for autonomy ideas, not as a replacement subject. Plug insertion can provide real-world lessons about perception error, contact ambiguity, force feedback, small corrective actions, failure labeling, and hardware constraints.

Those lessons can later feed back into spacecraft autonomy:

- docking
- robotic servicing
- sample capture
- on-orbit assembly
- lander foot contact
- manipulator-guided insertion
- fault-aware physical interaction

## 1.4 Why Edge AI And Chips Eventually Matter

Autonomy algorithms do not run in a vacuum. They eventually run on hardware with constraints:

- latency
- memory
- compute throughput
- power
- thermal limits
- fault tolerance
- quantized inference
- real-time deadlines
- communication limits

The repository should not become an electrical engineering project immediately. But it should eventually track hardware-aware fields:

- inference time
- control-loop rate
- model size
- memory footprint
- latency budget
- sensor timestamp delay
- dropped-frame rate
- degraded compute mode

The research question is not "which chip is best?" The research question is:

```text
How does autonomy design change when perception, planning, and control must run under real hardware constraints?
```

---

# 2. Connection Between Spacecraft Control And Robotic Plug Insertion

## 2.1 Shared Task Structure

The two domains share a staged recoverability pattern.

| Structure | Spacecraft Orbital Control | Robotic Plug Insertion |
|---|---|---|
| Approach | Transfer toward target orbit or rendezvous region | Move plug toward socket region |
| Alignment | Match radius, radial velocity, tangential velocity, phase | Align plug pose, angle, depth, lateral offset |
| Critical event | Target-radius crossing, docking gate, capture boundary | First contact, chamfer entry, partial insertion |
| Post-event stabilization | Reduce radial and tangential mismatch after crossing | Use contact/force feedback to correct after entry |
| Recoverability | State enters basin where controller can survive | Contact state allows insertion without jamming |
| Failure detection | Overspeed, instability, bad handoff, no crossing | misalignment, jam, slip, excessive force, wrong contact |
| Retry/correction | Adjust trajectory, coast, replan, safe mode | withdraw, reorient, spiral/search, compliance control |
| Final lock | CAPTURE/LOCK or stable orbit | seated plug, latch, stable electrical/mechanical connection |

The important commonality is that the boundary event is not enough:

- Crossing target radius is not insertion.
- Touching the socket is not insertion.
- Seeing a pose is not insertion.
- Entering contact is not insertion.

Both domains require post-boundary correction.

## 2.2 Where Plug Insertion Can Inform Spacecraft Research

Robotic plug insertion can teach the spacecraft project about physical interaction and sensing in ways that a 2D orbital simulator cannot.

### Contact-Rich Failure Modes

Plug insertion has visible failure classes:

- no contact
- wrong contact
- edge contact
- side contact
- tilted entry
- jammed insertion
- partial insertion
- excessive force
- slip
- compliant recovery

This can inspire a more mature failure taxonomy for spacecraft interaction tasks:

- no crossing
- bad crossing
- bad handoff
- overspeed
- unstable capture
- wrong relative pose
- unsafe contact
- failed docking latch
- failed recovery

### Multimodal Feedback

Plug insertion naturally combines:

- camera data
- pose estimation
- force/contact feedback
- robot joint state
- action history
- tactile or compliance cues

This can inform future spacecraft autonomy with:

- camera plus IMU fusion
- vision plus range/range-rate
- pose plus contact sensors
- relative navigation plus docking force cues
- estimator confidence plus control mode selection

### Small Corrective Actions

Current spacecraft experiments focus on long-horizon trajectory shaping and post-cross synchronization. Plug insertion emphasizes small, local, corrective actions under contact.

That is relevant to:

- docking final approach
- berthing
- on-orbit servicing
- manipulator contact
- capture mechanisms
- landing contact stabilization

### Failure-Case Analysis

The current repository already values negative results. Plug insertion can deepen this with real failure-case labeling:

```text
not just success/failure, but why the physical execution failed
```

This should become a shared platform idea.

## 2.3 Where Spacecraft Research Can Inform Plug Insertion

The spacecraft project contributes a useful discipline to robotics:

```text
Do not confuse intermediate geometry with recoverable task completion.
```

For plug insertion:

- visual alignment is not enough
- first contact is not enough
- partial entry is not enough
- low pose error is not enough
- low imitation loss is not enough

A recoverability-aware robotics benchmark would ask:

- Can the system recover from off-axis contact?
- Can it detect a jam early?
- Can it decide to retreat before damaging hardware?
- Can it distinguish useful contact from failure contact?
- Can it complete insertion after perception becomes unreliable?

This is a direct transfer of the spacecraft project's evaluation philosophy.

## 2.4 Shared Abstraction

A general abstraction:

```text
approach -> boundary event -> post-boundary synchronization -> recoverable execution -> stable final state
```

For spacecraft:

```text
transfer -> radius crossing -> post-cross sync -> recoverability basin -> CAPTURE/LOCK
```

For plug insertion:

```text
approach -> contact/entry -> post-contact correction -> insertion basin -> seated/locked
```

For docking:

```text
rendezvous -> docking corridor entry -> relative pose/velocity sync -> capture basin -> hard dock
```

For assembly:

```text
approach -> contact -> compliant alignment -> stable mating -> locked assembly
```

The platform should eventually support this abstraction across domains.

---

# 3. New Architecture Modules Needed

This section extends the architecture from the previous roadmap.

Target expansion:

```text
spacecraft_ai/
  perception/
  sensors/
  estimation/
  calibration/
  contact/
  hardware/
  embedded/
  sim2real/
  safety/
  faults/
```

These modules should be added gradually. They should not replace the existing simulator, controllers, benchmarks, or phase artifacts.

## 3.1 Perception Module

### Purpose

Own image/depth observation processing and perception outputs.

Examples:

- object detection
- keypoint detection
- fiducial detection
- target segmentation
- plug/socket localization
- docking target localization
- pose hypotheses

### Why It Belongs

Future autonomy will not receive perfect state. It will receive sensor data and estimates. Perception is the first step between raw data and control.

### Depends On

- datasets
- sensor data formats
- calibration metadata
- model definitions
- logging schema

### Should Not Depend On

- controller internals
- benchmark success criteria
- mission planner internals

### Enables

- pose-estimation benchmarks
- vision-conditioned control
- perception failure analysis
- uncertainty-aware control
- sim-to-real perception comparison

## 3.2 Sensor Models Module

### Purpose

Represent sensors and sensor imperfections.

Examples:

- camera model
- depth sensor model
- LiDAR/range model
- IMU model
- star tracker model
- force/torque sensor model
- contact binary sensor
- delayed sensor stream

### Why It Belongs

Controllers should not assume perfect state. Sensor models allow the same physical task to be evaluated under clean, noisy, delayed, failed, or partial observation.

### Depends On

- simulator state
- calibration parameters
- noise distributions
- timestamp/latency utilities

### Should Not Depend On

- controller classes
- learning algorithms
- plotting

### Enables

- noisy-state benchmarks
- estimator tests
- fault injection
- latency-stressed control
- hardware-aware simulation

## 3.3 State Estimation Module

### Purpose

Convert sensor histories into estimated state and uncertainty.

Examples:

- pose estimator
- velocity estimator
- relative navigation filter
- visual-inertial estimator prototype
- contact-state estimator
- confidence tracker

### Why It Belongs

Real autonomy controls estimated state, not true state. The platform needs to distinguish:

```text
true simulator state
sensor observation
estimated state
controller belief
```

### Depends On

- sensor outputs
- calibration
- state representation
- timestamps

### Should Not Depend On

- benchmark scoring logic
- controller implementation details

### Enables

- belief-state control
- estimator failure detection
- uncertainty-aware recoverability
- perception-to-control handoff analysis

## 3.4 Calibration Module

### Purpose

Represent transformations and calibration metadata.

Examples:

- camera intrinsics
- camera extrinsics
- robot hand-eye calibration
- sensor-to-body transform
- spacecraft body-to-camera transform
- force sensor zero/bias calibration

### Why It Belongs

Perception failures are often calibration failures. A future hardware-aware platform should treat calibration as an explicit artifact, not a hidden assumption.

### Depends On

- coordinate-frame definitions
- sensor model definitions
- metadata files

### Should Not Depend On

- controllers
- planners
- specific benchmarks

### Enables

- calibration sensitivity tests
- pose bias experiments
- sensor misalignment fault injection
- sim-to-real debugging

## 3.5 Contact Module

### Purpose

Represent contact events, contact states, and contact failure labels.

Examples:

- no contact
- expected contact
- edge contact
- sliding contact
- jam
- excessive force
- partial insertion
- seated insertion
- contact lost

### Why It Belongs

Robotic plug insertion and future spacecraft docking/assembly require post-contact recoverability. Contact is the robotics equivalent of crossing: an important event, not completion.

### Depends On

- simulated or logged force/contact signals
- state/action history
- task geometry

### Should Not Depend On

- vision model internals
- learning algorithm internals

### Enables

- contact-rich insertion benchmark
- failure taxonomy transfer
- contact-aware controllers
- recoverability prediction after contact

## 3.6 Hardware Abstraction Module

### Purpose

Represent physical devices through interfaces without binding the whole repo to a specific robot, camera, or board.

Examples:

- camera source interface
- robot arm interface
- force sensor interface
- actuator command interface
- embedded inference device interface
- simulated device adapter
- logged replay adapter

### Why It Belongs

Hardware will eventually matter, but the repo should not become hardware-specific too early. A hardware abstraction layer allows gradual integration.

### Depends On

- sensor interfaces
- action/command schemas
- timestamps
- logging

### Should Not Depend On

- spacecraft-specific benchmarks
- plug-insertion-specific code
- model training code

### Enables

- hardware-in-the-loop experiments
- replay-vs-live comparison
- device swapping
- simulation/live parity checks

## 3.7 Latency Models Module

### Purpose

Model delays in sensing, estimation, planning, inference, and actuation.

Examples:

- fixed sensor delay
- random frame delay
- dropped frames
- inference latency
- control command delay
- asynchronous sensor streams

### Why It Belongs

A controller that works with perfect synchronous state may fail when perception is stale by 50-200 ms. Spacecraft and robotics both face latency constraints.

### Depends On

- sensor stream formats
- timestamps
- evaluation runner

### Should Not Depend On

- controller internals
- specific hardware vendors

### Enables

- latency-stressed benchmarks
- real-time feasibility analysis
- edge inference tradeoff studies
- robust control under stale state

## 3.8 Embedded Inference Module

### Purpose

Track model constraints relevant to onboard/edge deployment.

Examples:

- model size
- inference time
- memory footprint
- quantized model metrics
- device class
- control-loop budget

### Why It Belongs

Autonomy algorithms must eventually respect hardware constraints. This module keeps those constraints visible without forcing early hardware deployment.

### Depends On

- model artifacts
- benchmark metadata
- profiling utilities
- logging schema

### Should Not Depend On

- simulator internals
- mission claims

### Enables

- hardware-budget benchmarks
- latency-aware model selection
- quantization experiments
- edge feasibility comparisons

## 3.9 Hardware-In-The-Loop Interface

### Purpose

Allow a real or simulated device to sit inside the evaluation loop.

Examples:

- live camera feed with simulated controller
- simulated plant with real perception model
- real robot replaying a scripted insertion benchmark
- controller reading delayed hardware sensor logs

### Why It Belongs

Hardware-in-the-loop is the bridge between offline datasets and real-world autonomy. It should arrive only after replay, logging, and benchmark interfaces are stable.

### Depends On

- hardware abstraction
- logger
- evaluator
- replay system
- safety monitor

### Should Not Depend On

- a specific lab setup
- a specific robot brand
- spacecraft benchmark internals

### Enables

- live replay validation
- sim-to-real comparison
- hardware latency measurement
- real failure-case capture

## 3.10 Sim-To-Real Evaluation Module

### Purpose

Compare simulation predictions, offline logs, and hardware outcomes.

Examples:

- simulated success vs hardware success
- pose error distribution shift
- force/contact mismatch
- failure-label mismatch
- latency effect
- controller robustness gap

### Why It Belongs

The platform should eventually ask:

```text
Which parts of the autonomy stack survive contact with the real world?
```

### Depends On

- benchmark results
- hardware logs
- replay system
- metrics
- dataset manifests

### Should Not Depend On

- live device control
- paper-specific claims

### Enables

- transfer studies
- sim-to-real failure analysis
- hardware-aware benchmark design

## 3.11 Safety Monitor And Fault Manager

### Purpose

Detect unsafe, unrecoverable, or unreliable conditions and trigger fallback behavior.

Examples:

- excessive force
- unstable velocity
- perception confidence collapse
- stale sensor data
- actuator saturation
- repeated failed insertion attempts
- estimator divergence
- overspeed
- out-of-range state

### Why It Belongs

The current project already values refusal of false progress. The safety monitor is the architectural form of that philosophy.

### Depends On

- metrics
- sensor health
- estimator confidence
- task state
- controller diagnostics

### Should Not Depend On

- a single controller implementation
- paper-specific result logic

### Enables

- abort modes
- safe-coast modes
- retry policies
- failure-aware autonomy
- hardware-safe experimentation

---

# 4. Sensor-Aware Autonomy Stack

## 4.1 Future Stack

A future sensor-aware autonomy stack should look like:

```text
              mission / task manager
                        |
                        v
                    planner
                        |
                        v
              desired mode / target / corridor
                        |
                        v
sensor data -> perception -> state estimator -> controller -> hardware interface -> plant
     |             |             |              |              |              |
     |             |             |              |              |              v
     |             |             |              |              |         physical world
     |             |             |              |              |
     |             |             |              |              v
     |             |             |              |        actuator/sensor logs
     |             |             |              |
     |             |             |              v
     |             |             |        controller diagnostics
     |             |             |
     |             |             v
     |             |       estimator confidence
     |             |
     |             v
     |       perception confidence
     |
     v
raw sensor logs

all streams -> logger / replay / metrics / safety monitor / fault manager
```

## 4.2 Data Flow

### Step 1: Sensors Produce Observations

Sensors may include:

- camera frames
- depth images
- IMU readings
- force/torque readings
- encoder/joint state
- range measurements
- star tracker estimates
- simulated perfect state for baseline comparison

Output:

```text
SensorObservation
```

### Step 2: Perception Extracts Features Or Measurements

Perception may output:

- target detected/not detected
- keypoints
- bounding boxes
- segmentation masks
- pose hypotheses
- confidence scores
- uncertainty estimates

Output:

```text
PerceptionResult
```

### Step 3: Estimator Produces Belief State

The estimator fuses perception and sensors into:

- estimated position/pose
- estimated velocity
- contact state
- covariance/confidence
- health flags

Output:

```text
EstimatedState
```

### Step 4: Planner Chooses Mode Or Goal

The planner chooses:

- approach
- alignment
- contact search
- post-contact correction
- retreat
- retry
- safe mode
- terminal insertion/lock

Output:

```text
ControlObjective
```

### Step 5: Controller Produces Action

The controller maps estimated state and objective into action:

- thrust command
- robot velocity command
- end-effector displacement
- compliance command
- safe stop

Output:

```text
ActionCommand
```

### Step 6: Hardware Interface Sends Command

The hardware layer translates abstract action into device-specific command.

Output:

```text
DeviceCommand
```

### Step 7: Safety Monitor Can Veto

The safety monitor can:

- allow command
- clip command
- slow down
- request re-observation
- switch controller
- abort
- trigger safe mode

This is the hardware-aware version of refusing false progress.

## 4.3 Logger As A First-Class System

Every step should be logged:

- raw observations
- perception outputs
- estimated states
- action commands
- device commands
- safety decisions
- task events
- failure labels
- timing/latency

Without this, hardware and perception experiments will become irreproducible.

---

# 5. Vision Roadmap

Vision should enter gradually. The repository should not jump directly to real cameras, ROS, or large vision models.

## Stage 1: Offline Image/Pose Dataset Notes

### Objective

Define what visual data would mean for the platform without implementing live perception.

### Work

- Write a perception architecture note.
- Define `ImageObservation`, `PoseEstimate`, and `PerceptionResult` schemas.
- Document possible dataset sources:
  - plug insertion images
  - simulated target renders
  - fiducial marker images
  - docking target mockups

### Artifact

```text
docs/architecture/perception_architecture.md
```

### Success

The project has clear vocabulary for vision data before any model training starts.

## Stage 2: Simulated Camera Observations

### Objective

Create simple synthetic observations from simulator state.

### Work

- Start with fake camera outputs:
  - target visible/not visible
  - noisy 2D keypoints
  - noisy relative pose
  - confidence score
- Do not render photorealistic images.

### Artifact

```text
spacecraft_ai/sensors/camera_model.py
analysis/vision_stage2_synthetic_observations/
```

### Success

Controllers/evaluators can run with noisy observations rather than true state.

## Stage 3: Pose-Estimation Benchmark

### Objective

Benchmark pose estimation separately from task control.

### Work

- Define pose error metrics.
- Define visibility/failure labels.
- Add synthetic pose datasets.
- Evaluate how pose error distribution changes with noise, occlusion, or calibration bias.

### Artifact

```text
docs/benchmarks/pose_estimation_v1.md
analysis/pose_estimation_v1/
```

### Success

The repo can distinguish perception performance from task performance.

## Stage 4: Vision-Conditioned Controller Inputs

### Objective

Allow controllers to consume estimated state rather than ground truth.

### Work

- Add evaluator option:

```text
true_state_controller
estimated_state_controller
```

- Compare results under perfect state, noisy state, and biased pose estimate.

### Artifact

```text
analysis/vision_conditioned_control_v1/
```

### Success

The benchmark reports how perception error affects crossing/recoverability or insertion success.

## Stage 5: Failure-Aware Perception

### Objective

Perception should report uncertainty and failure modes, not just a pose.

### Work

- Add perception confidence.
- Add stale/invalid estimate flags.
- Add occlusion or detection-loss events.
- Add safety monitor behavior when perception becomes unreliable.

### Artifact

```text
analysis/failure_aware_perception_v1/
```

### Success

The system can detect when vision should not be trusted.

## Stage 6: Sensor Fusion

### Objective

Fuse vision with another signal.

Possible pairs:

- vision + IMU
- vision + range
- vision + contact
- pose estimate + action history

### Artifact

```text
spacecraft_ai/estimation/
analysis/sensor_fusion_v1/
```

### Success

The benchmark can answer whether fusion improves task recoverability, not only pose error.

## Stage 7: Hardware-In-The-Loop Prototype

### Objective

Run a limited, safe hardware or recorded-hardware loop.

This should happen only after:

- replay schema is stable
- logging schema exists
- safety monitor exists
- offline perception benchmark exists

### Artifact

```text
analysis/hil_prototype_v1/
```

### Success

The repo can compare simulated, replayed, and live or logged sensor behavior under the same evaluator.

---

# 6. Sensor And Contact Roadmap

## 6.1 Why Contact Matters

Contact is the robotics analogue of crossing.

In spacecraft orbit insertion:

```text
crossing target radius is not insertion
```

In plug insertion:

```text
contact is not insertion
```

A robot may touch the socket but still be:

- tilted
- laterally offset
- jammed
- slipping
- pushing against the wrong surface
- applying excessive force
- unable to recover without retreat

The repository should treat contact as a boundary event requiring post-contact recoverability.

## 6.2 Contact-State Taxonomy

Initial taxonomy:

| Contact State | Meaning |
|---|---|
| `no_contact` | Plug has not touched socket or environment. |
| `expected_contact` | Contact occurs in expected region. |
| `edge_contact` | Plug touches rim/chamfer/edge. |
| `side_contact` | Lateral misalignment creates side force. |
| `tilted_contact` | Orientation error dominates contact. |
| `sliding_contact` | Contact exists but motion is sliding along surface. |
| `jammed_contact` | Motion stalls while force rises. |
| `partial_insertion` | Plug enters but is not seated. |
| `recoverable_contact` | Contact state can be corrected without full reset. |
| `unrecoverable_contact` | Must retreat/reset to continue safely. |
| `seated` | Final insertion state achieved. |

This taxonomy can later inspire docking/contact labels.

## 6.3 Contact Detection

Signals:

- force magnitude
- force direction
- torque
- end-effector velocity drop
- commanded vs actual motion mismatch
- sudden acceleration change
- visual contact cue
- tactile/contact binary sensor

The platform should distinguish:

```text
contact detected
contact classified
contact recoverable
task completed
```

## 6.4 Force Direction Ambiguity

Force feedback is not automatically clear.

Example ambiguity:

- lateral force may mean the plug is left of the hole
- or tilted
- or friction is binding
- or calibration is biased
- or the robot is contacting a different surface

Research implication:

```text
force/contact sensing should be treated as evidence, not truth
```

## 6.5 Failed Alignment

Failed alignment may occur even when vision says the pose is good.

Causes:

- pose estimation bias
- calibration error
- compliance
- object tolerance
- unmodeled geometry
- robot control latency
- socket movement

This directly parallels spacecraft cases where closest approach improves but crossing still does not occur.

## 6.6 Recoverability After Contact

Key question:

```text
After contact, is the system still in a state from which controlled insertion is possible?
```

Metrics:

- force magnitude
- force direction stability
- insertion depth
- lateral error estimate
- tilt estimate
- motion progress
- action effort
- number of corrective attempts
- retreat required

## 6.7 Small Corrective Actions

Plug insertion emphasizes local correction:

- small lateral shifts
- small rotations
- compliance adjustment
- spiral search
- force-guided centering
- retreat and retry

This can inform future spacecraft docking/assembly:

- small relative pose corrections
- soft capture
- latch alignment
- contact damping
- final approach correction

## 6.8 Failure-Case Generation

The platform should eventually generate contact-rich failure cases:

- pose offset sweeps
- angle offset sweeps
- friction variation
- compliance variation
- force threshold changes
- sensor noise
- camera occlusion
- delayed feedback

Each should produce failure labels, not just success/failure.

---

# 7. Embedded / Chip / Edge AI Roadmap

## 7.1 Framing

The project should not become chip-design work in the near term.

The right framing is:

```text
Autonomy algorithms must eventually respect hardware constraints.
```

This includes:

- latency
- compute budget
- memory budget
- power budget
- model size
- quantization
- real-time control-loop rate
- fault tolerance

## 7.2 Stage 1: Hardware-Aware Logging Fields

Add fields before adding hardware:

- inference_time_ms
- control_loop_period_ms
- sensor_latency_ms
- dropped_frame_count
- model_parameter_count
- model_size_mb
- estimated_memory_mb
- device_class
- quantized

Artifact:

```text
docs/logging_schema_hardware_v1.md
```

## 7.3 Stage 2: Latency And Budget Simulation

Simulate:

- fixed inference delay
- variable inference delay
- stale observations
- slow control loop
- missing frames

Research question:

```text
How much latency can the controller tolerate before recoverability collapses?
```

## 7.4 Stage 3: Model Profiling

For learned perception or policy models, record:

- CPU inference time
- GPU inference time if available
- batch size
- model size
- memory usage approximation
- quantized vs non-quantized metrics

No expensive hardware required.

## 7.5 Stage 4: Quantization And Compression

Later, evaluate:

- float32 vs float16
- int8 quantization
- pruning
- distillation
- small policy networks
- small perception networks

But only after task metrics are stable.

## 7.6 Stage 5: Edge Prototype

Eventually, test a low-cost edge device or embedded-like compute budget.

The platform should care about:

- whether autonomy still meets timing
- whether perception confidence changes
- whether delayed control becomes unsafe
- whether safety monitor detects compute degradation

## 7.7 Hardware Acceleration As Research Variable

Hardware acceleration should be treated as a variable:

```text
same autonomy stack, different compute constraints
```

Not:

```text
new chip means new scientific claim
```

---

# 8. Benchmark Evolution

The current crossing/recoverability benchmark should inspire future multimodal benchmarks.

Current structure:

```text
geometric event -> recoverability -> survival
```

Future structure:

```text
perception event -> estimated state -> physical interaction -> recoverability -> task success
```

## 8.1 Perception-Only Benchmark

Purpose:

Evaluate perception outputs independent of control.

Metrics:

- detection rate
- false positive rate
- pose error
- confidence calibration
- latency
- failure labels

Acceptance:

Perception benchmark must not claim task success.

## 8.2 Pose-Estimation Benchmark

Purpose:

Measure pose quality and uncertainty.

Metrics:

- translation error
- rotation error
- depth error
- covariance calibration
- bias under occlusion
- calibration sensitivity

Connection to current project:

Pose accuracy is like closest approach: useful but not sufficient.

## 8.3 Noisy-State Control Benchmark

Purpose:

Evaluate control under imperfect estimated state.

Metrics:

- crossing/recoverability under noise
- insertion success under pose noise
- failure labels
- safety aborts
- control effort

Connection:

This is the first bridge from perfect simulator state to sensor-mediated autonomy.

## 8.4 Latency-Stressed Control Benchmark

Purpose:

Evaluate control with delayed sensing, delayed inference, or delayed actuation.

Metrics:

- success degradation vs latency
- recoverability collapse threshold
- instability count
- safety intervention count
- stale-estimate failure labels

## 8.5 Contact-Rich Insertion Benchmark

Purpose:

Evaluate insertion under contact and force feedback.

Metrics:

- first contact
- contact classification
- recoverable contact
- seated insertion
- excessive force
- jam
- retry count
- contact-to-lock time

Connection:

This is the plug-insertion analogue of crossing/recoverability.

## 8.6 Sensor-Failure Benchmark

Purpose:

Evaluate degraded autonomy.

Scenarios:

- camera dropout
- pose estimate bias
- IMU drift
- force sensor bias
- delayed data
- corrupted confidence

Metrics:

- detection of failure
- safe fallback
- task completion if possible
- abort if necessary
- false continuation under bad data

## 8.7 Sim-To-Real Transfer Benchmark

Purpose:

Compare simulation, replay, and hardware or logged hardware data.

Metrics:

- sim-predicted success vs real/logged success
- failure-label mismatch
- sensor distribution shift
- contact force mismatch
- controller action mismatch

## 8.8 Hardware-Budget Benchmark

Purpose:

Evaluate autonomy under compute constraints.

Metrics:

- inference time
- loop rate
- memory footprint
- model size
- latency-induced failures
- power proxy if available

## 8.9 Unified Benchmark Principle

Every benchmark should separate:

```text
measurement quality
state estimate quality
control outcome
recoverability
safety
physical completion
```

This prevents the project from replacing one false metric with another.

---

# 9. Research Questions

## 9.1 Perception And Recoverability

- When does perception accuracy actually improve task recoverability?
- Is lower pose error always useful, or only near certain control regimes?
- Can a controller tolerate biased pose if uncertainty is correctly estimated?
- What perception failures are dangerous because the controller cannot detect them?
- Can perception confidence predict recoverability better than raw pose error?

## 9.2 Sensor Fusion

- Can a system detect that vision is unreliable and switch to contact feedback?
- When should a controller trust force more than vision?
- When does IMU propagation help versus amplify drift?
- Can multimodal history predict failure earlier than any single sensor?
- Which sensor combinations improve recoverability rather than just accuracy?

## 9.3 Contact And Insertion

- Is first contact analogous to target-radius crossing?
- What contact states are recoverable?
- Can a system distinguish edge contact from useful insertion contact?
- How much force/pose ambiguity can be tolerated before retreat is necessary?
- Can imitation learning learn recovery actions from failed insertions?

## 9.4 Hardware Constraints

- How much latency can the autonomy stack tolerate?
- Does delayed perception fail differently from noisy perception?
- Can edge AI constraints change controller design?
- Is a smaller, faster model safer than a larger, more accurate but slower model?
- Can a safety monitor compensate for compute degradation?

## 9.5 Sim-To-Real And Robotics Transfer

- Which recovery principles transfer from 2D spacecraft control to plug insertion?
- Which fail because contact physics are fundamentally different?
- Can failure labels be shared across domains at the abstraction level?
- Can a recoverability metric be domain-general?
- Can hardware-aware policies outperform purely simulation-trained policies?

## 9.6 Autonomy Architecture

- Can recoverability be predicted from multimodal sensor history?
- Can a planner decide when not to act because estimates are unreliable?
- What is the right interface between perception confidence and control authority?
- How should an autonomy stack choose between continue, correct, retreat, retry, and abort?
- Can a system refuse false progress without becoming overly conservative?

---

# 10. Repository Evolution

## 10.1 Future Directory Additions

Recommended additions:

```text
spacecraft_ai/
  perception/
  sensors/
  estimation/
  calibration/
  contact/
  hardware/
  embedded/
  sim2real/
  safety/
  faults/
```

These should sit beside the modules proposed in the prior roadmap:

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
```

## 10.2 Relationship To Existing Modules

| New Module | Depends On | Feeds Into |
|---|---|---|
| `perception/` | datasets, sensors, calibration | estimation, benchmarks |
| `sensors/` | sim, scenarios | estimation, evaluation |
| `estimation/` | sensors, perception, calibration | controllers, planners, safety |
| `calibration/` | sensor metadata | perception, estimation |
| `contact/` | sensors, state/action history | metrics, safety, controllers |
| `hardware/` | action/sensor schemas | HIL, logging |
| `embedded/` | models, profiling | hardware-budget benchmarks |
| `sim2real/` | replays, datasets, metrics | reports, benchmark comparison |
| `safety/` | metrics, estimator confidence | controllers, hardware interface |
| `faults/` | sensors, hardware, sim | robustness benchmarks |

## 10.3 Documentation Additions

Add:

```text
docs/architecture/perception_architecture.md
docs/architecture/sensor_model_interfaces.md
docs/architecture/contact_state_taxonomy.md
docs/architecture/hardware_abstraction.md
docs/architecture/embedded_constraints.md
docs/benchmarks/perception_pose_v1.md
docs/benchmarks/contact_insertion_v1.md
docs/benchmarks/latency_stressed_control_v1.md
```

## 10.4 Artifact Additions

Future experiments should write:

```text
analysis/perception_pose_v1/
analysis/sensor_noise_v1/
analysis/contact_taxonomy_v1/
analysis/latency_control_v1/
analysis/hardware_budget_v1/
analysis/sim2real_gap_v1/
```

Each directory should include:

```text
results.csv
summary.md
manifest.json
plots/
```

## 10.5 Minimal Interface Types

Future code should eventually define:

```python
ImageObservation
DepthObservation
IMUObservation
ForceTorqueObservation
PoseEstimate
EstimatedState
ContactState
HardwareCommand
SafetyDecision
LatencyProfile
EmbeddedProfile
```

Do not overbuild these before they are used. Start with dataclasses and simple fake data.

---

# 11. GitHub Milestones And Issues

The first issues should be lightweight and realistic. The goal is to create vocabulary, interfaces, and synthetic tests before any hardware work.

## Milestone 1: Perception And Sensor Architecture Notes

### Issue: Write Perception Architecture Note

Body:

Define how perception should enter the platform without changing current controller benchmarks. Clarify image observations, pose estimates, confidence, uncertainty, and failure labels.

Checklist:

- [ ] Define `ImageObservation`
- [ ] Define `PoseEstimate`
- [ ] Define `PerceptionResult`
- [ ] Explain perception vs task success
- [ ] Add examples from spacecraft and plug insertion

Expected artifact path:

```text
docs/architecture/perception_architecture.md
```

Acceptance criteria:

- Document separates perception accuracy from task success.
- Document includes no hardware claims.
- Document explains how perception connects to evaluator and metrics.

### Issue: Define Sensor Model Interfaces

Body:

Create a design document for sensor model interfaces, including camera, IMU, depth/range, and force/contact sensors.

Checklist:

- [ ] Define common timestamp field
- [ ] Define noise model concept
- [ ] Define dropout/failure flags
- [ ] Define simulated vs logged sensor source
- [ ] Define relationship to estimator

Expected artifact path:

```text
docs/architecture/sensor_model_interfaces.md
```

Acceptance criteria:

- At least four sensor classes are described.
- Interface does not depend on specific hardware.
- Relationship to future benchmarks is clear.

## Milestone 2: Synthetic Sensor And Latency Prototypes

### Issue: Add Fake Noisy Sensor Observations

Body:

Create lightweight synthetic sensor outputs from simulator state. Start with noisy position/velocity or pose estimates, not real images.

Checklist:

- [ ] Add simple dataclasses for sensor observation
- [ ] Add Gaussian noise model
- [ ] Add dropout flag
- [ ] Add deterministic seed support
- [ ] Add small unit test

Expected artifact path:

```text
spacecraft_ai/sensors/
Tests/test_synthetic_sensor_models.py
```

Acceptance criteria:

- Sensor outputs are deterministic under fixed seed.
- No existing Phase34/36/37 results change.
- Sensor model can be used offline without hardware.

### Issue: Add Latency/Delay Simulation Prototype

Body:

Implement a simple latency wrapper that delays observations by a fixed or sampled number of steps.

Checklist:

- [ ] Add fixed-step delay
- [ ] Add random-step delay
- [ ] Add stale observation flag
- [ ] Add unit test
- [ ] Add logging fields proposal

Expected artifact path:

```text
spacecraft_ai/sensors/latency.py
Tests/test_latency_model.py
docs/logging_schema_hardware_v1.md
```

Acceptance criteria:

- Delay behavior is deterministic under seed.
- Stale observations are explicitly labeled.
- No controller code needs to change to use the wrapper.

## Milestone 3: Contact And Plug-Insertion Abstraction

### Issue: Create Contact-State Taxonomy From Plug Insertion

Body:

Write a contact-state taxonomy inspired by robotic plug insertion and map it to recoverability concepts.

Checklist:

- [ ] Define contact states
- [ ] Define recoverable vs unrecoverable contact
- [ ] Define failure labels
- [ ] Connect contact to crossing analogy
- [ ] Avoid claiming spacecraft docking validation

Expected artifact path:

```text
docs/architecture/contact_state_taxonomy.md
```

Acceptance criteria:

- Taxonomy includes at least 8 contact states.
- Document clearly distinguishes plug insertion from spacecraft insertion.
- Document explains how contact metrics could be evaluated.

### Issue: Define Contact Benchmark Concept

Body:

Create a benchmark design note for contact-rich insertion without implementing real hardware.

Checklist:

- [ ] Define approach/contact/post-contact/final lock phases
- [ ] Define contact metrics
- [ ] Define failure labels
- [ ] Define synthetic contact data requirements
- [ ] Define what would be premature

Expected artifact path:

```text
docs/benchmarks/contact_insertion_v1.md
```

Acceptance criteria:

- Benchmark separates contact from insertion success.
- Benchmark includes recoverability after contact.
- Benchmark is software-only at this stage.

## Milestone 4: Hardware-Aware Logging

### Issue: Add Hardware-Aware Logging Schema Proposal

Body:

Extend logging vocabulary with latency, compute, model, and sensor-health fields.

Checklist:

- [ ] Add inference time field
- [ ] Add loop rate field
- [ ] Add sensor latency field
- [ ] Add dropped frame count
- [ ] Add model size field
- [ ] Add quantization flag
- [ ] Add hardware notes field

Expected artifact path:

```text
docs/logging_schema_hardware_v1.md
```

Acceptance criteria:

- Schema is proposal-only.
- No historical CSVs are reinterpreted.
- Fields are diagnostic, not success metrics.

### Issue: Create Hardware-Budget Benchmark Design Note

Body:

Define how compute/latency/memory constraints should be benchmarked later.

Checklist:

- [ ] Define latency budget
- [ ] Define memory/model-size fields
- [ ] Define control-loop frequency
- [ ] Define failure modes caused by compute limits
- [ ] Define edge-inference comparison rules

Expected artifact path:

```text
docs/benchmarks/hardware_budget_v1.md
```

Acceptance criteria:

- Does not require buying hardware.
- Connects hardware budget to autonomy performance.
- Separates profiling from task success.

## Milestone 5: Sim-To-Real Foundations

### Issue: Define Sim-To-Real Evaluation Contract

Body:

Create a document explaining how simulated, replayed, logged, and live data should be compared in the future.

Checklist:

- [ ] Define simulation artifact
- [ ] Define replay artifact
- [ ] Define logged hardware artifact
- [ ] Define comparison metrics
- [ ] Define failure-label mismatch

Expected artifact path:

```text
docs/architecture/sim2real_evaluation_contract.md
```

Acceptance criteria:

- Contract does not claim current sim-to-real transfer.
- Contract can apply to plug insertion and future spacecraft tasks.
- Contract requires provenance metadata.

---

# 12. What Not To Do Yet

Be strict.

## 12.1 Do Not Buy Hardware Immediately

The repo needs software interfaces, logging, replay, and benchmarks before hardware. Hardware without reproducibility infrastructure will create unrepeatable demonstrations.

## 12.2 Do Not Claim Sim-To-Real Transfer

The current repository is a simulator and research artifact base. It does not validate real hardware transfer.

## 12.3 Do Not Train Large Vision Models Yet

Vision should begin with schemas, synthetic observations, and pose-estimation benchmarks. Large models should come only after data, metrics, and failure labels exist.

## 12.4 Do Not Add ROS Immediately

ROS may eventually be useful, but adding it too early will pull the repository toward hardware integration before the architecture is ready.

## 12.5 Do Not Replace The Current Simulator

The current simulator is still the foundation for the spacecraft-control research trail. Add sensor wrappers and estimation layers around it before replacing or rewriting it.

## 12.6 Do Not Turn The Repo Into A Robotics Repo Too Quickly

Plug insertion should inform abstraction, contact taxonomy, sensing, and failure-aware control. It should not erase the spacecraft-control identity.

## 12.7 Do Not Mix Claims Across Domains

A plug-insertion result does not prove spacecraft insertion. A spacecraft simulator result does not prove robotic insertion. Transfer should be argued through shared architecture and benchmarks, not claim substitution.

## 12.8 Do Not Treat Perception Metrics As Task Metrics

Pose accuracy, detection rate, or segmentation quality should not be reported as task success. They are upstream measurements.

## 12.9 Do Not Hide Sensor Failure

Dropped frames, low confidence, calibration bias, and estimator divergence are not nuisances. They are central research data.

## 12.10 Do Not Over-Abstract Before Use

Start with simple dataclasses and fake sensor observations. Add abstraction only when multiple real use cases need it.

---

# 13. Final Strategy

## Most Important Conceptual Bridge

The most important bridge between spacecraft control and robotics is:

```text
intermediate success is not recoverable physical completion
```

In spacecraft:

```text
crossing is not insertion
```

In perception:

```text
detection is not task success
```

In plug insertion:

```text
contact is not insertion
```

In hardware autonomy:

```text
computed action is not safe execution
```

The long-term platform should study recoverability after uncertain, sensor-mediated, physically constrained boundary events.

## Safest First Hardware/Vision-Related Step

The safest first step is documentation and synthetic interfaces:

```text
write perception/sensor/contact architecture notes
define simple dataclasses
add fake noisy sensor observations
add latency wrappers
add hardware-aware logging fields
```

This adds hardware/vision thinking without claiming hardware capability.

## Highest-Risk/Highest-Reward Direction

The highest-risk/highest-reward direction is a unified recoverability-aware autonomy benchmark spanning:

- spacecraft crossing/recoverability
- plug contact/insertion recoverability
- perception uncertainty
- contact feedback
- latency/hardware constraints

This could become a distinctive research identity, but it is risky because cross-domain claims can easily become vague. It must be grounded in precise benchmarks and failure labels.

## 2-Week Plan

1. Create `docs/architecture/perception_architecture.md`.
2. Create `docs/architecture/sensor_model_interfaces.md`.
3. Create `docs/architecture/contact_state_taxonomy.md`.
4. Draft `docs/logging_schema_hardware_v1.md`.
5. Add no hardware code and make no claims.
6. Keep all current Phase34/36/37 regression guards unchanged.

## 2-Month Plan

1. Add simple synthetic sensor dataclasses under a new module.
2. Add noisy state/pose observation generation.
3. Add latency wrapper for delayed observations.
4. Add unit tests for deterministic sensor noise and delay.
5. Add a small noisy-state benchmark design document.
6. Add contact-rich insertion benchmark concept document.
7. Create first analysis artifact showing how perfect-state control degrades under fake noisy estimates.

Expected artifacts:

```text
spacecraft_ai/sensors/
spacecraft_ai/estimation/
docs/benchmarks/noisy_state_control_v1.md
docs/benchmarks/contact_insertion_v1.md
analysis/noisy_state_control_v1/
```

## 1-Year Plan

1. Stabilize sensor and estimator interfaces.
2. Add perception-only and pose-estimation benchmark definitions.
3. Add synthetic camera/pose datasets.
4. Add contact-state taxonomy and software-only contact benchmark.
5. Add latency-stressed control benchmark.
6. Add hardware-aware logging schema to new experiments.
7. Evaluate whether estimated-state control degrades crossing/recoverability in spacecraft tasks.
8. Evaluate whether contact-state labels from plug insertion can become a reusable recoverability taxonomy.
9. Only after this, consider a small hardware-in-the-loop prototype using safe, low-cost, well-logged data.

Final direction:

```text
Do not leave the project as a perfect-state simulator forever.
But do not jump to hardware before the platform can explain sensor-mediated failure.
```

The right expansion path is:

```text
2D recoverability -> noisy estimates -> perception -> contact -> latency -> hardware-aware evaluation -> hardware-in-the-loop
```

That path preserves the spacecraft-control foundation while letting the project grow toward robotics, sensing, embedded autonomy, and real-world physical execution.
