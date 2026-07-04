# Recoverability Formalism v0

Status: working research formalism.

This document is a conceptual and mathematical vocabulary for recoverability-aware physical autonomy. It is not a claim of a new scientific theory, not a flight-readiness argument, and not a replacement for existing concepts such as reachability, viability, stability, robustness, or runtime assurance. Its purpose is narrower: to give this research platform a precise language for asking when an autonomous physical system can still complete, stabilize, abort, retry, or degrade gracefully after an intermediate event has occurred.

The current spacecraft-control result can be summarized as:

> Crossing is not insertion.

The broader principle is:

> Intermediate success is not recoverable task completion.

That principle applies beyond orbital control:

- Detecting a docking target is not docking.
- Estimating a pose is not successful insertion.
- Reaching a contact point is not stable assembly.
- Seeing a socket is not inserting a plug.
- Computing a feasible plan is not surviving physical execution.
- Entering a desired region is not enough if the system cannot remain safe, stabilize, or continue the mission afterward.

Recoverability is the proposed first-class concept for representing the gap between an event that appears successful and the physical, controlled, resource-bounded ability to turn that event into useful task progress.

## 1. Motivation

Physical autonomy is often evaluated by event success. A spacecraft reaches a target radius. A robot touches a socket. A manipulator grasps an object. A vision model detects a target. A planner finds a trajectory. These events are useful, but they are not equivalent to task completion.

The repository's orbital-control experiments make this distinction concrete. A trajectory may cross a target-radius boundary without entering a state from which the controller can stabilize, synchronize, or continue toward a simulator-defined successful outcome. In the current evidence base, Phase34-style post-cross synchronization improved recoverability for crossing-producing cases, but the upstream bottleneck remained crossing generation. That result should not be overstated as full autonomous orbital insertion, but it gives a clear lesson: evaluation must separate crossing, recoverability, overspeed, instability, closest approach, and final task outcome.

The same structure appears in contact-rich robotics. A robotic plug may reach the mouth of a socket but fail to insert because of angular misalignment, excessive contact force, friction, jamming, latency, sensor noise, or poor corrective behavior. A camera may correctly identify the socket, but perception success does not imply contact success. A force sensor may detect contact, but contact is not insertion. A system may imitate a successful demonstration, but imitation alone does not establish that it can recover from off-nominal contacts.

Recoverability deserves first-class status because many real autonomous failures are not failures to ever reach an intermediate event. They are failures to convert that event into a stable, safe, completed, or gracefully degraded state.

Recoverability is especially important when:

- The task has phases: approach, alignment, contact or crossing, post-event stabilization, capture, lock, or completion.
- Intermediate event metrics are easy to measure but incomplete.
- The system operates under finite fuel, finite time, limited actuation, latency, sensor noise, compute constraints, or contact uncertainty.
- Some failures are irreversible: collision, escape, hardware damage, loss of grasp, unrecoverable spin, excessive force, unstable divergence, or resource depletion.
- A controller can be useful in a subset of states but dangerous or ineffective outside that subset.
- Evaluation must distinguish "reached a boundary" from "entered a state from which success remains physically achievable."

The core recoverability question is:

> Given the current state, belief, controller, resources, constraints, and environment assumptions, can the system still reach an acceptable continuation or completion condition before irreversible failure?

Several parts of this question are essential:

- "Given" makes recoverability relative to assumptions.
- "State or belief" allows partial observability.
- "Controller" makes clear that recoverability is often controller-relative, not absolute.
- "Resources" includes fuel, time, energy, actuator authority, compute, and sensing.
- "Constraints" includes safety envelopes, hardware limits, contact limits, and mission rules.
- "Acceptable continuation" allows goal completion, safe abort, retry, station-keeping, regrasp, or degraded mission success.
- "Before irreversible failure" separates temporary error from unrecoverable loss.

Recoverability is therefore not merely a binary label on a trajectory. It is a relation among a system state, a model, a task, a controller or controller class, a horizon, a cost budget, uncertainty assumptions, and a definition of acceptable continuation.

## 2. Existing Related Ideas

Recoverability overlaps with several mature ideas in control theory, robotics, planning, safety, and runtime assurance. This document does not claim that recoverability replaces those ideas. It organizes them around a practical question that appears repeatedly in this repository: after a system reaches an intermediate event, is useful physical completion still possible?

### Stability

Stability studies whether trajectories remain near, converge to, or stay within a set around an equilibrium, trajectory, or invariant set. Lyapunov stability, asymptotic stability, input-to-state stability, and related notions are central to control theory.

Recoverability is related to stability but not identical.

Similarities:

- Both care about behavior after perturbation or deviation.
- Both can be defined around sets rather than single points.
- Both can involve margins, basins, and robustness to disturbances.

Differences:

- Stability is often about remaining near or converging to a specified behavior.
- Recoverability is about whether a task-relevant acceptable continuation remains achievable.
- A state can be unstable in the local linear sense but still recoverable by a nonlinear or hybrid controller.
- A state can be stable around the wrong behavior while not recoverable for the mission.
- Recoverability includes resources, irreversible failure, perception uncertainty, and task phase semantics more explicitly than many stability definitions.

In the orbital-control setting, a target-radius crossing is not enough. What matters is whether the post-cross state lies in a basin from which the controller can stabilize the radial and tangential behavior without overspeed or instability.

### Reachability

Reachability asks whether a system can reach a set of states from an initial state under admissible controls and dynamics. Backward reachability asks which states can reach a target set, often while satisfying constraints.

Recoverability can be viewed as a task-structured, resource-aware, failure-aware form of backward reachability.

Similarities:

- Both ask whether a target set can be reached.
- Both can be defined over a horizon.
- Both can account for constraints and disturbances.
- Both can be computed approximately through dynamic programming, sampling, search, or learned approximators.

Differences:

- Recoverability emphasizes acceptable continuation, not only geometric target reach.
- Recoverability may target a safe continuation set, a terminal controller set, a retry set, or a degraded mission state rather than only a final goal.
- Recoverability explicitly distinguishes reversible errors from irreversible failures.
- Recoverability is usually evaluated relative to a concrete controller family or runtime stack, not just any mathematically admissible control.
- Recoverability often includes cost, resource, sensing, and hardware budgets.

In this sense, a recoverable state is often a state in a backward reachable set, but the target and constraints are chosen according to task continuation and physical failure semantics.

### Viability

Viability studies whether a system can remain inside an acceptable set indefinitely or over a horizon. A viability kernel is the set of states from which there exists a control policy that keeps the system inside constraints.

Recoverability and viability are close.

Similarities:

- Both are set-based and constraint-aware.
- Both can define safe operating regions.
- Both support runtime safety reasoning.

Differences:

- Viability often focuses on staying within a safe set.
- Recoverability can allow a temporary deviation if the system can return to a useful continuation state before irreversible failure.
- Recoverability may care about reaching a terminal set, retry set, or mission-degraded state, not just remaining safe.
- Recoverability can be phase-specific: a state may be viable during approach but not recoverable for insertion after contact.

Recoverability can use viability as a component. For example, the recovery target may be a viable set from which the system can safely continue.

### Controllability

Controllability asks whether a system can be driven between states, often under idealized assumptions. Classical controllability is a structural property of a dynamical system.

Recoverability is more operational.

Similarities:

- Both involve the ability to influence future state.
- Both depend on actuation and dynamics.

Differences:

- Controllability may ignore state constraints, obstacles, sensor uncertainty, irreversible failure, or resource budgets.
- Recoverability asks whether recovery is possible under the actual task constraints and available controller.
- A system can be controllable in principle but not recoverable in practice because the needed action exceeds thrust, fuel, time, force, torque, compute, or safety limits.
- A system can be locally underactuated but recoverable through task-specific maneuvers.

For this project, controllability is background structure. Recoverability is the evaluation concept that asks whether control authority remains useful for the task at the current phase.

### Robustness

Robustness describes preservation of performance or safety under disturbances, uncertainty, parameter variation, model mismatch, or adversarial perturbation.

Recoverability uses robustness but asks a different question.

Similarities:

- Both care about uncertainty and disturbance.
- Both can be measured by margins.
- Both can be evaluated under stress tests.

Differences:

- Robustness often asks whether a planned behavior still works under perturbation.
- Recoverability asks whether the system can still return to acceptable task progression after a deviation or partial failure.
- A controller can be robust before contact but not recoverable after a bad contact.
- A state can be recoverable under nominal assumptions but not robustly recoverable under bounded noise or latency.

Robust recoverability is a stronger level defined later in this document.

### Resilience

Resilience is the ability of a system to absorb, adapt to, and recover from disruptions. It is common in systems engineering, robotics, cyber-physical systems, and autonomy discussions.

Recoverability can be treated as a state-level or episode-level component of resilience.

Similarities:

- Both involve continued function after disruption.
- Both allow degradation, reconfiguration, retry, or adaptation.
- Both are important for long-duration autonomy.

Differences:

- Resilience is often system-level and broad.
- Recoverability is a more local formal object: a state, belief, trajectory segment, or mission phase is recoverable if acceptable continuation remains possible under specified assumptions.
- Resilience may include organizational, hardware, software, and mission design issues beyond the formal state-space view.

In this repository, resilience should remain the broader engineering ambition. Recoverability should be the sharper evaluation and design concept.

### Safety

Safety usually means avoiding unacceptable harm, unsafe states, collisions, constraint violations, or mission-critical failures. Runtime assurance systems often monitor safety and intervene when necessary.

Recoverability is related but not the same.

Similarities:

- Both require defining unsafe or failure sets.
- Both are useful for runtime monitors and veto logic.
- Both can be probabilistic or worst-case.

Differences:

- Safety can be satisfied by doing nothing useful if the system remains within constraints.
- Recoverability requires a path to acceptable task continuation, completion, abort, or graceful degradation.
- A state can be safe but not recoverable: for example, a spacecraft may be far from collision but out of fuel for insertion.
- A state can be temporarily outside a nominal operating envelope but still recoverable if it has not entered an irreversible failure set.

Recoverability is therefore not a substitute for safety. It adds task-progress meaning to safety.

### Recoverable State

The phrase "recoverable state" appears in many fields. In this repository, the phrase has practical meaning: a crossing may be recoverable if the system can continue from that crossing into a simulator-defined success condition.

This document treats "recoverable state" as an instance of a broader predicate:

> A state is recoverable relative to a task, model, controller class, horizon, and constraints if acceptable continuation remains achievable before irreversible failure.

That definition is intentionally parameterized. Without the parameters, the phrase "recoverable" is ambiguous.

### Terminal Set

In model predictive control and related methods, a terminal set is a set of states at the end of a planning horizon from which a known controller can safely stabilize or complete the task. Terminal sets are often used to guarantee recursive feasibility or stability.

Recoverability can use terminal-set reasoning.

Similarities:

- Both identify states from which a known continuation controller works.
- Both can support planning horizons and safety arguments.
- Both can be represented as sets, approximations, or classifiers.

Differences:

- Terminal sets are often defined for a particular MPC or stabilization proof.
- Recoverability may include retry, abort, degraded mission, perception recovery, contact recovery, or fault management.
- Recoverability may be evaluated at many intermediate events, not only at the end of a planning horizon.

For this platform, a recoverability set can often be implemented as a terminal set for a downstream controller, but the concept is broader than terminal MPC.

## 3. Formal Objects

This section defines the objects used throughout the document. The notation is intentionally lightweight. It is meant to support consistent reasoning and benchmark design, not to present a complete mathematical theory.

### State

The state is the physical and internal configuration relevant to future task evolution.

Let:

```text
x_t in X
```

where `X` is the state space and `t` is time or decision step.

The state may include:

- Spacecraft position, velocity, orbital elements, fuel, attitude, angular rate, actuator health, and mode.
- Robot end-effector pose, object pose, joint state, contact mode, force state, gripper state, and controller mode.
- Sensor calibration parameters, thermal state, compute state, battery, and health flags when they affect future task success.

The true state may not be directly observable.

### Observation

An observation is the measured information available to the controller or estimator.

Let:

```text
y_t in Y
```

Observations may include:

- Simulated state readouts.
- Camera images.
- Pose estimates.
- IMU measurements.
- Force or torque readings.
- Star tracker outputs.
- LiDAR or depth readings.
- Contact flags.
- Health telemetry.

Observations may be noisy, delayed, biased, missing, corrupted, or low-rate.

### Belief State

A belief state represents uncertainty over the true state.

Let:

```text
b_t in B
```

where `b_t` may be:

- A probability distribution over `X`.
- A particle set.
- A bounded uncertainty set.
- A mean and covariance.
- A learned latent representation with uncertainty estimates.
- A history-dependent information state.

Belief is needed because recoverability in real autonomy is often not recoverability from the true state alone, but recoverability given what the system can know and infer.

### Action

An action is a command applied by the controller or planner.

Let:

```text
u_t in U
```

Actions may include:

- Thrust commands.
- Torque commands.
- Desired accelerations.
- Joint velocities.
- End-effector pose increments.
- Gripper commands.
- Mode switches.
- Abort commands.
- Sensor scheduling decisions.
- Compute allocation decisions.

The admissible action set may depend on the state, hardware, time, actuator limits, fuel, contact mode, or safety rules.

### Controller

A controller maps available information to actions.

For a fully observed system:

```text
pi: X -> U
```

For a partially observed system:

```text
pi: B -> U
```

or:

```text
pi: H_t -> U
```

where `H_t` is the observation-action history.

A controller may be:

- Explicitly programmed.
- Learned.
- Hybrid.
- Mode-based.
- Optimization-based.
- Safety-filtered.
- Runtime-assurance monitored.

Recoverability is often controller-relative. A state may be recoverable for a strong controller but not for a weak controller. It may be recoverable for an explicit recovery controller but not for a nominal planner.

### Planner

A planner chooses goals, modes, trajectories, recovery strategies, or controller parameters over a longer horizon.

Let:

```text
P: (task, b_t, context) -> plan
```

A plan may include:

- A trajectory.
- A sequence of controller modes.
- A recovery branch.
- A retry policy.
- An abort decision.
- A degraded mission objective.
- A request for additional perception.

The planner and controller should be distinguished. The planner reasons over task structure and future options; the controller executes local actions. Some systems combine both, but the distinction is useful for recoverability.

### Environment

The environment defines the dynamics, observations, disturbances, constraints, and task context.

A simple stochastic form is:

```text
x_{t+1} = f(x_t, u_t, w_t)
y_t = h(x_t, v_t)
```

where:

- `w_t` is process disturbance or unmodeled dynamics.
- `v_t` is observation noise.
- `f` may be known, approximate, learned, or simulated.
- `h` may include perception and sensor models.

The environment includes physical limits and external conditions:

- Gravity model.
- Contact and friction.
- Target motion.
- Lighting.
- Sensor occlusion.
- Latency.
- Actuator saturation.
- Compute timing.
- Hardware faults.

### Failure Set

The failure set contains states or histories that violate task or safety requirements.

Let:

```text
F subset X
```

Examples:

- Overspeed.
- Collision.
- Excessive contact force.
- Loss of target.
- Resource depletion.
- Leaving the valid simulation region.
- Hardware damage.
- Unstable divergence.
- Failed insertion.

Not all failure states are irreversible. Some may be temporary, recoverable, or acceptable under degraded mission semantics.

### Safe Set

The safe set contains states that satisfy the relevant safety constraints.

Let:

```text
S subset X
```

Safety may include:

- No collision.
- No excessive force.
- No actuator saturation beyond limits.
- No keep-out-zone violation.
- No thermal or power violation.
- No unrecoverable attitude or velocity state.

The safe set is task- and model-dependent. A state may be safe for a simulator benchmark but unsafe for a hardware experiment. A state may be physically safe but mission-useless.

### Goal Set

The goal set contains states that satisfy the desired task completion condition.

Let:

```text
G subset X
```

Examples:

- Stable orbital insertion according to simulator criteria.
- Docked and latched.
- Plug fully inserted.
- Object placed and released.
- End-effector aligned within tolerance and load transferred safely.

The goal set should not be confused with intermediate event sets. A radius crossing, target detection, or first contact may define an event set, but it is usually not the final goal set.

### Irreversible Failure Set

The irreversible failure set contains states or histories from which acceptable continuation is no longer possible under the assumptions of the task.

Let:

```text
I subset F
```

Examples:

- Collision that damages hardware.
- Escape trajectory with insufficient fuel.
- Plug or socket damage.
- Object dropped out of reach.
- Loss of spacecraft attitude with no remaining recovery authority.
- Battery depletion below survival level.
- A contact jam that cannot be cleared without external intervention.

The irreversible failure set is one of the most important objects in recoverability. Recoverability is not lost merely because the system is imperfect; it is lost when no acceptable continuation remains.

### Recovery Horizon

The recovery horizon is the allowed time or number of steps for returning to an acceptable continuation condition.

Let:

```text
H in N
```

or in continuous time:

```text
T_rec > 0
```

The horizon may be determined by:

- Fuel limits.
- Mission timing.
- Thermal limits.
- Orbital geometry.
- Contact dwell time.
- Human safety constraints.
- Real-time control deadlines.

A state may be recoverable with a long horizon but not recoverable under the operational horizon.

### Recovery Cost

Recovery cost measures the resources or penalties required to recover.

Let:

```text
J_rec(x, pi, H)
```

represent a recovery cost under controller `pi` over horizon `H`.

Cost may include:

- Fuel.
- Control effort.
- Time.
- Energy.
- Peak force.
- Risk.
- Deviation from nominal trajectory.
- Compute time.
- Wear.
- Number of retries.

Recoverability should not be binary only. A state may be recoverable but too expensive to be acceptable.

### Recovery Margin

Recovery margin measures how far the system is from losing recoverability.

Possible margins include:

- Distance to an irreversible failure boundary.
- Maximum tolerable disturbance before recovery fails.
- Remaining fuel after successful recovery.
- Remaining time slack.
- Minimum safety clearance during recovery.
- Maximum pose error that can still be corrected.
- Probability gap above a required success threshold.
- Robustness radius in state, belief, or parameter space.

Recovery margin is important because a barely recoverable state and a robustly recoverable state should not be treated as equivalent.

## 4. Definitions

This section defines several recoverability concepts. These are working definitions intended for research use and may need revision as the platform matures.

### Deterministic Recoverability

Let:

- `X` be the state space.
- `S subset X` be the safe set.
- `I subset X` be the irreversible failure set.
- `R subset X` be an acceptable recovery target set.
- `H` be a recovery horizon.
- `U_adm(x)` be the admissible action set at state `x`.
- `f` be deterministic dynamics.

The recovery target `R` may be:

- The goal set `G`.
- A terminal set for a known controller.
- A viable set.
- A retry state set.
- A safe abort set.
- A degraded mission continuation set.

A state `x_0` is deterministically recoverable to `R` within horizon `H` if there exists an admissible action sequence:

```text
u_0, u_1, ..., u_{H-1}
```

such that the resulting trajectory:

```text
x_{t+1} = f(x_t, u_t)
```

satisfies:

```text
x_t notin I              for all t = 0, ..., H
x_t satisfies constraints for all required t
x_tau in R               for some tau <= H
```

Optionally, require:

```text
J_rec(x_0, u_{0:H-1}) <= C_rec
```

for a recovery cost budget `C_rec`.

This definition is intentionally existence-based. It says recovery is possible under the model. It does not say that the deployed controller will find or execute the recovery.

### Probabilistic Recoverability

For stochastic dynamics and observations:

```text
x_{t+1} = f(x_t, u_t, w_t)
y_t = h(x_t, v_t)
```

a state or belief is probabilistically recoverable if a policy can reach an acceptable recovery target with probability at least a specified threshold while avoiding irreversible failure with sufficient probability.

One sketch is:

```text
P_pi(reach R before I within H | x_0) >= alpha
```

with optional constraints such as:

```text
E[J_rec] <= C_rec
P(J_rec <= C_rec) >= beta
P(violate S before recovery) <= delta
```

where:

- `alpha` is the minimum recovery probability.
- `beta` is a cost-confidence threshold.
- `delta` is an acceptable safety-violation probability.

Probabilistic recoverability must state the uncertainty model. A probability without a model is not meaningful.

### Controller-Relative Recoverability

A state may be recoverable in principle but not recoverable by the controller actually available to the system.

Let `Pi` be a class of controllers or `pi` be a specific controller.

A state `x` is recoverable relative to `Pi` if:

```text
exists pi in Pi such that pi recovers x
```

A state is recoverable relative to a fixed controller `pi` if that controller recovers it under the defined model, horizon, target, and constraints.

This distinction is central to the repository's current work. A crossing state should not be called recoverable unless the downstream controller can actually convert it into the success condition under the benchmark definition. Phase34-style results should therefore be interpreted as controller-relative recoverability for existing crossing-producing cases, not as a universal statement about all states or all controllers.

### Belief Recoverability

Under partial observability, the system does not know the true state `x_t`. It has a belief `b_t`.

A belief `b` is recoverable if there exists a policy based on observations or beliefs that recovers with sufficient probability under the belief distribution.

A probabilistic form is:

```text
P_pi(reach R before I within H | b_0 = b) >= alpha
```

A conservative set-based form is:

```text
for all x in support(b), x is recoverable
```

The conservative form is stronger and often too strict. The probabilistic form is more practical but depends on the correctness of the belief model.

Belief recoverability is essential for perception-driven autonomy. A vision system may report a high-confidence pose, but if the belief underestimates uncertainty or ignores calibration error, the controller may incorrectly treat the situation as recoverable.

### Mission Recoverability

Mission recoverability asks whether the system can still achieve an acceptable mission-level outcome, not only the immediate local goal.

A mission may have:

- Primary goals.
- Secondary goals.
- Abort goals.
- Degraded goals.
- Retry allowances.
- Resource budgets.
- Time windows.
- Safety priorities.

A state is mission-recoverable if there exists an acceptable continuation of the mission specification from that state.

This matters because local failure and mission failure are not always the same. A robot may fail one insertion attempt but remain able to retry. A spacecraft may miss an ideal transfer but still enter a safe loiter or alternate orbit. A manipulator may lose a grasp but still regrasp if the object remains reachable and undamaged.

Mission recoverability therefore depends on task semantics, not only physics.

### Physical Recoverability

Physical recoverability means recovery is possible under the actual physical constraints of the system, not only under an abstract model.

It includes:

- Actuator limits.
- Fuel, energy, or battery limits.
- Contact forces.
- Friction.
- Latency.
- Sensor noise.
- Calibration error.
- Compute limits.
- Thermal limits.
- Structural limits.
- Hardware faults.
- Real-time deadlines.

A mathematically reachable recovery path is not physically recoverable if it requires impossible thrust, unmodeled contact behavior, unrealistically precise sensing, unlimited compute, or unsafe force.

This definition is important for the long-term platform because future work may involve vision, sensors, embedded inference, and hardware-in-the-loop prototypes. The project should not treat software success as physical success without checking the physical assumptions.

## 5. Recoverability Levels

Recoverability should be reported as more than a binary whenever possible. The following hierarchy is a working classification.

| Level | Name | Meaning |
| --- | --- | --- |
| 0 | Irrecoverable | No acceptable continuation exists under the stated assumptions, or the state is already in an irreversible failure set. |
| 1 | Marginal | Recovery appears possible only under narrow conditions, high cost, long horizon, low disturbance, accurate sensing, or fragile controller behavior. |
| 2 | Recoverable | A credible controller or policy can reach an acceptable continuation state within the horizon and cost budget while avoiding irreversible failure. |
| 3 | Robustly Recoverable | Recovery remains possible under specified bounded disturbances, model errors, latency, observation noise, or parameter variation. |
| 4 | Guaranteed Recoverable | Recovery is proven for a defined model, controller or controller class, uncertainty set, constraints, and horizon. The guarantee is only as strong as those assumptions. |

### Level 0: Irrecoverable

A state is irrecoverable if the system has already entered `I`, or if no admissible recovery exists before `I` under the specified assumptions.

Examples:

- A spacecraft has insufficient fuel to avoid escape or collision.
- A plug is jammed and cannot be removed without damaging the socket.
- An object is dropped outside the reachable workspace.
- A vehicle violates a hard safety boundary with no remaining control authority.

Irrecoverability should be defined carefully. A state may be irrecoverable for the current controller but recoverable for another controller. Reports should specify whether the claim is absolute under a model, controller-relative, or benchmark-relative.

### Level 1: Marginal

A marginal state is recoverable only with little slack.

Characteristics:

- Very small fuel, time, or force margin.
- Sensitivity to sensor noise.
- Sensitivity to latency.
- Narrow controller parameter range.
- High peak control effort.
- High probability of entering failure under small disturbance.
- Recovery depends on a precise sequence of actions.

Marginal recoverability is useful to identify because it often appears as a success in small deterministic benchmarks but fails under randomized initial conditions, threshold sensitivity tests, or hardware constraints.

### Level 2: Recoverable

A recoverable state has a credible recovery path or controller under the defined benchmark assumptions.

This is the practical minimum for calling an intermediate event useful. For example, an orbital target-radius crossing should count as recoverable only if the post-cross state can be converted into the benchmark's success condition by the available controller within the evaluation horizon and without overspeed or instability.

Recoverable does not mean robust. It means the system has crossed the threshold from "event happened" to "event can be turned into task progress."

### Level 3: Robustly Recoverable

A robustly recoverable state remains recoverable under a specified uncertainty set.

The uncertainty set must be stated:

- Bounded state perturbation.
- Bounded observation noise.
- Bounded process disturbance.
- Parameter variation.
- Contact uncertainty.
- Latency.
- Actuator saturation.
- Sensor dropout.

Robust recoverability is the level most relevant to physical autonomy experiments. If a result only holds under exact simulator state, no latency, perfect sensing, and deterministic dynamics, it should not be described as robustly recoverable.

### Level 4: Guaranteed Recoverable

A guaranteed recoverable state has a proof or certified computation showing recovery under specified assumptions.

The word "guaranteed" must always be qualified:

- Guaranteed for which dynamics?
- Guaranteed for which uncertainty set?
- Guaranteed for which controller?
- Guaranteed for which horizon?
- Guaranteed for which constraints?
- Guaranteed under which sensing assumptions?

In this project, guarantee language should be used sparingly. Most near-term results will be empirical, controller-relative, and benchmark-relative.

## 6. Mathematical Sketches

This section gives notation that can guide implementation and future papers. It is not a complete formal theory.

### Dynamics and Observation

A general discrete-time model:

```text
x_{t+1} = f(x_t, u_t, w_t)
y_t = h(x_t, v_t)
```

where:

- `x_t` is the true state.
- `u_t` is the action.
- `w_t` is process disturbance.
- `y_t` is the observation.
- `v_t` is observation noise.

A belief update can be written as:

```text
b_{t+1} = update(b_t, u_t, y_{t+1})
```

This may represent a Kalman filter, particle filter, learned estimator, bounded-set estimator, or other state-estimation method.

### Recovery Target

Let the recovery target be:

```text
R subset X
```

`R` should be selected according to task semantics. Possible choices:

```text
R = G                 goal set
R = T_pi              terminal set for controller pi
R = V                 viable continuation set
R = A                 abort-safe set
R = D                 degraded mission set
R = Retry             retry-ready set
```

The choice of `R` changes the meaning of recoverability. A state recoverable to abort may not be recoverable to mission completion.

### Deterministic Predicate

For a fixed controller `pi`, define:

```text
Rec_H^pi(x; R, S, I) = true
```

if the closed-loop trajectory from `x` under `pi` reaches `R` within `H`, avoids `I`, and satisfies required constraints.

For a controller class `Pi`:

```text
Rec_H^Pi(x; R, S, I) = exists pi in Pi such that Rec_H^pi(x; R, S, I)
```

This distinguishes fixed-controller evaluation from controller-search evaluation.

### Recovery Cost

Define:

```text
J_rec = sum_{t=0}^{tau-1} c_rec(x_t, u_t) + c_terminal(x_tau)
```

where `tau <= H` is the first time the trajectory reaches `R`.

Cost terms might include:

```text
c_rec = fuel_cost + effort_cost + time_cost + risk_cost + force_cost + compute_cost
```

For current orbital experiments, useful cost terms may include:

- Total delta-v proxy.
- Integrated thrust magnitude.
- Peak speed.
- Time to recover.
- Minimum distance to failure threshold.
- Number of unstable steps.

For plug insertion, useful terms may include:

- Peak contact force.
- Integrated force.
- Number of retries.
- Time in contact before insertion.
- Pose correction distance.
- Probability of jamming.

### Recovery Margin

A simple geometric margin can be written as:

```text
m_I(x, pi) = min_{t in 0:tau} dist(x_t, I)
```

This measures closest approach to irreversible failure during recovery.

A robustness-radius sketch:

```text
m_rec(x) = sup epsilon such that all x' with d(x', x) <= epsilon are recoverable
```

A probabilistic margin:

```text
m_prob(x) = P_pi(reach R before I within H | x) - alpha
```

A resource margin:

```text
m_fuel(x) = fuel_remaining_after_recovery
```

No single margin is universal. The platform should report margins that match the physical task.

### Probabilistic Predicate

For a stochastic system:

```text
PRec_H^pi(x; R, I, alpha) =
    [P_pi(reach R before I within H | x) >= alpha]
```

For belief:

```text
PRec_H^pi(b; R, I, alpha) =
    [P_pi(reach R before I within H | b) >= alpha]
```

This can be estimated empirically by rollouts, bounded by analysis, or approximated by learned predictors. Reports should distinguish those cases.

### Mission-State Predicate

Let `q_t` be a mission or task automaton state, such as:

```text
approach -> align -> contact/crossing -> stabilize -> lock/complete
```

The combined state is:

```text
z_t = (x_t, q_t)
```

Mission recoverability can be defined over `z_t`:

```text
MRec_H(z_t) = true
```

if there exists an acceptable mission continuation path from `(x_t, q_t)`.

This is useful because the same physical state may have different meaning in different task phases. A contact force that is acceptable during insertion may be unsafe during free-space motion. A radius crossing may matter only after a planned transfer phase.

## 7. Cross-Domain Examples

### Orbital Insertion

In the current 2D orbital-control setting, the system state may include radius, radial velocity, tangential velocity, time, fuel proxy, controller mode, and diagnostic flags.

An intermediate event is:

```text
radius crosses target radius
```

The goal is not merely crossing. A useful goal or recovery target may require:

- Remaining within speed limits.
- Avoiding unstable divergence.
- Avoiding overspeed.
- Reducing radial error.
- Synchronizing tangential behavior.
- Entering a state where the post-cross controller can complete the simulator-defined success condition.

The recoverability distinction is:

- Crossing-producing state: the trajectory reaches the target radius.
- Recoverable crossing: the state at or after crossing can still be converted into success by the available controller.
- Irrecoverable crossing: the trajectory crosses but with velocity, geometry, or energy conditions that lead to overspeed, escape, collision, or failure.

The current evidence should be interpreted conservatively:

- Phase34-style post-cross synchronization improved recoverability for known crossing-producing cases.
- It did not solve upstream crossing generation for all benchmark cases.
- Phase36B transfer-family variants did not expand the crossing count beyond the known successes.
- Phase37 radial timing and weak-tangential subset experiments did not establish new full-benchmark successes.

In this domain, recoverability is a controller-relative property of the post-cross state and a benchmark-relative property of the rollout. It should be reported separately from crossing count, final success, overspeed, instability, and closest approach.

### Docking

For spacecraft docking, the state may include relative position, relative velocity, attitude, angular rate, target pose, sensor state, fuel, actuator health, and docking mechanism status.

Intermediate events include:

- Target detected.
- Relative pose estimated.
- Docking corridor entered.
- Soft contact made.
- Capture mechanism touched.

None of these is docking.

Recoverability depends on whether the system can still:

- Reduce relative velocity.
- Maintain alignment.
- Avoid keep-out-zone violations.
- Abort safely.
- Reacquire target if perception degrades.
- Avoid damaging contact.
- Enter a capture-ready state.
- Complete latch or retreat.

An apparently good pose estimate may not imply recoverability if uncertainty is underestimated. A successful approach may not imply recoverability if fuel margin is too low for abort. A contact event may be irrecoverable if it induces tumble, damage, or latch misalignment that cannot be corrected.

Docking therefore requires both physical recoverability and belief recoverability.

### Robotic Plug Insertion

For plug insertion, the state may include plug pose, socket pose, end-effector pose, joint state, contact mode, force and torque measurements, frictional state, vision estimates, calibration, and controller mode.

Intermediate events include:

- Socket detected.
- Pose estimated.
- Plug reaches socket mouth.
- First contact occurs.
- Tip enters chamfer.
- Partial insertion occurs.

None of these is complete insertion.

Recoverability depends on whether the system can still:

- Correct angular misalignment.
- Detect contact mode.
- Reduce lateral error.
- Avoid excessive force.
- Avoid jamming.
- Withdraw and retry.
- Switch from vision to contact feedback.
- Complete seating without damaging the plug or socket.

Contact-rich manipulation makes recoverability especially visible because contact can be both useful and dangerous. First contact may be a necessary event, but the post-contact state determines whether the system is in an insertion basin, a retryable misalignment, or an irreversible jam.

The lesson from spacecraft control transfers directly:

```text
crossing is not insertion
contact is not insertion
seeing the hole is not insertion
pose accuracy is not recoverability
```

The robotics lesson transfers back to spacecraft:

```text
event success must be followed by post-event stabilization and failure-aware correction
```

### Manipulation

For general manipulation, the state may include robot configuration, object pose, grasp state, contact state, environment geometry, perception belief, and task phase.

Intermediate events include:

- Object detected.
- Grasp pose planned.
- Fingers contact object.
- Object lifted.
- Object placed near target.
- Assembly feature aligned.

Recoverability asks whether the task can still continue if:

- The object slips.
- The pose estimate is wrong.
- Contact occurs early.
- The object is partially occluded.
- A grasp is weak but not lost.
- A placement is near but not stable.
- The robot must regrasp or retry.

A manipulation state can be safe but not recoverable if the object is out of reach, the grasp is unstable, or the remaining workspace prevents correction. Conversely, a state can look messy but be recoverable if the object is still reachable, undamaged, and the robot has a retry strategy.

Recoverability is therefore a useful bridge between task planning, contact control, perception, and failure handling.

## 8. Open Questions

This formalism raises questions that should remain explicit rather than hidden inside scripts or benchmark conventions.

### Choosing the Recovery Target

What should `R` be for each task?

Possible choices include final success, a terminal controller set, a safe abort set, a retry-ready state, or a degraded mission state. Different choices produce different recoverability claims. The benchmark must state which target is used.

### Estimating Recoverability Under Partial Observability

How should recoverability be estimated when the true state is unknown?

Belief recoverability depends on the estimator. If the estimator is overconfident, the system may believe it is recoverable when it is not. Future work should measure the effect of calibration, latency, dropout, and biased perception on recoverability.

### Measuring Recovery Margin

What margin best predicts real failure?

Distance to failure set, fuel margin, time margin, force margin, probability margin, and robustness radius may all matter. The right margin is task-specific. Benchmarks should avoid collapsing all margins into one scalar too early.

### Controller-Relative Versus Absolute Recoverability

Should a state be called recoverable if some unknown controller could recover it, or only if the current controller can?

For empirical repository work, controller-relative recoverability is usually more honest. The stronger existence claim requires broader controller search or formal analysis.

### Recovery Under Model Mismatch

How much model error can recovery tolerate?

This question becomes central when moving from clean simulation to randomized initial conditions, noisy sensors, contact models, embedded constraints, or hardware-in-the-loop experiments.

### Recoverability and Runtime Assurance

Can a runtime monitor estimate that the system is about to leave the recoverable set and veto the nominal controller?

This connects recoverability to trust decay, failure-mode labeling, goal degradation, and final veto logic. The monitor should not only ask "is the state safe now?" but also "will the system still be able to recover if it continues this action?"

### Learning Recoverability Predictors

Can recoverability be predicted from trajectory history, sensor history, or belief state?

This is attractive for robotics and spacecraft autonomy, but it must be evaluated carefully. A learned recoverability classifier is only useful if it is calibrated, validated on held-out conditions, and tied to clear failure definitions.

### Multi-Controller Recoverability

When should a system switch controllers?

A nominal controller may be good for approach, a recovery controller for post-event stabilization, an abort controller for safety, and a contact controller for insertion. Recoverability may depend on whether the system can select the correct controller at the correct time.

### Irreversibility

How should irreversible failure be defined?

In simulation, irreversibility may be a threshold. In hardware, it may mean damage, loss of object, unsafe force, or mission loss. Some failures are irreversible only under current resources, not in an absolute sense. This must be made explicit.

### Evaluation Granularity

Should recoverability be measured per state, per crossing, per rollout, per task phase, or per benchmark family?

The repository's current work shows why this matters. Subset results should not be treated as full benchmark results, and crossing counts should not be conflated with recoverable crossing counts.

## 9. Future Research

This formalism suggests several research directions for the platform. These should be built incrementally, preserving the existing spacecraft-control identity while allowing future robotics, perception, and hardware-aware work.

### Recoverability Metrics Engine

The evaluator should report separate metrics for:

- Intermediate event occurrence.
- Final success.
- Recoverable event count.
- Irrecoverable event count.
- Closest approach.
- Overspeed.
- Instability.
- Irreversible failure.
- Recovery time.
- Recovery cost.
- Recovery margin.

The goal is to prevent a single metric from hiding the difference between reaching an event and being able to complete the task.

### Recoverability Set Estimation

Future work can estimate recoverability sets for controllers:

```text
R_pi = {x in X : Rec_H^pi(x; R, S, I)}
```

Approximation methods may include:

- Grid evaluation.
- Monte Carlo rollouts.
- Sensitivity analysis.
- Backward reachable approximations.
- Learned classifiers.
- Conservative set approximations.

For this repository, the first step should remain empirical and benchmark-grounded.

### Recovery-Aware Controller Search

Controller search should optimize not only final success but also:

- Number of recoverable intermediate events.
- Preservation of known recoverable cases.
- Recovery margin.
- Reduced overspeed.
- Reduced instability.
- Fuel or effort cost.
- Robustness under randomized initial conditions.

This aligns with the current scientific conclusion: protect the known successes while searching for new crossing-producing and recoverable cases.

### Belief and Perception-Aware Recoverability

As perception enters the platform, recoverability should be evaluated over belief rather than perfect simulator state.

Research questions include:

- When does better pose accuracy improve recoverability?
- When is perception confidence misleading?
- How much state-estimation error can a controller tolerate?
- Can a controller detect that vision is unreliable and switch to contact, inertial, or fallback sensing?

This generalizes "crossing is not insertion" into "perception success is not task success."

### Runtime Assurance and Veto Logic

Recoverability can support a runtime assurance monitor:

```text
if predicted action leaves recoverable set:
    veto, switch controller, abort, retry, or degrade goal
```

This connects to the project's existing emphasis on audit, trust decay, failure labeling, and final veto power. A useful autonomy stack should know when continued optimization is dangerous because it is consuming the remaining recovery margin.

### Contact-Rich Recovery

Plug insertion and manipulation motivate contact-specific recoverability:

- Is the current contact state retryable?
- Is the force direction informative or misleading?
- Can small corrective actions reduce error without jamming?
- When should the robot withdraw and retry?
- Can recoverability be inferred from force and vision history?

This line of work can inform spacecraft docking and capture, where contact or near-contact events also require post-event stabilization.

### Hardware-Aware Recoverability

Future embedded or hardware-aware work should ask:

- Does latency destroy recoverability?
- Does quantization change controller switching behavior?
- Does compute budget force simpler recovery policies?
- Does sensor dropout cause false recoverability estimates?
- Does power or thermal budget constrain recovery time?

This does not require immediate hardware. The first step can be simulation with latency, noise, rate limits, and resource logging.

### Publication Directions

Potential future papers could build on this v0 formalism without overclaiming:

- A benchmark paper separating intermediate events from recoverable task completion.
- A controller-search paper optimizing recoverability margin under regression constraints.
- A belief-recoverability paper studying pose uncertainty and task success.
- A contact-recoverability paper for robotic insertion and retry policies.
- A runtime-assurance paper using recoverability margins for veto and goal degradation.
- A hardware-aware autonomy paper evaluating latency, compute, and sensor constraints on recoverability.

Each paper should state:

- The system model.
- The controller class.
- The recovery target.
- The horizon.
- The cost budget.
- The uncertainty model.
- The irreversible failure definition.
- The benchmark scope.

## Closing Definition

For this research platform, the recommended working definition is:

> Recoverability is the property that, from a given state or belief, under specified dynamics, constraints, resources, controller assumptions, uncertainty model, and horizon, the system can still reach an acceptable task continuation, completion, retry, abort, or degraded mission state before entering irreversible failure.

This definition is intentionally relative. A recoverability claim without its assumptions is incomplete.

The practical evaluation principle is:

> Report the event, report the outcome, and report whether the event entered a recoverable state.

For the current spacecraft benchmark, this means separating target-radius crossing from recoverable crossing and final success. For future robotics and perception work, it means separating target detection, pose estimation, first contact, partial insertion, recovery, and final task completion.

The long-term platform should treat recoverability as a bridge between simulation, controllers, planners, perception, contact, hardware constraints, and runtime assurance. It should not be used as a slogan. It should be implemented as a measurable, assumption-bound, regression-tested concept.
