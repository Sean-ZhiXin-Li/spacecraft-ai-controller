# Research Direction — Orbital Insertion Architecture

## 1. Core Problem

This project studies recoverable orbital insertion in a simplified 2D control sandbox. The simulator is useful for control-architecture research because it exposes radius, radial velocity, tangential velocity, thrust-limited control, target-radius crossing, CAPTURE, LOCK, and simulator-defined survival criteria under repeatable conditions.

The core problem is that crossing the target radius is not enough. A trajectory can geometrically touch or cross the target orbit while still carrying radial and tangential velocity errors that make the state dynamically unrecoverable.

Recoverable insertion in this sandbox requires:

    crossing -> post-cross synchronization -> recoverability basin -> survival

The project therefore studies orbital insertion as an architecture problem, not as a single reward score or one-step radius objective.

## 2. Research Evolution

- Phase20: tested predictive local planning as a way to improve crossing and insertion behavior through short-horizon action selection.
- Phase31: explored global transfer families, including named transfer structures and burn/coast variants, but did not produce recoverable crossings.
- Phase32: used direct optimal-control style search as an upper-bound probe and showed that recoverable states were physically reachable in the simplified simulator.
- Phase33: extracted the important structure from the best optimal-control behavior: the recoverable state occurred after first crossing through smooth post-cross synchronization.
- Phase34: implemented post-cross synchronization as an explicit terminal controller and converted the existing crossing-producing cases into recoverable crossings.
- Phase35: tested local pre-cross steering biases and found that they did not expand the crossing basin.
- Phase36: prepares a transfer-family search to investigate which long-horizon trajectory structures can create Phase34-compatible crossings.

The research direction has moved from asking whether a controller can reach the target radius to asking which architecture can produce and then recover from a dynamically useful crossing.

## 3. Current Architecture

The current architecture is layered:

Layer 1: transfer family / pre-cross generation

This layer must route an initial condition toward a trajectory that will actually cross the target radius. Phase35 indicates that local steering biases are not enough for this layer.

Layer 2: first target-radius crossing

This is a geometric event, not a final success condition. The crossing state must be evaluated by its radial velocity, tangential velocity error, and compatibility with downstream recovery.

Layer 3: Phase34 post-cross synchronization

This layer is the current terminal controller. Once a crossing exists, Phase34 smooth post-cross synchronization can reduce radius, radial-velocity, and tangential-velocity mismatch until the state enters the recoverability basin.

Layer 4: CAPTURE / LOCK / survival

These simulator regimes represent the post-recoverability stabilization sequence. They are useful internal labels, but they should not be interpreted as real flight validation.

## 4. Current Bottleneck

Phase34 converts crossing-producing cases into recoverable cases. On the reduced benchmark, Phase34 kept the same 8 / 24 crossing count but converted those 8 crossings into 8 recoverable crossings.

Phase35 showed that local upstream biases do not create new crossing-producing cases. The `predictive_crossing_bias` variant matched the Phase34 baseline at 8 / 24 crossings, while `radial_energy_push` and `tangential_corridor_entry` collapsed crossing performance to 0 / 24.

Therefore the current bottleneck is crossing-generation.

The open problem is no longer what to do after a crossing exists. The open problem is how to generate more target-radius crossings that are compatible with the Phase34 terminal controller.

## 5. Current Hypothesis

Crossing-generation is likely a global trajectory-geometry problem rather than a local steering problem.

It may require:

- long-horizon geometry shaping
- timing coordination
- energy/angular-momentum evolution
- transfer-family selection
- planner-level search

This hypothesis is based on the negative Phase35 result. Local radial push did not add crossings. Local tangential corridor correction did not add crossings. A simple predictive local action selector improved crossing-potential scoring but still did not increase the crossing count. That pattern suggests the transfer arc itself must be shaped coherently.

## 6. Near-Term Research Plan

Phase36 should focus on transfer-family search.

The near-term plan is:

- define candidate transfer families with interpretable geometry
- compare families at the trajectory level rather than only by local controller gains
- evaluate which families create target-radius crossings
- measure whether those crossings are compatible with Phase34 post-cross synchronization
- identify families that approach the target radius but fail to commit to crossing
- separate geometric crossing gains from recoverable handoff gains

Phase36 should not jump directly to 3D, C++, SPICE, or high-fidelity astrodynamics. The current research need is to understand the transfer structures that generate usable crossings in the existing 2D sandbox.

## 7. Medium-Term Engineering Plan

After the transfer-family question is clearer, the project can move toward more engineering-realistic tools and environments.

Medium-term directions include:

- C++ simulation core for faster rollouts and larger trajectory-family sweeps
- 3D orbital mechanics
- multi-orbit regimes, including LEO, MEO, GEO, HEO, cislunar, and interplanetary-style transfers
- realistic perturbations such as J2, atmospheric drag, solar radiation pressure, thrust degradation, mass depletion, communication delay, and sensor uncertainty
- fault injection, including actuator asymmetry, fuel faults, navigation corruption, and partial subsystem loss
- MPC-lite for exploiting promising transfer-family geometry
- direct trajectory optimization after family-level structure is understood

These steps should come after the current crossing-generation bottleneck is better characterized. Higher-fidelity simulation will not by itself answer which transfer structures create recoverable insertion attempts.

## 8. Long-Term Vision

The long-term direction is resilient spacecraft autonomy, developed from simplified control-architecture principles toward more realistic space environments.

The durable ideas are:

- survival over optimization
- explicit distinction between milestone events and recoverable regimes
- final veto power when an optimization path becomes unsafe
- failure-mode labeling
- layered autonomy across planning, control, and recovery
- distributed autonomy with structured divergence rather than identical centralized behavior
- operation in complex space environments with partial models, degraded information, and long-horizon uncertainty

This remains a staged research path. The current repository is not a spacecraft autonomy stack. It is a simplified platform for finding control-architecture principles that may later be stress-tested in more realistic simulators.

## 9. What This Project Is Not

This project is not:

- real spacecraft readiness
- full orbital autonomy
- validated guidance, navigation, and control software
- proof that PPO solves spacecraft control
- a claim that current 2D results directly transfer to real missions

The current results are simulation evidence about controller architecture in a simplified 2D orbital sandbox. They are useful because they clarify bottlenecks, not because they certify operational capability.

## 10. Bottom Line

The current research direction is to understand which trajectory structures can produce recoverable orbital insertion, then scale that architecture toward more realistic spacecraft autonomy.
