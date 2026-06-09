# Research Direction - Orbital Insertion Architecture

## 1. Core Research Problem

This repository studies recoverable orbital insertion in a simplified 2D orbital-control sandbox. It is not real spacecraft guidance, navigation, and control software, and it is not flight validation.

The core distinction is:

    first crossing is not orbital insertion

A target-radius crossing is only a geometric event. A trajectory can cross the target radius while carrying radial velocity or tangential velocity errors that make the state dynamically unrecoverable.

The current working architecture is:

    crossing -> post-cross synchronization -> recoverability basin -> survival

The project has therefore moved from asking whether a controller can touch or cross a target radius to asking which trajectory structures produce crossing states that can be recovered by a terminal controller.

## 2. Scientific Arc

The current scientific arc is:

- Early PPO, imitation-learning, and heuristic phases tested whether reactive policies or local explicit controllers could solve insertion directly.
- Phase31 showed that transfer families could produce geometric crossings without recoverable crossings.
- Phase32 showed, with a scoped SciPy direct-shooting probe, that recoverable states were physically reachable in the simplified dynamics. It was not full CasADi/IPOPT direct collocation and was not a production controller.
- Phase33 extracted the important structure from the best direct-shooting behavior: the useful recoverable state occurred after first crossing.
- Phase34 introduced post-cross synchronization and converted the existing crossing-producing benchmark cases into recoverable crossings.
- Phase35 tested local upstream biases and found that they did not expand the crossing basin.
- Phase36A visualized transfer-family geometry on representative cases. It clarified differences between families, but it was not a full benchmark and did not prove a new family works generally.
- Phase36B tested four transfer families on the full 24-case benchmark. All four matched the Phase34 baseline crossing set and did not expand the crossing basin.
- Phase36C isolated the remaining baseline non-crossing cases and prepared a parameterized planner-level transfer search space.
- Phase37A tested radial commitment timing and bounded radial magnitude as the smallest evidence-backed search slice. It did not create new crossings on the baseline non-crossing cases.
- Phase37B tested weak tangential shaping as a narrow subset diagnostic. It created no selected-case crossings and failed to preserve the Phase36B regression crossing set.

## 3. Phase31 Through Phase37B Evidence

Phase31 established the crossing/recoverability gap. In the Phase34 reduced comparison, the Phase31-style reference produced `8 / 24` geometric crossings and `0 / 24` recoverable crossings.

Phase32 provided an upper-bound probe. Because CasADi/IPOPT was unavailable in the checked runtime, the phase used SciPy direct shooting. Its value is that it showed recoverable states can exist under the simplified 2D dynamics, not that the repo has a deployable optimal-control solver.

Phase33 showed why first crossing was insufficient. The best recoverable state occurred after the first crossing, which motivated treating post-cross behavior as an explicit control problem.

Phase34 added a post-cross synchronization mode while preserving the early transfer behavior and simulator thresholds. In the reduced benchmark, Phase34 `radius_priority` kept the same `8 / 24` crossing count but converted those cases into `8 / 24` recoverable crossings.

Phase35 asked whether local pre-cross biases could create more crossing-producing cases. They did not. The Phase34 baseline and `predictive_crossing_bias` both stayed at `8 / 24` crossings, while `radial_energy_push` and `tangential_corridor_entry` collapsed crossing performance.

Phase36A shifted from local steering to transfer-family visualization. Its representative subset should be read as geometry evidence only. It did not improve crossing count and should not be presented as full benchmark proof.

Phase36B then tested `baseline_phase34`, `spiral_approach`, `grazing_corridor`, and `redesigned_delayed_crossing` on the full reduced benchmark. Every family produced `8 / 24` geometric crossings, `8 / 24` Phase34-compatible crossings, `8 / 24` recoverable crossings, `0` overspeed cases, and `0` instability cases. No family expanded the crossing basin beyond `baseline_phase34`.

Phase36C analyzed the `16 / 24` baseline non-crossing cases without running a new controller. The baseline failures split into `8` `near_crossing` cases and `8` `over_conservative_transfer` cases. Across the Phase36B families, closest-approach and crossing-potential metrics changed without producing new target-radius crossings.

Phase37A then tested `early_commit`, `mid_commit`, and `delayed_commit` with `low` and `medium` radial magnitudes over `144` rollouts. It created `0` new crossings on the `16` Phase36B baseline non-crossing cases. `delayed_commit_low` and `delayed_commit_medium` preserved `8 / 24` crossings and `8 / 24` recoverable crossings; early and mid commitment degraded the existing crossing set. No overspeed or instability occurred.

Phase37B then tested a weak tangential correction on four Phase37A-improved `over_conservative_transfer` cases plus eight Phase36B regression crossing cases. It created `0 / 4` selected-case crossings and `0 / 4` selected-case recoverable crossings. It produced no overspeed or instability, but preserved only `4 / 8` regression crossings and `4 / 8` regression recoverable crossings. This makes Phase37B a negative diagnostic, not a controller candidate.

## 4. Current Architecture

The current architecture has four conceptual layers:

Layer 1: transfer-family or pre-cross generation

This layer must route initial conditions into target-radius crossing trajectories. Phase35 showed that simple local radial or tangential biases are not enough.

Layer 2: first target-radius crossing

This is a geometric event, not a final success condition. The crossing state must be judged by radial velocity, tangential velocity error, sync error, and compatibility with Phase34 recovery.

Layer 3: Phase34 post-cross synchronization

This is the current terminal controller. Once a crossing exists, Phase34 post-cross synchronization attempts to reduce radius error, radial velocity, and tangential velocity error until the trajectory enters the recoverability basin.

Layer 4: CAPTURE / LOCK / survival

CAPTURE and LOCK are simulator state-machine labels. They are useful internal labels in the 2D sandbox and should not be interpreted as flight-validation states.

## 5. Current Bottleneck

The current bottleneck is crossing-generation.

Phase34 solved the downstream problem for crossing-producing cases in the reduced benchmark. It did not solve non-crossing trajectory families.

Phase35 showed that local upstream steering biases did not create new crossing-producing cases. Phase36B then showed that four interpretable transfer families also did not expand the crossing set beyond `8 / 24`. Phase36C found that geometry metrics can improve or worsen while the remaining cases still fail to cross. Phase37A showed that radial commitment timing alone is not enough to create new crossings. Phase37B showed that weak tangential shaping should not be expanded blindly because it did not create selected-case crossings and failed regression preservation.

The open problem is upstream crossing-generation, not post-cross stabilization for crossing-producing cases.

## 6. Current Hypothesis

Crossing-generation is likely a global trajectory-geometry problem rather than a local steering problem.

It may depend on:

- long-horizon transfer shape
- timing coordination
- energy and angular-momentum evolution
- tangential corridor entry
- controlled coast duration
- family-level trajectory structure

This hypothesis is not yet proven. Phase36B, Phase36C, Phase37A, and Phase37B narrow the search space by showing that local metric movement, manually named transfer-family variants, radial commitment timing, and weak tangential shaping are not sufficient by themselves.

## 7. Next Direction

The next step should not be a new controller. Phase38 should analyze why crossing-basin expansion keeps failing and rank candidate variables before any implementation.

Any future search should keep the grid small and protect the existing crossing-producing cases as regression guards.

Phase34 `radius_priority` post-cross synchronization should remain fixed as the terminal controller. The immediate question is whether any evidence-backed upstream shaping variable can create new target-radius crossings and preserve recoverable handoff behavior in the simplified 2D sandbox.

## 8. What Not To Add Yet

The project should not jump directly to:

- larger PPO or RL systems
- MPC-lite
- direct trajectory optimization as the next default step
- 3D orbital mechanics
- SPICE
- C++ simulation rewrites
- high-fidelity perturbation models

These may matter later, but the current 2D transfer-family question is not mature enough to justify that complexity.

## 9. What This Project Is Not

This project is not:

- real spacecraft readiness
- validated GNC software
- full orbital autonomy
- proof that PPO solves spacecraft control
- proof that Phase34 solves all initial conditions
- evidence that manually defined Phase36B families solve the remaining non-crossing cases
- evidence that Phase37A radial timing solves the remaining non-crossing cases
- evidence that Phase37B weak tangential shaping solves the remaining non-crossing cases

The current evidence is simulator evidence about control architecture in a simplified 2D setting.

## 10. Bottom Line

The current research direction is to define an evidence-based Phase38 search space before implementing any new upstream controller, then test only variables that can plausibly create target-radius crossings without damaging the known crossing-producing cases.
