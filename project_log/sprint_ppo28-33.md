# Sprint Log — Phase28 to Phase33
## From Trajectory Family Mapping to Optimal Structure Extraction

## 1. Sprint Objective

Phase28 began after Phases 22-27 exhausted the late-insertion hypothesis.

By that point, the project had already shown that Burn A could create insertion windows, Burn B could be made more deterministic, tangential velocity could be corrected locally, and timing-aware insertion logic could be added. None of those changes produced recoverable crossings, improved CAPTURE, or improved success on the reduced benchmark. The repeated result was structurally stable: crossings existed, windows existed, but recoverability did not.

The sprint therefore moved away from asking which immediate controller should be tuned. The new question was whether the system was selecting the wrong trajectory family in the first place, and whether useful orbital insertion required a structure that the existing staged controllers could not express.

This made family mapping, Burn-A-end geometry, global transfer search, and optimal-trajectory structure extraction necessary. Local improvements were no longer enough because the failure was no longer isolated to one controller module. It involved the relationship between early orbital family, crossing geometry, recoverability basin entry, and the timing of radius, radial velocity, and tangential velocity alignment.

## 2. Phase-by-Phase Breakdown

### Phase28

#### Goal

Map historical trajectory families and determine whether window-producing cases and crossing-producing cases were actually the same kind of orbital geometry.

#### What was tested

Phase28 analyzed existing CSV outputs from Phases 22, 23, 24, 25, 26, and 27 without rerunning controllers or modifying physics, thresholds, CAPTURE, LOCK, reward, or prior outputs. It normalized `576` rows across `48` unique initial cases and labeled trajectories as `dead_geometry`, `window_no_crossing`, `crossing_bad_sync`, or `near_recoverable_crossing`.

#### Key result

The dataset contained `144` crossing rows and `132` window rows. All `132` window rows were dead windows; good windows were `0`. Best-controller family labels by case were `36` dead geometry, `6` crossing bad sync, and `6` near-recoverable crossing.

#### Failure / limitation

Phase28 could not prove a causal Burn A rule because Burn-A-end geometry was mostly missing from prior CSVs. It could describe family separation, but it could not yet directly instrument why Burn A selected those families.

#### Structural insight

Window existence is not useful orbital geometry. Window-producing and crossing-producing families were mostly disjoint. The project needed to stop treating insertion-window creation as progress unless that window also led to controllable crossing geometry.

### Phase29

#### Goal

Audit the repository and test whether a Burn-A family selector could intentionally target the trajectory family that historically correlated with useful crossings.

#### What was tested

Phase29 first produced a full repo audit covering README narrative, project logs, Phase 20-28 outputs, and controller scripts. It then ran a 48-case reduced benchmark with Burn-A selector variants: baseline Phase22 Burn A, crossing-family targeting, angular-momentum targeting, balanced energy/angular-momentum targeting, and dead-window avoidance. Burn-A-end geometry was directly instrumented.

#### Key result

All variants stayed at `12` crossings, `5` near-recoverable crossings, `0` recoverable crossings, `12` CAPTURE, and `12` success. Some variants changed Burn-A-end energy or angular momentum substantially, but downstream crossing structure did not move.

#### Failure / limitation

The Phase28 wrong-family hypothesis was not supported by simple selector rules. Dead windows were not reduced below the baseline count of `12`, and crossing-producing trajectories did not increase.

#### Structural insight

Historical correlation was not enough to design Burn A. If early family selection mattered, it could not be captured by shallow Burn-A heuristics. The next test had to search the endpoint manifold explicitly.

### Phase30

#### Goal

Replace heuristic Burn-A family selection with explicit Burn-A endpoint optimization over the post-Burn-A orbital manifold.

#### What was tested

Phase30 searched bounded Burn-A endpoint candidates over duration, thrust norm, radial bias, and tangential bias. It mapped `6912` endpoint candidates and selected endpoints using physical diagnostics such as energy, angular momentum, periapsis, apoapsis, target crossing potential, and preview crossing quality.

#### Key result

Selected endpoint previews with passive crossing numbered `9`, and the endpoint manifold was successfully mapped. Downstream performance did not improve. Best variants still had `12` crossings, `5` near-recoverable crossings, `0` recoverable crossings, `12` CAPTURE, and `12` success.

#### Failure / limitation

Endpoint search altered Burn-A-end geometry but did not alter the downstream crossing-state structure. Dead windows were not reduced, near-recoverable crossings did not increase, and CAPTURE/success did not improve.

#### Structural insight

A better endpoint is not necessarily a better transfer. Selecting a post-Burn-A state does not solve the full trajectory path from initial state to recoverable insertion geometry. Endpoint selection remained too local in time.

### Phase31

#### Goal

Test whether named global transfer architectures could produce better crossing-state structure than Burn-A/B heuristics or endpoint selection.

#### What was tested

Phase31 searched bounded global transfer families over first burn timing, first burn magnitude and direction, coast duration, second burn timing, second burn magnitude and direction. It tested direct transfer, Hohmann-like transfer, Lambert-like transfer, energy-ladder transfer, and baselines. It mapped `576` transfer candidates, with `21` selected transfer previews showing crossings.

#### Key result

The original Phase22-style baseline remained best by crossing count and quality: `12` crossings, `5` near-recoverable crossings, `0` recoverable crossings, `12` CAPTURE, and `12` success. Direct, Lambert-like, and energy-ladder variants generally reduced crossings to `8`; Hohmann-like transfer produced `0`.

#### Failure / limitation

Global transfer families did not reduce dead windows, did not improve recoverable crossings, did not improve CAPTURE, and did not improve success. A mean sync improvement existed in some variants, but without crossing/CAPTURE improvement it was not an architecture breakthrough.

#### Structural insight

Named burn/coast transfer classes still lacked something essential: continuous synchronization. They chose architecture-level templates, but they did not continuously coordinate radius, radial velocity, and tangential velocity over the whole horizon.

### Phase32

#### Goal

Establish an optimal-control baseline: under current physics, can continuous trajectory optimization produce a better recoverability state than all heuristic architectures?

#### What was tested

Phase32 built a coarse finite-horizon direct optimal-control prototype with state `[x, y, vx, vy]` and bounded thrust controls. CasADi was unavailable in the checked runtime, so the phase used SciPy direct shooting. It tested objective modes including radius-only, recoverability target, sync-error minimization, and fuel-constrained recoverability over `512` physics steps and `64` control intervals.

#### Key result

Optimal control outperformed heuristic architectures as an upper-bound prototype. `recoverability_target` solved `4 / 4` cases, produced `1` crossing, `1` near-recoverable crossing, `1` recoverable crossing, `2` recoverable states, and the best mean recoverability distance. `sync_error_minimization` also produced `1` recoverable crossing.

#### Failure / limitation

This was not a production controller. CAPTURE improvement was not directly evaluated as a closed-loop CAPTURE rollout. CasADi/IPOPT collocation was not available, so the result was a SciPy direct-shooting prototype rather than the intended full direct-collocation baseline.

#### Structural insight

Recoverability is physically reachable as a state under current dynamics, at least in the coarse optimal-control solve. Prior failures were therefore plausibly architecture failures rather than proof of physical impossibility.

### Phase33

#### Goal

Reverse engineer the best Phase32 trajectory and identify what structural behavior optimal control used that prior architectures lacked.

#### What was tested

Phase33 selected the best Phase32 case, decomposed its trajectory into phases, compared Phase31 and Phase32 behavior, and extracted control motifs. The best case was `recoverability_target / baseline_crossing_high_angle`, with crossing step `81`, best sync `0.000464`, and best distance `0.000470`.

#### Key result

The best trajectory crossed target radius early, but the crossing state itself was not recoverable. At crossing, sync error was `1.676881` and distance to recoverable was `2.313443`. The best recoverability state occurred much later at step `512`, after a long smooth low-thrust correction arc. Tangential velocity error was minimized after the first crossing, not before it.

#### Failure / limitation

Phase33 did not produce a controller. It extracted structure from an optimal trajectory. It also clarified that Phase32's `recoverable_crossing` label meant a trajectory both crossed and later reached recoverability, not that the first crossing state itself was recoverable.

#### Structural insight

First crossing is not insertion. The missing law was continuous post-cross synchronization: crossing, then smooth steering, then late alignment of radius, radial velocity, and tangential velocity inside the recoverability basin.

## 3. Major Scientific Turning Points

### Turning Point 1: Trajectory families > isolated controllers

Phase28 reframed the problem from controller tuning to trajectory-family classification. The key discovery was that insertion windows, crossings, near-recoverable states, and dead geometries were not interchangeable outcomes of one controller. They represented different families in the reachable trajectory manifold.

This changed the research object. The project was no longer asking whether Burn B should be stronger or whether tangential velocity should be corrected earlier. It was asking which orbital families are worth controlling at all.

### Turning Point 2: Crossing geometry > crossing count

The repeated `12` crossing cases across many variants made crossing count a weak metric by itself. Phase28 and Phase31 showed that crossing count could stay constant while sync quality, window count, or transfer geometry changed. Phase33 made the point sharper: even the best optimal trajectory's first crossing was outside the recoverability basin.

The relevant object became crossing-state geometry: radius error, radial velocity, tangential velocity error, and distance to recoverability.

### Turning Point 3: Global transfer still lacked continuous synchronization

Phase31 was the strongest explicit transfer architecture before optimal control. It tested named transfer families with bounded burn/coast schedules and previewed crossing geometry. It still failed to improve recoverable crossings, CAPTURE, or success.

That negative result showed that architecture-level burn sequencing was not enough. The missing structure was not simply direct transfer versus Hohmann-like transfer versus Lambert-like transfer. The missing structure was continuous state-control synchronization across the trajectory.

### Turning Point 4: Optimal control revealed post-cross synchronization structure

Phase32 proved that a recoverable state could be reached under unchanged physics. Phase33 explained how: the successful trajectory did not treat the first radius crossing as the insertion point. It crossed early, then used a long low-authority steering arc to align `r`, `v_r`, and `v_t` later.

This was the sprint's most important architectural discovery. The controller should not stop thinking when it reaches target radius. The crossing is a transition into a synchronization regime.

## 4. Biggest Negative Results

Several ideas looked plausible and were ruled out or weakened.

- Dead windows were not useful progress. Phase28 found `132` window rows and `0` good windows.
- Simple Burn-A family selection did not improve anything. Phase29 changed early geometry but left crossings, recoverability, CAPTURE, and success unchanged.
- Better Burn-A endpoints were not enough. Phase30 mapped `6912` endpoint candidates, but selected endpoint families still did not improve downstream structure.
- Named global transfer families were not enough. Phase31 mapped transfer families and selected candidates, but recoverable crossings remained `0`.
- Mean sync changes without crossing or CAPTURE movement were not architecture breakthroughs. Phase31 had some sync movement, but it did not change the actual performance frontier.
- Radius-only optimal control was not enough. Phase32's `radius_only` objective produced no crossings and no recoverable states.
- Phase32 was not a production win. It was an upper-bound result from SciPy direct shooting, not a deployed controller or a validated CAPTURE rollout.

The sprint's negative results were productive because they eliminated increasingly broad classes of fixes: late Burn B tuning, simple family selectors, endpoint optimization, and bounded named transfer templates.

## 5. Core Discovery of Sprint

First crossing is not insertion.

Before this sprint, the research path often treated target-radius crossing as the event that should enable capture. Phase25 had already shown that crossing-state tangential velocity was a dominant blocker, and Phases26-27 showed that local vt correction and timing sync did not move the crossing-state distribution. Phase33 explained why that framing was incomplete.

The first crossing state can be close in radius but still outside the recoverability basin. In the best Phase32 trajectory, the first crossing occurred at step `81`, but crossing sync was `1.676881` and distance to recoverability was `2.313443`. The best recoverable state occurred at step `512`, after a long smooth correction arc.

Recoverability is not just radius. It is the simultaneous basin condition:

```text
radius near target
radial velocity near zero
tangential velocity near circular
```

The missing architecture before Phase34 was a post-cross synchronization mode. Earlier controllers treated crossing as an endpoint, a handoff trigger, or a CAPTURE opportunity. The optimal trajectory treated crossing as a midpoint. It allowed temporary imperfection and kept steering until radius, radial velocity, and tangential velocity aligned together.

## 6. Architecture Evolution Summary

### Phase28: Family map

The project mapped historical outcomes into trajectory families and separated dead windows, crossing-bad-sync cases, near-recoverable crossings, and dead geometry.

### Phase29: Repo-wide structural audit

The project audited prior phases, separated controller phases from diagnostic phases, clarified which results were true performance results, and tested a first Burn-A family selector.

### Phase30: Burn-A endpoint geometry

The project moved from heuristic Burn-A behavior to explicit post-Burn-A endpoint manifold search over duration, thrust norm, radial bias, and tangential bias.

### Phase31: Global transfer families

The project moved from endpoint selection to complete transfer-family templates with burn timing, burn magnitude, burn direction, coast duration, and crossing-state preview.

### Phase32: Direct optimal upper bound

The project moved from explicit transfer templates to finite-horizon continuous control optimization under unchanged 2D physics.

### Phase33: Structural law extraction

The project moved from asking whether optimal control could do better to asking what optimal control actually did. The extracted law was post-cross smooth synchronization.

## 7. End-of-Sprint Conclusion

### 1. What did this sprint actually prove?

It proved that the repeated failure of Phases 22-31 was not merely a lack of controller variants. The project mapped trajectory families, tested Burn-A selectors, tested endpoint manifolds, tested global transfer templates, and still could not improve recoverable crossings until direct optimal control optimized the full state-control trajectory.

### 2. What architecture limit became undeniable?

The staged architecture was incomplete. Burn A -> coast -> Burn B -> handoff treated insertion as a sequence of discrete events. The successful optimal structure required continuous synchronization after crossing, not just a better crossing trigger.

### 3. Why were earlier controllers fundamentally incomplete?

Earlier controllers lacked permission and mechanism to continue smooth low-authority steering after the first radius crossing while tolerating temporary radius imperfection. They optimized local safety, window creation, endpoint geometry, or transfer templates, but not the full late alignment of radius, radial velocity, and tangential velocity.

### 4. Why is Phase34 the logical next step?

Phase33 extracted a deployable hypothesis: crossing should start a post-cross synchronization controller, not immediately end the transfer problem. Phase34 should test whether a hand-built explicit controller can imitate the Phase32 motif: crossing -> smooth post-cross steering -> recoverability basin entry.

## Sprint Bottom Line

This sprint transitioned the project from controller iteration to control architecture science.
