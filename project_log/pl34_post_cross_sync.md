# PL34 — Crossing Is Transition, Not Success

## Phase34 Post-Cross Smooth Synchronization Controller

## 1. Objective

Phase34 was not designed to create more target-radius crossings. The experiment held the early transfer behavior close to the Phase22/31-style explicit controller and asked a narrower question:

Can trajectories that already produce a target-radius crossing be converted into recoverable trajectories by changing the controller architecture after crossing?

That distinction matters because the project had already shown that a geometric crossing can be misleading. A vehicle may reach the target radius while carrying the wrong radial velocity, tangential velocity, or post-cross state alignment. Phase34 therefore treats crossing as the start of the most important control work, not as the finish line.

The tested structure was:

`crossing -> post-cross smooth synchronization -> recoverability basin -> CAPTURE -> LOCK`

## 2. Background

Phase31 demonstrated that the explicit transfer stack could create crossings on the reduced benchmark, but those crossings were not dynamically recoverable. In the Phase31 reference set, 8 of 24 cases crossed the target radius, yet 0 of those crossings reached the recoverability basin.

Phase32 showed that recoverability was physically reachable in the simulator. Its direct optimal-control work, even with SciPy fallback rather than a full CasADi/IPOPT solve, exposed trajectories that could move into the desired state neighborhood.

Phase33 extracted the important hidden structure from those optimal runs: recoverable behavior required post-cross synchronization, not only pre-cross transfer shaping. The controller had to align radius error, radial velocity, and tangential velocity after the crossing event.

Phase34 turned that observation into an explicit controller architecture. It made post-cross synchronization a named control phase instead of leaving recovery to the same transfer logic that created the crossing.

## 3. Method

The Phase34 controller preserves the Phase22/31-style early transfer logic so the comparison remains focused on the downstream architecture. It then detects the first target-radius crossing and switches into a post-cross synchronization mode.

After crossing, the controller applies smooth low-thrust correction rather than a discontinuous terminal maneuver. The correction is evaluated against joint state alignment:

- radius error
- radial velocity
- tangential velocity error

Three post-cross modes were tested:

- `radius_priority`
- `sync_balanced`
- `vt_priority_then_sync`

The goal was not to tune for a visually satisfying crossing. The goal was to test whether explicit post-cross synchronization can move crossing-producing trajectories into the recoverability basin without relaxing the simulator's CAPTURE/LOCK thresholds or physics settings.

## 4. Results

Phase31 reference:

- 24 cases
- 8 geometric crossings
- 0 recoverable crossings

Phase34 best mode, `radius_priority`:

- 24 cases
- 8 geometric crossings
- 8 recoverable crossings
- crossing-case best distance improved from 3.9923 to 0.9855 (lower is better)
- overspeed: 0

The source CSV also reports the `success` column as true for 8 rows in the Phase31 reference and 8 rows in the Phase34 best mode. This log treats that as a legacy simulator label and uses geometric crossings and recoverable crossings as the primary scientific comparison.

The crossing count did not increase. The important change is that every crossing-producing Phase34 case reached the recoverability basin later in the post-cross arc.

The README demo artifact is consistent with the same control-state vocabulary: one explicit-controller sandbox run records 1 radius crossing, first crossing at step 48,269, final radius error of 27,657.63 m, and phase transitions `DESCENT -> CAPTURE` and `CAPTURE -> LOCK`. That demo is visual context, not a replacement for the 24-case benchmark.

## Structural Delta

Phase34 did not expand the crossing set.
Phase34 changed the fate of existing crossings.

## 5. Interpretation

Phase34 did not solve the non-crossing trajectory families. Sixteen of the 24 benchmark cases still did not produce a target-radius crossing, so they remain outside the solved region of this phase.

What Phase34 solved is the downstream problem: what to do after crossing exists. The result supports the claim that post-cross synchronization is a missing deployable control law in the earlier explicit-controller stack. Phase31 could hit the geometric event but failed to convert it into recoverability. Phase34 keeps the same crossing count and changes the post-cross state evolution.

This is architecture progress, not just metric tuning. The key result is not a larger number of crossings. It is the conversion of crossing-producing cases into recoverable cases under unchanged benchmark thresholds.

## 6. Limitations

This result is bounded by the reduced 24-case benchmark. It should not be read as universal orbital insertion or full end-to-end autonomy.

The main unsolved limitation is still the crossing basin. Phase34 leaves 16 non-crossing cases unresolved, which means the controller cannot yet drive all sampled initial conditions into the post-cross synchronization regime.

The work is simulator-only. It is a 2D physics-based orbital control sandbox, not real spacecraft readiness, not flight validation, and not a claim about deployable hardware.

The `recoverable_crossing` label also has a precise meaning: a crossing occurred and the later post-cross synchronization arc reached the recoverability basin. It does not mean the first crossing instant itself was already recoverable.

## 7. Next Step

Phase35 should focus on crossing basin expansion. The next architecture question is how to turn more non-crossing initial conditions into crossing-producing cases while keeping Phase34 as the terminal/post-cross controller.

That implies a two-part controller stack:

- a pre-cross module that expands the set of cases that can reach the target radius
- a Phase34-style post-cross module that converts those crossings into recoverable CAPTURE and LOCK behavior

## 8. Bottom Line

Phase34 changes the project’s definition of success: crossing is not the goal; crossing is the transition into post-cross synchronization.
