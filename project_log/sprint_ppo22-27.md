# Sprint PPO 22–27: From Oscillation Forcing to Predictive Planning Boundary

## 1. Sprint Objective

After the Phase 14 event-guidance failure, the central question changed:

Can local control layers create new reachability?

The earlier PL22-PL27 path had already shown that explicit, phase-structured local control can work very well inside a narrow 2D regime. WS-1 improved CAPTURE access, adaptive WS reduced near-misses, pre-window shaping widened reachability, hard switching failed, and Phase 7.6 `soft_linear_3e4` became the current best 2D controller with `217 / 270` successes, `217` CAPTURE entries, and `8` near-misses.

Phase 8 then showed the limit of that result on a broader map: the dominant failure mode became `no_capture_access`. The focus therefore shifted from local stabilization to global reachability. Phases 15-19 tested whether increasingly structured explicit controllers could create useful target-radius crossings before CAPTURE.

## 2. Phase 15 — Oscillation-Inducing Reachability

Goal: Can forced oscillation create crossings?

Method: Phase 15 added bounded radial sign/damping control in DESCENT, with a small tangential correction to preserve angular momentum. The intent was to push inward when outside the target radius, push outward when inside, and create radial oscillation around the target orbit.

Key Result: CAPTURE and success remained `12 / 48`, unchanged from both the Phase 7.6 baseline and Phase 14. Total radius crossings stayed at `15`, and turning points stayed at `0`.

Main Failure: Overspeed became worse. Phase 14 had `0` overspeed terminations; Phase 15 produced `36`.

Insight: Directly forcing radial oscillation without a timed transfer target adds energy risk but does not create useful orbital events. Oscillation is not something a local sign controller can reliably manufacture from nonlocal geometry.

## 3. Phase 16 — Explicit Trajectory Tracking

Goal: Can artificial trajectories create bounded transfer?

Method: Phase 16 replaced direct oscillation forcing with an explicit desired radial trajectory:

```text
r_desired(t) = target_radius + A(t) * sin(omega * t)
```

The amplitude decayed slowly, and a bounded PD controller tracked desired radius and radial velocity. A conservative tangential term maintained circular-speed consistency.

Key Result: CAPTURE and success remained `12 / 48`. Total crossings stayed at `15`, and turning points stayed at `0`.

Main Failure: The artificial sinusoidal reference improved boundedness but did not create new target-radius access. The mean absolute tracking error ratio was `0.0225`, overspeed dropped to `0`, and mean max speed ratio stayed controlled at `1.0538`, but reachability did not move.

Insight: A trackable reference is not necessarily a feasible transfer. Phase 16 clarified that bounded tracking quality is not the same as reaching a useful CAPTURE entry state.

## 4. Phase 17 — Physics-Consistent Elliptical Transfer

Goal: Can physically valid orbit targets solve reachability?

Method: Phase 17 constructed a transfer ellipse between the initial radius and the target radius. It guided DESCENT toward transfer-orbit specific energy and angular momentum, then blended back to Phase 7.6 behavior near the target radius.

Key Result: CAPTURE and success remained `12 / 48`. Crossings stayed at `15`, turning points stayed at `0`, and no-CAPTURE cases stayed at `36 / 48`.

Main Failure: The physically valid target did not produce a usable crossing phase. Overspeed rose to `9`, and transfer energy/angular-momentum errors worsened by the final state.

Insight: Matching a plausible transfer orbit is not enough. The controller must also phase the trajectory so target-radius crossing occurs with acceptable radial velocity and tangential velocity error.

## 5. Phase 18 — Crossing-State Targeting

Goal: Can crossing-state quality solve capture?

Method: Phase 18 reused Phase 17 far from the target. When the trajectory was approaching target-radius crossing, it switched to targeting mode: damp radial velocity first, then softly correct tangential velocity toward circular speed.

Key Result: CAPTURE and success remained `12 / 48`. Total radius crossings stayed at `15`. Overspeed dropped from Phase 17's `9` to `0`.

Main Failure: Crossing quality did not improve. Mean `|v_r| / v_circ` stayed at `0.0120`, and mean `|v_t error| / v_circ` stayed at `0.8721`. The targeting mode only engaged when a crossing was already approaching.

Insight: Crossing-state targeting is useful as a safety/stabilization idea, but it cannot create crossings. It acts too late to solve no-CAPTURE access.

## 6. Phase 19 — Minimal Transfer Planning

Goal: Can minimal planning outperform reactive control?

Method: Phase 19 tested a simple planned sequence: short tangential injection, forced coast, then Phase 18-style targeting only if a crossing was being approached. This was the first attempt to move beyond purely reactive local layers.

Key Result: Performance degraded. Baseline and Phase 18 each had `12 / 48` CAPTURE and success; Phase 19 had `0 / 48` CAPTURE and `0 / 48` success. Total radius crossings fell from `15` to `0`, targeting steps were `0`, and overspeed terminations rose to `12`.

Main Failure: The injection/coast sequence did not predict the future crossing state. It removed the baseline crossings instead of creating better ones.

Insight: Naive planning is worse than reactive control when it is not predictive. A burn plus wait is not a transfer planner; it must be tied to a forecast of crossing time, crossing radial velocity, and tangential velocity error.

## 7. Sprint-Wide Findings

- Local heuristics repeatedly failed to expand global reachability.
- Safety improved in some phases: Phase 16 controlled overspeed relative to Phase 15, and Phase 18 eliminated Phase 17 overspeed.
- Reachability did not improve: Phases 15-18 stayed at `12 / 48` CAPTURE and success.
- Phase 19 proved naive planning can be worse than reactive control: CAPTURE and success dropped to `0 / 48`.
- Phase 7.6 `soft_linear_3e4` remains the strongest current 2D controller result.

The sprint produced a consistent negative result: each added layer clarified one missing property, but none expanded the reachable set beyond the local Phase 7.6 baseline on the reduced comparison grid.

## 8. Core Research Insight

Local control cannot solve a global reachability problem.

The boundary between reactive control and planning became clear. Reactive controllers can stabilize, damp, shape, or protect trajectories that already approach the capture window. They cannot decide the future orbital crossing state when the spacecraft starts outside the local basin.

Reachability requires predictive crossing-state planning. The planned object is not just thrust direction, energy, angular momentum, or a reference radius. It is the future state at target-radius crossing: whether the crossing occurs, when it occurs, what `v_r` is, what `v_t - v_circular` is, and whether that state lies inside the local Phase 7.6 capture basin.

## 9. Technical Growth

This sprint improved the engineering model of the problem even though the controller results were negative.

Failure classification separated no-CAPTURE access from post-CAPTURE stabilization. That prevented wasted work on LOCK behavior when the system was failing before CAPTURE.

Controller architecture evolved through increasingly explicit structure: oscillation forcing, trajectory tracking, physically motivated transfer invariants, crossing-state targeting, and injection/coast planning. Each step preserved physics and CAPTURE/LOCK definitions, which kept comparisons meaningful.

Physics-consistency mattered, but it was not sufficient. Phase 17 showed that valid orbital quantities can still fail without timing and phasing.

The negative results mattered because they ruled out a broad class of tempting fixes. More local heuristics, more damping, more radial forcing, and simple burn/coast scheduling do not solve the global access problem.

## 10. Next Direction

The next step should not be another heuristic reactive layer.

Recommended direction:

- Predictive planning around target-radius crossing state.
- Shooting methods that solve for burns/coasts producing a desired future crossing.
- Trajectory optimization with explicit constraints on crossing time, `v_r`, `v_t`, speed safety, and control effort.
- MPC or optimal-control formulations that use Phase 7.6 as the terminal/local capture controller.

The next research question should be:

Which planned trajectory reaches the Phase 7.6 local capture basin?
