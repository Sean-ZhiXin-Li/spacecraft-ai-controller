# Phase 8-19 Research Summary

This summary covers the 2D Phase 8-19 work trail only. It summarizes existing repository outputs and scripts; it does not introduce new physics, rerun experiments, or replace the phase-specific artifacts.

## 1. Current Best Result

Phase 7.6 `soft_linear_3e4` remains the best current 2D controller in this repository.

On the Phase 7.6 local 270-regime grid, `soft_linear_3e4` achieved:

- Success: `217 / 270`
- CAPTURE entries: `217 / 270`
- Near-misses: `8`

The broad Phase 8 result is important but more limited. When the same Phase 7.6 controller was evaluated over an expanded 1296-regime 2D map, it achieved:

- Success: `220 / 1296` (`0.170`)
- CAPTURE entries: `265 / 1296` (`0.204`)
- Radius crossings: `265 / 1296`
- Dominant failure mode: `no_capture_access`

This means Phase 7.6 generalizes to a meaningful but narrow subset of the wider map. It remains the best current 2D controller because Phases 9-19 did not produce a controller that improved CAPTURE or success over the Phase 7.6 baseline on the reduced comparison grid.

Sources: [Phase 7.6 summary](phase76_soft_hybrid/phase76_summary.md), [Phase 8 summary](phase8_multiregime/phase8_summary.md).

## 2. Research Question

Can the Phase 7.6 local soft-hybrid structure generalize to wider 2D regimes?

If not, what prevents global reachability?

The working hypothesis tested across Phases 8-19 was that a local controller might be extended with more pre-CAPTURE structure: energy shaping, geometry correction, burn/coast timing, event guidance, oscillation forcing, trajectory tracking, elliptical transfer logic, crossing-state targeting, or minimal transfer planning.

The result was negative. The obstacle was not post-CAPTURE instability. The dominant obstacle was getting the spacecraft to a useful target-radius crossing in the first place.

## 3. Phase-by-Phase Findings

### Phase 8: Multi-Regime Generalization

- Goal: Test whether Phase 7.6 `soft_linear_3e4` generalizes beyond the local Phase 7 grid.
- Method: Run the unchanged Phase 7.6 controller over an expanded 1296-regime 2D map.
- Result: `220 / 1296` success, `265 / 1296` CAPTURE entries, `1008` `no_capture_access` failures, and `68` near-misses.
- Insight: The controller has a real wider success region, but most failures happen before CAPTURE. The broad-regime problem is reachability, not local CAPTURE/LOCK behavior.

### Phase 9: Pre-CAPTURE Reachability

- Goal: Diagnose Phase 8 no-CAPTURE failures and test whether simple energy-directional reachability terms help.
- Method: Extract no-CAPTURE cases, classify failure patterns, and compare baseline against light/medium DESCENT-only energy-directional variants on a reduced grid.
- Result: Baseline, light, and medium variants all reached `12 / 48` CAPTURE and `12 / 48` success. Improvement was `0`.
- Insight: Most failures never cross the target radius. Simple energy-directional shaping does not change the access boundary.

### Phase 10: Failure-Conditioned Reachability

- Goal: Use the diagnosed failure type to select different DESCENT strategies.
- Method: Switch among `outer_orbit`, `inner_orbit`, `angle_misaligned`, and `near_window`; keep Phase 7.6 behavior inside the near window.
- Result: Baseline and conditioned controller both reached `12 / 48` CAPTURE and success.
- Insight: Geometry diagnosis is useful for labeling failures, but reactive mode selection did not produce new reachable crossings.

### Phase 11: Energy-Guided Planning

- Goal: Test whether specific orbital energy correction can guide nonlocal cases into CAPTURE access.
- Method: Compare current orbital energy to target circular-orbit energy and thrust prograde or retrograde, blending back to Phase 7.6 near the target radius or target energy.
- Result: CAPTURE and success stayed at `12 / 48`; no-CAPTURE cases stayed at `36 / 48`. Overspeed was controlled, but mean energy error increased from `0.0219` initially to `0.2356` finally.
- Insight: Energy alone is not a sufficient transfer objective because it does not determine crossing geometry or phase.

### Phase 12: Angular Momentum + Orbit Intersection

- Goal: Add angular-momentum and radial-crossing geometry to the energy planner.
- Method: Control specific energy, angular momentum, radial velocity, and tangential velocity toward target-radius crossing.
- Result: CAPTURE and success stayed at `12 / 48`; crossing improvement was `0`; no-CAPTURE cases stayed at `36 / 48`; overspeed terminations rose to `36`.
- Insight: Angular momentum can be shaped without solving reachability. Geometry terms need predictive timing, not only instantaneous correction.

### Phase 13: Burn-Coast Guidance

- Goal: Reduce continuous-thrust overspeed by introducing explicit burn/coast timing.
- Method: Burn only when energy error, radial sign, or near-turning logic calls for it; otherwise coast with action `(0, 0)`.
- Result: CAPTURE and success stayed at `12 / 48`; crossing improvement was `0`; overspeed dropped from Phase 12's `36` to `12`.
- Insight: Burn/coast timing improved safety relative to Phase 12 but did not create additional CAPTURE access.

### Phase 14: Event-Triggered Orbital Guidance

- Goal: Trigger burns from orbital events such as turning points or near-target moving-away states.
- Method: Detect events during DESCENT; apply short tangential burns only on detected events; coast otherwise.
- Result: CAPTURE and success stayed at `12 / 48`; overspeed dropped to `0`; events detected were `0`; burns started were `0`.
- Insight: Event-triggered guidance cannot help when useful events do not naturally occur before the trajectory misses the reachable window.

### Phase 15: Oscillation-Inducing Reachability

- Goal: Force radial oscillation to generate new target-radius crossings.
- Method: Apply bounded radial sign/damping control with a small tangential angular-momentum correction.
- Result: CAPTURE and success stayed at `12 / 48`; total crossings stayed at `15`; turning points stayed at `0`; overspeed terminations rose to `36`.
- Insight: Forcing oscillation without a timed transfer objective adds energy risk without creating useful crossings.

### Phase 16: Explicit Trajectory Construction + Tracking

- Goal: Replace reactive oscillation with an explicit desired radial trajectory.
- Method: Track a decaying sinusoidal radius reference with bounded PD radial control and conservative tangential correction.
- Result: CAPTURE and success stayed at `12 / 48`; total crossings stayed at `15`; overspeed dropped to `0`; mean absolute tracking error ratio was `0.0225`.
- Insight: Artificial trajectory tracking improved boundedness and tracking quality, but the reference did not solve reachability within the available horizon.

### Phase 17: Physics-Consistent Elliptical Transfer

- Goal: Use a physically meaningful transfer target rather than an artificial radial reference.
- Method: Construct a transfer ellipse between initial radius and target radius; guide toward transfer energy and angular momentum; blend back to Phase 7.6 near target radius.
- Result: CAPTURE and success stayed at `12 / 48`; crossings stayed at `15`; overspeed was `9`; transfer energy/angular-momentum errors worsened by the final state.
- Insight: A physically plausible transfer shape is better motivated than sinusoidal tracking, but it still needs phasing so the target-radius crossing arrives in a usable CAPTURE state.

### Phase 18: Crossing-State Targeting

- Goal: Improve the quality of target-radius crossings once the trajectory approaches them.
- Method: Reuse Phase 17 far from the target; near crossing, damp radial velocity and correct tangential velocity.
- Result: CAPTURE and success stayed at `12 / 48`; crossings stayed at `15`; overspeed dropped to `0`; crossing `v_r` and `v_t` error ratios did not improve.
- Insight: Crossing-state targeting is safe, but it only activates near crossings. It cannot create crossings for cases that never approach target radius.

### Phase 19: Minimal Transfer Planning

- Goal: Test whether a simple planned injection plus coast can create useful crossings before local targeting.
- Method: Apply a short small tangential injection burn, force at least 20 no-thrust coast steps, then allow Phase 18-style targeting only near a crossing.
- Result: CAPTURE and success fell to `0 / 48`; near-misses were `12`; total radius crossings fell to `0`; targeting steps were `0`; overspeed terminations were `12`.
- Insight: Simple injection plus wait is not real planning. Without predicting the future crossing state, the planner can remove the very crossings that the local controller needs.

## 4. Main Negative Results

- Energy shaping alone failed. Phase 11 preserved overspeed safety but did not increase CAPTURE or reduce no-CAPTURE cases.
- Angular momentum / geometry failed. Phase 12 added orbital geometry terms but did not improve crossings or CAPTURE, and it introduced overspeed.
- Burn-coast timing reduced overspeed but did not improve capture. Phase 13 cut overspeed relative to Phase 12, but CAPTURE and success remained unchanged.
- Event triggering failed because useful events did not naturally occur. Phase 14 detected `0` events and started `0` burns.
- Oscillation forcing caused overspeed. Phase 15 did not create new crossings and produced `36` overspeed terminations.
- Artificial trajectory tracking improved boundedness but not reachability. Phase 16 tracked safely but did not add crossings or CAPTURE entries.
- Elliptical transfer was physically better but still lacked phasing. Phase 17 had a more meaningful transfer objective, but it did not place the crossing in a usable CAPTURE window.
- Crossing-state targeting was safe but could not create crossings. Phase 18 improved overspeed safety but only acted when a crossing was already approaching.
- Naive injection + coast planning degraded performance. Phase 19 removed crossings and reduced CAPTURE/success to `0 / 48`.

## 5. Core Insight

Local control cannot solve a global reachability problem.

Phase 7.6 works when the trajectory is already near the correct local capture window. For nonlocal regimes, the system needs a predictive trajectory planner, not more reactive control layers.

The repeated Phase 9-18 pattern is that local additions can sometimes reduce overspeed, improve boundedness, or make the trajectory more interpretable, but they do not decide when and how the spacecraft will cross the target radius. Phase 19 confirms that even a nominal "planner" is not enough if it does not predict the future crossing state.

## 6. Why Phase 19 Matters

Phase 19 is an important failed result.

It shows that "planning" cannot mean simple injection + wait. The short tangential injection and forced coast did not create target-radius access; they removed the baseline crossings and prevented the targeting layer from engaging at all.

A real planner must predict future crossing state. The relevant planned quantity is not just an initial burn or an energy target. It is the future target-radius crossing: whether it occurs, when it occurs, what the radial velocity is, what tangential velocity error remains, and whether that state lies inside the local Phase 7.6 CAPTURE window.

## 7. Next Direction

Do not add another heuristic reactive controller layer as the next step.

Recommended future work:

- Keep Phase 7.6 `soft_linear_3e4` as the current 2D demo controller.
- Treat Phases 8-19 as evidence that the next problem is predictive planning.
- Use offline trajectory optimization to find feasible transfer paths into the Phase 7.6 capture window.
- Use shooting methods to solve for burns/coasts that produce a desired future crossing state.
- Evaluate MPC / optimal control formulations that explicitly optimize future crossing time, crossing radial velocity, tangential velocity error, and control effort.
- Keep CAPTURE/LOCK physics and success definitions unchanged while testing planners, so new results remain comparable.

The next research question is not "which local correction should be added?" It is "which planned trajectory reaches the Phase 7.6 local capture basin?"

## 8. README Note

Recommended one-line README note:

> Phase 8-19 summarize failed global-reachability attempts and motivate predictive planning.

This note should link to this summary if added to the README.
