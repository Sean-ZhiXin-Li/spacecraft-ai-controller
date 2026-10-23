# Observation-driven fuel-aware transfer development v1

Completed: 2026-10-23

Executable prototype: `scripts/closed_loop_transfer_dev_v1.py`.
Reporting: `scripts/report_closed_loop_transfer_dev_v1.py`.

## Working architecture

Mission target and reserve -> current estimated 5D state -> constrained transfer search
-> finite variable-mass departure -> observation -> forecast discrepancy / resource check
-> continue, bounded correction, remaining-transfer replan, or development coast fallback
-> observed target-radius crossing -> finite capture -> 100-day unpowered follow-through.

The planner receives `(x,y,vx,vy,m)`, elapsed days, reserve, deadline, and an optional
correction norm limit. It no longer uses the original initial radius or velocity in its
prediction. Its angular momentum/energy quadrature is independently checked against
DOP853 Cartesian crossing propagation. It returns no candidate when the restricted
outward-transfer family cannot meet the budget and deadline; this is not proof of physical
infeasibility. Inbound, post-target, and turning-point transfer families are still missing.

The initial objective minimizes arrival time. Corrective planning minimizes correction
delta-v subject to terminal capture cost, current mass, operational reserve, arrival time,
and a 120 m/s cumulative requested correction envelope. At most three corrective burns
are attempted. A 0.5 kg planning margin and 2 kg forecast buffer are development choices,
not scientifically validated safety parameters. When the nominal resource allocation
leaves insufficient contingency, v1 plans with at least 30 kg above the 80 kg operational
reserve; this can increase transfer duration.

## Prediction and execution

Planning uses impulses; execution integrates gravity, thrust, and mass continuously with
DOP853. Guidance updates every 100 seconds or less. Unknown thrust magnitude changes
delivered acceleration and mass consumption. The simulation enforces a hardware-style
fuel cutoff at dry mass plus operational reserve, including the final fractional burn.
The cutoff uses actual flow in the plant; the optimizer assumes nominal thrust. This is
an explicit idealization rather than an independent reserve-uncertainty guarantee.

Guidance reads deterministic bounded-biased navigation measurements. Truth is retained
only for propagation and outcome assessment. Mass telemetry is assumed exact. Each pair
uses identical initial perturbations, delivered-velocity errors, navigation bias, thrust
scale, and observation schedule. The baseline reuses the existing nominal two-burn
departure target and local capture law with the same added reserve floor and measurement
interface. It is therefore a fair development baseline, not a rerun of the frozen Phase35
benchmark or an entirely open-loop terminal controller.

Cruise forecasts are checked every 30-120 days; the first post-departure observation is
immediate. Terminal crossing acquisition uses a continuous biased-radius sensor event.
The current model therefore does not demonstrate fully intermittent terminal navigation.
The fresh-seed set's corrections happened immediately after departure. A separate
fixed-departure ablation injected the disturbance after 120 days and executed corrections
from those intermediate states.

## Decision evidence

- Continue when the predicted capture remains affordable and the arrival forecast is acceptable.
- Replan departure using observed state and a declared contingency allocation.
- Request a bounded correction when velocity residual exceeds the navigation envelope plus
  1 m/s or predicted arrival delay exceeds 30 days, provided the constrained search returns
  a feasible nonzero correction.
- Preserve fuel when the declared navigation error exceeds 2 m/s.
- Mark the restricted capture/deadline objective infeasible when no candidate is returned,
  and enter a named development coast fallback. This fallback is not certified safe.

These thresholds were recorded before each development iteration. The sequence includes
failed algorithms and changed experimental contracts, so combined counts must not be
treated as one preregistered robustness benchmark.

## Scope and results

All five iterations, configurations, implementation snapshots, raw records, paired
comparisons, and hashes are under `development_results/closed_loop_transfer_v1_2026-10-23*`.
The project log gives denominators and negative findings. The final fresh-seed comparison
had 16/16 capture and sampled survival in both arms, three actual correction traces, and
zero operational reserve violations. The final 24-pair fixed-departure ablation had 22/24
captures in both arms and four executed cruise corrections, saving 3.74-31.93 days while
using 2.26-18.33 kg additional propellant. This demonstrates executed observation-driven
decisions and a causal timing/fuel tradeoff, but no capture-rate advantage at the matched
reserve policy.

The model remains 2D Sun-only gravity, constant Isp, no attitude or thermal constraints,
known propellant telemetry, and a bounded deterministic navigation bias. The 100-day
survival result is based on 101 samples, not a continuous safety certificate. Crossing,
near-circular capture, survival, and deadline completion are reported separately.

No historical controller, Final Veto, Stage2 authority, or frozen evidence is modified.
The impulsive planner's speed check is a development feasibility condition; it does not
inherit Final Veto protection or constitute a verified continuous-path safety guard.
