# Position-aided estimation restores capture under biased navigation

Completed: 2026-10-23

Starting HEAD: `39d3766f153bce12932fb3291db0746b5f604703`.
Environment: `E:/conda3/envs/spacecraft/python.exe`; SciPy 1.16.3, NumPy 2.3.5.

## Research question and objective

The v1 controller could improve transfer time but had no capture-rate advantage in its
strongest comparison. Investigation of the actual records separated two failure modes:
cases 43/44 had approximately 3 kg predicted capture-reserve shortfalls with small
navigation bias; the earlier seed-10 case had adequate fuel but failed with biased
velocity measurements. Those are distinct limitations.

The objective fixed before the discovery experiment was to improve near-circular capture
under persistent 5-15 m/s per-axis velocity bias while holding departure burns, disturbance
draws, propulsion, deadlines, operational reserve, and terminal predicates equal. The
selected intervention changes the navigation estimate delivered to terminal guidance.
It does not add propellant, extra maneuver opportunities, a longer deadline, or new
capture thresholds.

## Implemented architecture

`scripts/estimated_capture_dev_v2.py` contains the executable simulator/experiment runner,
measurement-only `CoastBiasEstimator`, finite-burn measurement adapter, and two estimators:

1. Across an unpowered coast interval, solve a two-dimensional shooting problem for the
   velocity bias whose model-propagated position matches the next measured position.
2. Correct subsequent measured velocity by subtracting that inferred bias.
3. For drifting sensors, fit bias and bias rate to the six most recent fitted bias
   observations. Use their actual reference epochs and weight by coast-interval duration;
   shorter position-difference baselines have higher velocity uncertainty.
4. Extrapolate the bias to the current measurement epoch and supply the corrected velocity
   to the existing finite-thrust capture guidance.

The estimator accepts only measurement vectors, timestamps, and the declared gravitational
model. It never reads the sensor's injected bias or simulated true state. Hidden truth is
used by plant propagation and post-run scoring. Mass telemetry is assumed exact. Constant
position bias largely cancels in displacement; gravity evaluated at biased position still
creates a small modeled error, so perfect cancellation is not claimed.

This is a two-body coast shooting estimator with temporal bias regression, not an EKF or
a general navigation solution. The recorded displacement-derived bound is a development
diagnostic, not a formal confidence interval or active safety certificate.

## Paired experimental design

Both arms use the same raw-sensor departure burn and the same precomputed 110 kg reserve
plan. Initial velocity error, a physical velocity disturbance at day 120, thrust scale,
and sensor observations are paired. No cruise correction is executed in either arm: this
isolates estimation during capture from v1's transfer-replanning effect. The same v1
finite-burn routine integrates all powered motion and cuts off at 300 kg dry mass plus
80 kg operational reserve, including final partial burns.

Capture requires converged guidance, eccentricity `<0.001`, and radius-error ratio
`<=0.0025`. The deadline is 1100 days. Survival is checked at 101 samples over the next
100 unpowered days; this is sampled evidence, not continuous safety verification.

All scenario arrays and source snapshots were written before executing each set.

| Experiment | Paired conditions | Reference | Improved | Capture counts |
|---|---:|---|---|---|
| Discovery | 11 | Raw velocity | Static-bias estimate | 0/11 -> 9/11 |
| Unseen constant-bias validation | 24 | Raw velocity | Static-bias estimate | 4/24 -> 24/24 |
| Drift stress | 24 | Raw velocity | Static-bias estimate | 0/24 -> 19/24 |
| Drift adaptation, development seeds | 8 | Static estimate | Bias-rate tracking | 3/8 -> 8/8 |
| Unseen drift holdout | 24 | Static estimate | Bias-rate tracking | 8/24 -> 24/24 |

Comparison experiments: 182 mission simulations in 91 pairs; baseline reproduction and
unit-test integrations are additional verification. These sets have different contracts
and some reused development seeds; their success counts must not be pooled into a single
validation rate. Each validation/holdout set has twelve independent seeds and two related
conditions per seed, not 24 independent seeds.

## Generalization evidence

Constant-bias validation used seeds 101-112, initial/delivered velocity bounds +/-100 m/s,
initial velocity bias bounds +/-5 and +/-15 m/s, constant position bias up to 500 km,
additional bounded time-varying position errors up to 200 km, actual thrust 70% of nominal,
and 30-day observations. Both arms retained the same measurements and capture predicates.
The estimator rescued twenty failed conditions, caused no capture regressions, and reduced
worst eccentricity from 0.006888 to 0.0001814. Worst post-run bias-estimation error was
0.535 m/s. Extra capture fuel relative to raw navigation was at most 1.276 kg; some cases
used less fuel. Every improved capture passed sampled survival.

The static estimator then failed five of eight 0.2 m/s/day drift stress conditions. This
was diagnosed as estimator lag, not a propulsion failure. The bias-rate extension repaired
all five without changing the plant or terminal predicate.

The final holdout used unseen seeds 301-312, +/-150 m/s physical velocity disturbances,
initial +/-15 m/s velocity bias, drift rates 0.2 and 0.4 m/s/day, the same position noise,
60% actual thrust, and 30-day observations. The drift accumulates throughout the mission;
15 m/s is the initial bias bound, not the final bound. Capture and sampled survival rose
from 8/24 to 24/24 with zero regressions. Worst eccentricity fell from 0.005585 to
0.00003369. Maximum post-run velocity-bias error was 0.0919 m/s. The extra fuel cost was
0.033-0.745 kg, and the minimum final fuel across both arms was 90.597 kg, above the
unchanged 80 kg operational floor.

## Baseline fidelity and remaining failures

A numerical test re-executes the raw-sensor case 43 and checks its final state against
the frozen v1 revision-5 record. Both new arms have identical initial and post-departure
states and physical disturbance draws. Root-shooting correctness, bias-rate time stamps,
measurement ownership, invalid inputs, reversed clocks, reserve compliance, and snapshot
hashes are tested separately.

The discovery set intentionally included v1 cases 43 and 44. Neither is recovered by
estimation: both still reach the fuel floor before capture. Better state knowledge cannot
create missing propellant. The retained diagnostic shows why a trajectory-family/resource
intervention remains necessary for those states.

Another important failure is retained in the drift stress set: a constant-bias estimator
can report a small shooting residual while its current-time velocity estimate is wrong.
The measurement epoch and bias evolution matter; fit residual alone is not navigation
confidence.

## Limits of the result

- The estimator and plant share the two-body coast model. No unknown gravity, sustained
  acceleration, third body, or within-interval unknown thrust was tested.
- The improved model assumes approximately linear velocity-sensor drift. Abrupt jumps,
  nonlinear drift, position-sensor drift, outages, and outliers remain unvalidated.
- Powered-arc measurements hold their sensor epoch and bias estimate during the short
  burn. Bias variation over those minutes is neglected in this development model.
- Position sensing is much more reliable than velocity sensing in the tested assumption.
  This is a declared sensor model, not a statement about actual navigation hardware.
- The new runner isolates terminal estimation. The reusable estimator is not yet wired
  into v1's cruise action selection or a production estimator/controller interface.
- Target crossing, captured orbit, reserve compliance, and sampled survival remain
  distinct. A biased radius event does not itself establish true geometric crossing.
- The estimator does not improve the planner's inbound/post-target trajectory coverage.

## Reproduction and artifacts

```powershell
E:/conda3/envs/spacecraft/python.exe -m scripts.estimated_capture_dev_v2 --experiment discovery
E:/conda3/envs/spacecraft/python.exe -m scripts.estimated_capture_dev_v2 --experiment validation
E:/conda3/envs/spacecraft/python.exe -m scripts.estimated_capture_dev_v2 --experiment stress
E:/conda3/envs/spacecraft/python.exe -m scripts.estimated_capture_dev_v2 --experiment adaptation
E:/conda3/envs/spacecraft/python.exe -m scripts.estimated_capture_dev_v2 --experiment holdout
E:/conda3/envs/spacecraft/python.exe -m scripts.report_estimated_capture_dev_v2
E:/conda3/envs/spacecraft/python.exe -m unittest -q Tests.test_estimated_capture_dev_v2 Tests.test_closed_loop_transfer_dev_v1
```

Each numerical command refuses to overwrite its result directory. Historical experiments
include their executed implementation snapshot. The last source-only change adds invalid
measurement/time guards; valid-input numerical behavior is unchanged.

Results are in `development_results/estimated_capture_v2_2026-10-23_<experiment>/`:
pre-execution configuration, executed source snapshot, raw runs, summary, offline paired
comparison, and artifact hashes. Validation and holdout also include comparison figures.

## Preservation and next decision

All five experiments preserved the same 773 frozen tracked files:
`ad532df0291e04b73a96bdf3bce6ff1354ddd55dedf5ee1d1453825244319c0c`.
The 36 historical v1 result/source files also remained byte-identical, aggregate
`8d0f2ba1bf607f25e21f1b93bf79cd8d1ecd2cb84f39d271f3ae52931cd1ee69`.
No existing controller, Final Veto, benchmark threshold, Stage2 authority, or historical
artifact was modified. No commit or push was performed.

The strongest advance is capture reliability under the declared biased/drifting sensor
models, with paired finite-thrust execution and independent-seed evidence. It does not
establish general robustness, flight readiness, formal safety, or a universal estimator.

Next: test estimator fault discrimination under abrupt sensor bias jumps versus real
acceleration/impulse faults, then connect confidence-aware estimated state to cruise
replanning. Keep a separate trajectory/resource study for cases 43/44, whose failure
mechanism remains fuel-limited.
