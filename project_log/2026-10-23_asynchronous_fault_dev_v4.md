# Asynchronous faults and contaminated-interval quarantine

Completed: 2026-10-23

## Decision and hypothesis

The README's emphasis on trustworthy interpretation of degraded information makes
estimator evidence validity a more immediate bottleneck than adding another controller.
v3's holdout synchronized physical faults with observations. Its coast fit assumes an
unpowered, undisturbed interval, which an intervening impulse violates. The selected
objective was to recover capture under asynchronous impulses by excluding a suspect
position interval from estimator history, with identical actions, sensor models,
resources, deadlines, and evaluation predicates available to both arms.

The implementation was fixed before the development comparison; no estimator parameters
were tuned after viewing results. The sparse-observation set was added after holdout as
a separately declared robustness experiment. No previous output was overwritten.

## Mechanism

For a short coast with impulse delta-v at fraction f of the interval, the displacement
includes approximately (1-f)*delta-v*dt. A constant-velocity-bias fit can absorb that
displacement as an apparent bias of approximately -(1-f)*delta-v. Subsequent correct
fault classification does not remove this contaminated historical fit in v3's physical
fault branch. This approximation explains the mechanism; executed results use nonlinear
two-body integration and exact plant event splitting.

QuarantinedEstimator saves the valid bias history before each update. When the inherited
innovation detector first reports ambiguity, it discards the interval fit and restores
the extrapolated prior bias. It retains the new raw endpoint to fit the following coast.
No fault time, class, vector, true state, or evaluation output enters the estimator.
The inherited engineering innovation gate and capture predicates are unchanged.

The isolated v4 runtime reuses the v3 runtime structure and imports its estimator and
v1/v2 helpers. AsynchronousCoast splits only plant integration at the exact injected
epoch, preserves absolute event times, and applies no estimator-side correction. Both
arms use this runtime. An epoch skipped by a powered arc causes an explicit error.
Faults during powered arcs are not supported by this experiment.

## Executed experiments

144 comparison mission simulations in 72 pairs were completed, plus unit integrations.
Each seed supplies three related fault conditions; do not treat these as independent seeds.

| Set | Seeds | Pairs | Actually exposed pairs | v3 capture | Quarantine capture |
|---|---|---:|---:|---:|---:|
| Development | 801-804 | 12 | 12 | 5/12 | 12/12 |
| Holdout | 901-912 | 36 | 30 | 20/36 | 36/36 |
| Sparse observations | 1001-1008 | 24 | 24 | 12/24 | 24/24 |

Holdout seeds 903 and 908 completed before the declared fault, accounting for six
unexposed pairs. Among the 30 exposed holdout pairs, capture was 14/30 versus 30/30:
sensor-only 10/10 versus 10/10; physical 0/10 versus 10/10; combined 4/10 versus 10/10.
All successful quarantine captures passed the sampled 100-day survival check.
There were no capture regressions in any set. Rates are not pooled across sets.

Both arms share 150 m/s fault amplitude and direction per seed, initial and day-120
physical velocity perturbations bounded by 100 m/s per axis, velocity sensor bias up to
15 m/s, drift 0.2 m/s/day, position bias up to 500 km, bounded time-varying position
errors up to 200 km, 50% thrust, an 80 kg hard reserve, and a 1100-day deadline.
Planning reserve is 110 kg. Capture requires guidance convergence, eccentricity <0.001
and radius error ratio <=0.0025. Survival uses 101 samples over 100 days.

Development and holdout fault times are seeded uniform values in [722,748] days with
30-day observations. Sparse tests use [602,658] days and 60-day observations.
Sensor faults change measurements at their actual epoch. Physical faults split the
continuous plant coast at that epoch; they are not rounded to observation times.

## Outcomes and causal evidence

In the exposed holdout, both methods classified the 30 late faults consistently with
their injected classes. Nevertheless, reference terminal velocity-estimation error
reached 93.52 m/s; quarantine reduced its maximum to 0.533 m/s. Thus improvement came
from protecting estimator history, not more correct classification. Neither arm used
cruise corrections in these experiments. Finite-thrust capture consumed the corrected
estimate; the result does not demonstrate an improvement in cruise replanning.

Holdout quarantine added at most 8.449 kg propellant consumption relative to reference,
and saved up to 3.948 kg in other pairs. Minimum remaining propellant was 88.139 kg.
Maximum completion delay was 0.000432 days. Onset-to-confirmation was 10.77-54.71 days.

With sparse observations, exposed capture improved from 12/24 to 24/24; maximum terminal
velocity-estimation error fell from 66.51 to 0.163 m/s. Quarantine confirmation required
63.62-116.50 days. Reference produced three extra confirmed late records, illustrating
how contamination can produce repeated apparent changes. Sparse quarantine minimum
propellant was 81.537 kg, close to the 80 kg floor. Reference reached that floor in some
failures. Reserve checks retained their existing numerical tolerance; no violations
occurred. No burns during ambiguity occurred in either arm.

## Limitations and negative evidence

All reference failures and raw records are retained. The original v1 fuel-limited cases
43/44 remain unsolved. The new method needs a clean subsequent coast interval, adequate
position measurements, and a detectable innovation. Sustained unknown acceleration,
repeated impulses within consecutive intervals, position sensor faults, small undetected
impulses, and nonlinear bias drift remain unvalidated. Combined diagnoses do not exclude
model mismatch. No universal diagnosis, formal safety, or flight-readiness claim follows.

The slow confirmation latency remains substantial. Earlier source experiments had
synchronous faults; their success counts cannot be used as this asynchronous baseline.
The reference here is the unchanged v3 estimator inside the same new plant runtime.
The detector has a development engineering gate, not a calibrated false-positive bound.

Next research should test persistence of unknown acceleration and repeated disturbances.
If no clean interval exists, quarantine alone cannot establish a trustworthy estimate;
that calls for multiple motion hypotheses or an explicit unknown-input estimator.

## Reproduction and validation

Run from E:/spacecraft_ai_project with E:/conda3/envs/spacecraft/python.exe:

```powershell
E:/conda3/envs/spacecraft/python.exe -m scripts.asynchronous_fault_dev_v4 --experiment development
E:/conda3/envs/spacecraft/python.exe -m scripts.asynchronous_fault_dev_v4 --experiment holdout
E:/conda3/envs/spacecraft/python.exe -m scripts.asynchronous_fault_dev_v4 --experiment sparse
E:/conda3/envs/spacecraft/python.exe -m scripts.report_asynchronous_fault_dev_v4
E:/conda3/envs/spacecraft/python.exe -m unittest -q Tests.test_asynchronous_fault_dev_v4 Tests.test_fault_aware_transfer_dev_v3 Tests.test_estimated_capture_dev_v2 Tests.test_closed_loop_transfer_dev_v1
```

Generation refuses existing directories. Each output includes configuration, executed
implementation snapshot, raw runs, summary and hashes; reports separate exposure status.
Historical snapshots retain the exact development/holdout code before the sparse CLI
mode was added. Estimator and propagation code did not change between experiments.

Tests exercise exact impulse splitting, event-time continuity, terminal-before-fault
behavior, skipped-epoch rejection, sensor-only plant neutrality, quarantine history,
owned measurements, no-fault equivalence with v3, and artifact hashes/pairing.

## Preservation

Tracked HEAD remains 39d3766f153bce12932fb3291db0746b5f604703. No commit or push.
All experiment before/after snapshots agree:

- Frozen evidence, 773 files: ad532df0291e04b73a96bdf3bce6ff1354ddd55dedf5ee1d1453825244319c0c.
- v1 historical aggregate: 8d0f2ba1bf607f25e21f1b93bf79cd8d1ecd2cb84f39d271f3ae52931cd1ee69.
- v2 historical aggregate: ad22633d67d7f4c4957fad6e84ff7dd8d403316b19a93f8fe03f43163b948c38.
- v1-v3 development files/results, 103 files: 80a0460b0ee5d47e9f38dd313e6ded1205cdf4931614ee9e69a5e8dc2d00e773.

Frozen runtime contracts and unrelated workspace materials were untouched.
