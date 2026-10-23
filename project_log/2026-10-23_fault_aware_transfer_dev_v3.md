# Position-confirmed fault isolation and capture recovery

Completed: 2026-10-23

Starting HEAD: `39d3766f153bce12932fb3291db0746b5f604703`.
Environment: `E:/conda3/envs/spacecraft/python.exe`, SciPy 1.16.3, NumPy 2.3.5.

## Question and objective fixed before execution

The v2 estimator corrected steady and drifting velocity-sensor bias but assumed that its
recent bias history remained valid. An abrupt change could therefore corrupt estimated
state and trigger an inappropriate correction or failed capture.

The objective was to distinguish a late velocity-sensor jump from a real velocity impulse
using subsequent position evidence, and improve capture without granting the estimator
hidden truth, additional propellant, or altered terminal thresholds. At the instant of a
jump, these mechanisms can generate the same measured state. Immediate unique diagnosis
is impossible under the specified measurements. The implementation must retain ambiguity
until later position information becomes available.

## Implemented capability

`scripts/fault_aware_transfer_dev_v3.py` extends the v2 coast-shooting estimator with:

1. A predicted measurement innovation compared against a displacement-derived gate plus
   a declared 1 m/s engineering model allowance.
2. An `ambiguous_velocity_jump` state. Cruise correction is held during that state.
3. Position-confirmed diagnosis on the next unpowered measurement interval. A changed
   fitted sensor bias identifies a sensor jump; unchanged fitted bias with changed actual
   motion identifies a physical state change. Both signatures produce the deliberately
   nonexclusive label `combined_or_model_mismatch`.
4. Reset of obsolete bias-level history while retaining the pre-fault drift-rate estimate.
   Short noisy intervals are duration-weighted while the bias history rebuilds.
5. Reuse of the v1 bounded transfer-replanning search and finite-thrust burn execution when
   arrival prediction requires correction after ambiguity clears.
6. A capture hold if target acquisition occurs while diagnosis is pending. A seven-day
   coast baseline is then allowed before re-evaluation; the same hard 1100-day mission
   deadline remains in effect.

No injected fault label or true state enters the estimator. Unit tests construct sensor
and physical faults with indistinguishable instantaneous measurements and show that later
position observations separate them. The confirmation delay is a real information cost.

## Matched experimental design

Both arms use the same raw-sensor departure, initial disturbances, day-120 physical
disturbance, late-fault vector, position noise, thrust degradation, 110 kg nominal planning
reserve, 80 kg hard operational reserve, and 1100-day deadline. The reference is the v2
trend estimator inside the same new runtime, including the same available v1 replanning
routine. A no-fault numerical test compares its terminal state against v2 execution.

Capture retains eccentricity `<0.001`, radius error ratio `<=0.0025`, and guidance
convergence. Survival is assessed at 101 points across 100 unpowered days. Crossing,
capture, fuel compliance, and sampled survival remain distinct measurements.

The original discovery/validation fault amplitudes were 80 m/s at day 660. Stress and
holdout used 150 m/s late faults at day 720, initial/delivered velocity bounds +/-100 m/s,
initial sensor bias bounds +/-15 m/s, linear bias drift 0.2 m/s/day, position bias up to
500 km plus bounded time-varying errors up to 200 km, 50% actual thrust, and 30-day cruise
observations. The late fault is realized at the first cruise observation at or after the
declared epoch; this synchronous fault model is an important limitation.

## Actual experiments and negative evidence

| Set | Paired conditions | Reference capture | Fault-aware capture |
|---|---:|---:|---:|
| Discovery, seeds 401-404 | 8 | 4/8 | 8/8 |
| Unseen validation, seeds 501-512 | 24 | 12/24 | 24/24 |
| Initial harder stress, seeds 601-608 | 24 | 10/24 | 14/24 |
| Refined stress, same retained cases | 24 | 10/24 | 24/24 |
| Unseen late-fault holdout, seeds 701-712 | 36 | 12/36 | 36/36 |

These are 232 comparison mission simulations in 116 pairs, plus unit integrations and
baseline reproduction. Some seeds are deliberately reused for refinement, and multiple
fault classes share each seed. Do not pool these counts into a validation success rate.
The final holdout has twelve independent seeds and three fault conditions per seed.

The initial stress result included two capture regressions. Diagnosis was correct in
24/24 cases, yet capture failed in ten fault-aware cases. Inspection showed that resetting
the bias history also discarded its useful drift rate, leaving a stale estimate during
the last short coast. The refined implementation retains drift while rebuilding the bias
level. It eliminated the ten failures on the retained cases; the earlier failed outputs
and their exact implementation snapshot are preserved.

## Final holdout result

The final holdout rescued 24 failed conditions and caused no capture regressions:

| Fault class | Reference | Fault-aware |
|---|---:|---:|
| Sensor jump | 0/12 | 12/12 |
| Physical impulse | 12/12 | 12/12 |
| Combined sensor and physical fault | 0/12 | 12/12 |

All 36 fault-aware captures passed sampled survival. Worst terminal eccentricity fell
from 0.0437023 to 0.00012514. Late-fault classifications were consistent with all 36
injected cases, with confirmation delays of 19.73-30 days. The combined label is compatible
with both mechanisms but does not exclude other unmodeled causes.

Neither arm violated the operational reserve. Minimum final propellant across both arms
was 87.134 kg. Fault-aware capture used at most 4.952 kg additional propellant relative
to the reference; some pairs saved up to 2.484 kg. The maximum capture-epoch difference
was about 21.85 seconds. These are improvements in actual finite-thrust outcomes, not
only in a prediction residual.

No holdout cruise correction was needed in either arm. Consequently, the holdout capture
gain establishes the estimator-recovery mechanism rather than a replanning advantage.
The ambiguity interlock was exercised observationally but had no competing holdout burn
to suppress. The harder development stress set did exercise physical cruise interventions:
in seed 607, the reference spent 49.02 m/s on a sensor-only fault, while the aware runtime
spent zero. For the corresponding physical/combined faults, the aware runtime executed
approximately 82.2 m/s corrections after confirmation. This single development example
supports feasibility of action gating, not a general false-intervention-rate claim.

## Scientific finding

Correct fault classification is insufficient if estimator recovery discards useful state.
A resilient navigation/control boundary needs both delayed evidence-based diagnosis and
a recovery rule that preserves valid pre-fault model information. Sensor jumps and real
impulses are distinguishable only after accumulating additional position information in
this measurement model; the initial ambiguity should constrain action authority.

## Limits and unresolved questions

- Faults are synchronized with observations. Between-observation impulses can contaminate
  the coast-shooting fit and remain untested.
- Plant and estimator share two-body coast dynamics. Sustained unknown forces can mimic
  sensor bias; the combined diagnosis is intentionally nonexclusive.
- Initial jump magnitude/direction and linear drift are controlled development models.
  Arbitrary position-sensor faults, outages, nonlinear drift, and repeated simultaneous
  faults are not validated.
- The diagnosis gate is an engineering convention, not a scientifically calibrated
  false-positive bound or formal safety guarantee.
- The fuel-limited v1 cases 43/44 have not been solved by this work; they require a
  separate trajectory/resource intervention.
- The seven-day terminal confirmation hold can trade geometry and fuel against certainty.
  No longer deadline is granted, but that tradeoff needs testing near a tight deadline.
- All results remain simplified 2D development evidence; no frozen Phase34/35 or Stage2
  authority or formal Final Veto protection is inherited.

## Reproduction and files

```powershell
E:/conda3/envs/spacecraft/python.exe -m scripts.fault_aware_transfer_dev_v3 --experiment discovery
E:/conda3/envs/spacecraft/python.exe -m scripts.fault_aware_transfer_dev_v3 --experiment validation
E:/conda3/envs/spacecraft/python.exe -m scripts.fault_aware_transfer_dev_v3 --experiment stress
E:/conda3/envs/spacecraft/python.exe -m scripts.fault_aware_transfer_dev_v3 --experiment stress_refined
E:/conda3/envs/spacecraft/python.exe -m scripts.fault_aware_transfer_dev_v3 --experiment holdout
E:/conda3/envs/spacecraft/python.exe -m scripts.report_fault_aware_transfer_dev_v3
E:/conda3/envs/spacecraft/python.exe -m unittest -q Tests.test_fault_aware_transfer_dev_v3 Tests.test_estimated_capture_dev_v2 Tests.test_closed_loop_transfer_dev_v1
```

Each numerical command refuses to overwrite its result directory. Reproducing historical
implementations uses the recorded `implementation_snapshot.py`. Current source also
contains an explicit terminal ambiguity interlock added before holdout execution.

Results: `development_results/fault_aware_v3_2026-10-23_<set>/`, containing pre-execution
configuration, executed source, raw records, summary, comparison, hashes, and a holdout
figure. Tests cover measurement indistinguishability, delayed diagnosis, retained drift,
information ownership, no-fault baseline fidelity, paired source states, reserve floors,
and source/artifact hashes.

## Preservation

All before/after snapshots agree:

- Frozen tracked evidence: 773 files,
  `ad532df0291e04b73a96bdf3bce6ff1354ddd55dedf5ee1d1453825244319c0c`.
- Historical v1 development: 36 files,
  `8d0f2ba1bf607f25e21f1b93bf79cd8d1ecd2cb84f39d271f3ae52931cd1ee69`.
- Historical v2 development: 34 files,
  `ad22633d67d7f4c4957fad6e84ff7dd8d403316b19a93f8fe03f43163b948c38`.

Unrelated materials were untouched. No commit, push, or frozen authority change occurred.

Next research: asynchronous fault onset and unknown sustained acceleration, including
cases where several hypotheses remain observationally plausible. That directly tests
the current isolation mechanism's strongest remaining assumption.
