# Closed-loop fuel-aware transfer research sprint

Completed: 2026-10-23

Starting HEAD: `39d3766f153bce12932fb3291db0746b5f604703`.
Environment: `E:/conda3/envs/spacecraft/python.exe`, SciPy 1.16.3, NumPy 2.3.5.
No commit or push was authorized or performed.

## Implemented capability

The new isolated development module accepts current Cartesian position, velocity, mass,
elapsed time, reserve and deadline. It forecasts crossing/capture cost, searches bounded
corrective burns, executes them with finite thrust and variable mass, and records
observation-driven reasons and failed searches. It never restores initial mass or resets
the mission clock. True state and bounded-biased navigation measurements are separate.

The departure/capture integrator enforces an 80 kg operational fuel reserve, including
partial burns, under all tested thrust scales. The same reserve applies to both members
of every experimental pair.

## Iteration trail and actual experiments

| Iteration | Paired scenarios | Baseline capture | Feedback capture | Corrective-burn traces |
|---|---:|---:|---:|---:|
| 1: initial uncertainty, cruise-only intervention | 24 | 19 | 19 | 0 |
| 2: observed departure planning, delivered-velocity error | 24 | 20 | 19 | 0 |
| 3: explicit contingency and smooth deadline constraint | 24 | 20 | 22 | 0 |
| 4: fresh seeds, matched 110 kg planning policy | 16 | 16 | 16 | 3 |
| 5: fixed-departure, 120-day cruise disturbance | 24 | 22 | 22 | 4 |

Total: 224 executed mission simulations in 112 pairs, plus independent numerical unit
tests. Every mission includes a 100-day follow-through assessment. These are five
development iterations with different disturbance/deadline contracts, not 112 independent
validation cases from a single frozen policy.

Iteration 1 reproduced the prior 710.50-day/80.50-kg nominal result. Four of twelve
80-kg-policy cases failed capture under their hard reserve; the 110-kg policy failed only
the large-navigation-bias case. Waiting until cruise could not undo excessive departure
fuel consumption. This was a genuine negative result, retained in full.

Iteration 2 added independent post-departure delivered-velocity errors and observed-state
departure replanning. The 80-kg planning arm worsened from 9/12 to 8/12 capture; the 110-kg
arm remained 11/12 for both controllers. An implicit discontinuity in the optimizer's
deadline handling prevented useful searches from late initial guesses.

Iteration 3 moved the deadline into an explicit smooth inequality and gave feedback a
30 kg contingency above the hard reserve. In the 80-kg comparison capture increased from
9/12 to 11/12, while the matched 110-kg comparison remained 11/12 in both arms. The former
is a combined resource-allocation/controller change and cannot establish feedback-only
superiority. The largest paired time penalty was 72.44 days. Navigation uncertainty
remained the unresolved failure mechanism.

## Fresh-seed validation

Seeds 21-36 were fixed before the fourth invocation. All sixteen cases used independent
uniform per-axis initial and delivered-velocity errors within +/-150 m/s, navigation
velocity bias within +/-0.5 m/s, position bias within +/-500 km, actual thrust at 80% of
nominal, 60-day cruise observation spacing, 110 kg planning reserve, and 80 kg hard reserve.

Both controllers achieved 16/16 geometric crossings, near-circular captures, and sampled
100-day survival; neither violated reserve. Worst terminal eccentricity was approximately
0.000192715 for the baseline and 0.000192813 for feedback. There is no capture-rate gain.

| Seed | Baseline completion days | Feedback completion days | Corrective delta-v m/s | Feedback fuel kg |
|---|---:|---:|---:|---:|
| 21 | 809.34 | 792.65 | 40.02 | 113.06 |
| 23 | 804.37 | 789.73 | 26.10 | 118.24 |
| 28 | 816.08 | 779.80 | 54.69 | 111.45 |

These three burns used observed post-departure states with already depleted masses
between approximately 698 and 706 kg. They were triggered by forecast delay and velocity
residual, not a periodic correction command. They were all early corrections immediately
after departure, not demonstrations of late-cruise recovery.

Across the sixteen fresh pairs, the largest time saving was 36.28 days and largest time
penalty 28.61 days. Departure replanning also changes arrival and fuel, so the paired time
changes cannot be attributed solely to the additional corrective burn. A future ablation
must hold departure planning fixed to isolate correction benefit.

## Fixed-departure cruise ablation

Iteration 5 held the departure target identical in both arms and moved the delivered-
velocity disturbance to 120 days into cruise. It reused seeds 21-36 for +/-150 m/s errors
and added seeds 41-48 at +/-250 m/s with 60% thrust and 30-day cruise observations.
Every pair retained the 110 kg nominal planning / 80 kg hard reserve policy. This is a
development ablation, not an independent validation of the preceding seed set.

Both arms captured and passed sampled survival in 22/24 cases. Four feedback runs
executed nonzero corrections after day 120:

| Seed | Time saved days | Correction delta-v m/s | Additional fuel used kg |
|---|---:|---:|---:|
| 21 | 8.00 | 25.17 | 4.69 |
| 23 | 3.74 | 11.88 | 2.26 |
| 28 | 13.70 | 42.77 | 8.00 |
| 47 | 31.93 | 97.23 | 18.33 |

The other twenty pairs produced identical terminal physical outcomes. No reserve
violation occurred. Cases 43 and 44 failed in both arms; case 44's terminal eccentricity
was 0.009106. A valid finite corrective candidate was unavailable under the declared
resource/time constraints, and the fuel floor prevented spending the reserve. This is
not proof of infeasibility for a richer trajectory family.

Unlike the resource-policy comparison, this ablation isolates intermediate-state
feedback and establishes its modeled arrival-time benefit with an explicit fuel cost.
It still provides no capture-rate improvement.

## Failure interpretation

The largest retained eccentricity was 0.07721 in the second/third iteration baseline
`extreme_9` case. The fuel floor stopped the burn rather than allowing hidden reserve
spending. A +/-5 m/s navigation-bias scenario failed even with ample residual fuel,
ending near eccentricity 0.00163. This supports adding an estimator or confidence-aware
terminal guidance, not merely increasing correction thrust.

The outward radial transfer family cannot represent inbound/post-target trajectories or
turning-point solutions. Failed searches mean no candidate in this family, not physical
infeasibility. Coast fallback is a named development behavior, not demonstrated safe mode.

## Reproduction

```powershell
E:/conda3/envs/spacecraft/python.exe -m scripts.closed_loop_transfer_dev_v1 --experiment --revision 5
E:/conda3/envs/spacecraft/python.exe -m scripts.report_closed_loop_transfer_dev_v1
E:/conda3/envs/spacecraft/python.exe -m unittest -q Tests.test_closed_loop_transfer_dev_v1
```

The experiment refuses to overwrite existing results. Use the recorded implementation
snapshot for historical revisions; the live module continues to evolve. Reporting is
offline and reads every retained iteration.

## Protected evidence and claims

All five before/after snapshots match over 773 tracked files under `analysis/`,
`runtime_assurance/`, and `simulator/`:
`ad532df0291e04b73a96bdf3bce6ff1354ddd55dedf5ee1d1453825244319c0c`.
Git shows no tracked modifications. Unrelated untracked files were left untouched.

This implements a fuel-aware feedback bridge with actual finite-thrust corrective
decisions and measured tradeoffs. It does not establish flight robustness, general
recoverability, global optimality, frozen Phase35 improvement, or Final Veto protection
for the continuous model.

Next technical priorities are estimator-aware terminal capture and a trajectory family that handles
inbound/post-target states. The current negative and positive results are sufficient to
choose those changes without another general audit phase.
