# Bidirectional transfer generation and the mission-time boundary

Completed: 2026-10-23

## Research choice

After reading the full README, docs/research_direction.md, the Phase32/34 sources,
BC training code and PPO evaluation evidence, and the recent transfer/estimation
implementations, the selected bottleneck was narrow trajectory geometry. The recent
planner rejects every initial state outside the target radius and requires positive
post-injection radial velocity. Better navigation cannot remove that restriction.

This work returns to the historical upstream crossing-generation question in the
separate variable-mass solar two-body development model. It does not rerun or improve
the frozen Phase34 benchmark. Historical Phase34 success was explicit control, not RL.
The inspected PPO evaluation artifact records zero crossings and false success; it
does not establish why PPO failed. BC uses a five-input supervised network and a random
sample split, which by itself is not evidence of trajectory-level generalization.

RL was not selected: the chosen limitation is a known orbital intersection problem
with accurate available dynamics and inexpensive low-dimensional optimization.
A future learned warm start or feasibility model would need disjoint trajectory tasks
and measured latency/coverage benefits against the optimizer implemented here.

## Implemented architecture and physics

scripts/multiregime_transfer_research.py separates orbital prediction, constrained search,
measurement-based finite-burn execution, and truth-based evaluation. The new predictor
uses bound elliptic elements from radius, radial velocity, and angular momentum:

    a = -mu / (2 * specific_energy)
    e^2 = 1 - h^2 / (mu*a)
    cos(E_target) = (1 - R_target/a) / e
    M = E - e*sin(E)
    time = positive_mod(M_target-M_initial, 2*pi) / sqrt(mu/a^3)

Both target eccentric anomalies are evaluated, yielding the earliest future inward or
outward intersection, including a radial turning point if required. Capture impulse
is computed from radial and tangential velocities at the target. Rocket-equation mass
loss constrains departure plus capture. Unbound, retrograde, exactly circular and tangent
intersection cases remain unsupported, explicitly returning no prediction.

The optimizer uses seven deterministic seeds and scaled SLSQP, with shared +/-2000 m/s
radial/tangential departure bounds, mass budget, deadline, and 1.90*target circular-speed
departure constraint. That development constraint is not the frozen Final Veto monitor.
There is no global-optimality or global-infeasibility claim.

Execution independently integrates departure and capture with the existing 2000 N,
variable-mass burn implementation and bounded measurement errors. Navigation bias is
up to 0.2 m/s per velocity component and 100 km per position component; mass is known.
Actual thrust is seeded between 50% and 100%. Capture requires guidance convergence,
eccentricity <0.001, radius error ratio <=0.0025, and completion within the declared
deadline. Survival uses 101 samples over 100 days. Crossing alone is never capture.

## Comparison contract and negative first experiment

All methods receive the same initial state, measurements, thrust, wet mass 722 kg,
planning reserve 110 kg, hard operational reserve 80 kg, and terminal predicates.
Seeded tasks have radius 0.94-0.99 or 1.01-1.06 times the target, radial velocity
between -350 and 350 m/s, tangential speed 0.85-1.08 times local circular speed,
and random planar orientation. The distribution was fixed before experiments.

Three methods are recorded:

- Legacy v1 planner, unchanged. Its internal prediction horizon remains 2000 days.
- Matched outward predictor with the new optimizer/seeds/scaling and full allowed horizon.
- Bidirectional predictor with that same optimizer/seeds/scaling and horizon.

The primary comparison is matched_outward versus bidirectional. The legacy method's
additional horizon/search limitations make it an informative historical reference,
not a clean causal control for geometry alone.

Initial development used the inherited 1100-day deadline on seeds 1101-1108. All three
methods returned no plan on all eight tasks. Zero mission trajectories were executed.
These negative outputs and their implementation snapshot remain intact.

The larger radius gaps expose a time/fuel constraint absent from the narrow original
task. A separate contract then gave EVERY method 4400 days. This is explicitly a new
mission horizon, not improved performance under the old deadline. Development seeds
were reused to understand the change; final seeds 1201-1224 were unseen.

## Executed results

| Contract | Tasks | Legacy capture | Matched outward | Bidirectional |
|---|---:|---:|---:|---:|
| Initial 1100-day development | 8 | 0/8 | 0/8 | 0/8 |
| Extended 4400-day development | 8 | 1/8 | 3/8 | 7/8 |
| Unseen 4400-day holdout | 24 | 1/24 | 11/24 | 24/24 |

All executed captures passed sampled survival. There were 120 task/method planning
evaluations and 47 full mission executions (11 development, 36 holdout), plus unit-test
propagations and one finite-burn test. No-plan cases execute no physical mission.
Two additional diagnostic optimization calls explored seed 1104 with a 10000-day search
limit (one exact-state diagnostic and one recorded measurement-based diagnostic);
neither executed a mission or contributes to capture counts.

The new holdout planner captured 12/12 exterior and 12/12 interior starts. The matched
outward planner captured 11/12 interior and 0/12 exterior starts. Its one interior
no-plan case is a numerical search limitation, not evidence that outward motion there
is impossible. On the eleven common captures, completion times differ by less than
4.88e-7 days and propellant by less than 2.61e-8 kg: this isolates expanded coverage
rather than improved execution on already supported arcs.

Holdout bidirectional completion times were 765.85-4377.00 days, minimum remaining
propellant 110.498 kg, and maximum terminal eccentricity 0.0001133. There were no reserve
violations. Maximum difference between impulsive predicted coast time and completed
finite-burn arrival was 0.589 days, including burn duration. Median planning times on
this machine were 0.067 s matched outward and 0.086 s bidirectional. No speed advantage
is claimed. All task details, runtime records, executed source and hashes are retained.

## Remaining failure and interpretation

Development seed 1104 remains no-plan at 4400 days. The recorded measurement-based
diagnostic found a candidate at 4537.751 days with predicted final mass 410.5 kg,
on the planning budget boundary. This supports a mission-time/resource interpretation,
but finite multi-start optimization does not prove the absence of a better maneuver.
No longer-deadline candidate was counted as a successful mission.

The architectural result is that explicit representation of reachable inward/outward
arcs expands demonstrated capture coverage. It also exposes a separate need for mission
time allocation. Better estimation and generic policy training would not automatically
repair an excluded action/trajectory family.

This remains a two-impulse search with finite-burn replay in planar two-body gravity.
It does not solve free terminal longitude, rendezvous, multi-revolution selection,
hyperbolic transfers, low-thrust continuous arcs, three-dimensional plane changes,
uncertain dynamics, or the old fuel-limited cases 43/44. Navigation uncertainty here is
small and no fault-adaptation benefit is claimed. Longer missions have operational costs
not modeled by the current fuel/time constraints. No formal safety or deployment claim.

Next capability: mission-level time/resource negotiation across multiple transfer arcs,
with explicit feasible-set evidence. Learning becomes useful if a verified solution
library can reduce search effort or improve convergence on unseen geometry; it should
be evaluated against this physics-based solver with the same authority and information.

## Reproduction and preservation

```powershell
E:/conda3/envs/spacecraft/python.exe -m scripts.multiregime_transfer_research --experiment development
E:/conda3/envs/spacecraft/python.exe -m scripts.multiregime_transfer_research --experiment extended_development
E:/conda3/envs/spacecraft/python.exe -m scripts.multiregime_transfer_research --experiment holdout
E:/conda3/envs/spacecraft/python.exe -m scripts.report_multiregime_transfer_research
E:/conda3/envs/spacecraft/python.exe -m unittest -q Tests.test_multiregime_transfer_research
```

Generation refuses to overwrite outputs under development_results/multiregime_transfer_*.
Historical code snapshots preserve each executed contract. Current source additionally
rejects zero-radius inputs explicitly, added after experiments; this does not affect any
executed task. Tests compare Kepler times and terminal velocities against independent
continuous integration, including inward motion and turning points; they also cover
outward-quadrature agreement, rotation invariance, unsupported states, disjoint seeds,
finite-burn exterior capture, paired task identity, and artifact hashes.

All before/after historical snapshots agree, including 773 frozen evidence files and
the prior v1-v4 development outputs. Frozen aggregate:
ad532df0291e04b73a96bdf3bce6ff1354ddd55dedf5ee1d1453825244319c0c.
The full snapshot objects are recorded in every summary.json. Tracked source and all
unrelated materials remain unchanged. No commit, staging, or push occurred.
