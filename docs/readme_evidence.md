# README evidence and reproduction guide

This guide supports the November 2026 README refresh. Engineering evidence is pinned to [`ff4d06229016511d38bba2c432b990e0c38be001`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/ff4d06229016511d38bba2c432b990e0c38be001), the October 23, 2026 project snapshot. It introduces no new experiment results and changes no research implementation or frozen artifact.

## Evidence index

Paths below are relative to the repository root. Each named experiment directory contains `configuration.json`, `implementation_snapshot.py`, `runs.jsonl`, `summary.json` and `artifact_hashes.json`; selected reports also have `comparison.json` and `report_hashes.json`. A configuration's earlier `source_head` records its working baseline before later commits, not necessarily the commit that published the experiment.

| Study and publishing commit | Primary evidence directory | Interpretation boundary |
|---|---|---|
| Fuel-aware v1 · [`44676cc`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/44676cc) | [revision 5](../development_results/closed_loop_transfer_v1_2026-10-23_revision5/) | 24 paired conditions, 48 simulations; reference and feedback both capture/survive 22/24. Feedback executes four correction traces but does not improve capture count. Cases 43/44 remain failures. |
| Position-aided v2 · [`f65e865`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/f65e865) | [drift holdout](../development_results/estimated_capture_v2_2026-10-23_holdout/) | Twelve seeds 301–312 × drift rates 0.2/0.4 m/s/day; static 8/24 vs trend 24/24 captures and sampled survival. |
| Fault-aware v3 · [`9308023`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/9308023) | [late-fault holdout](../development_results/fault_aware_v3_2026-10-23_holdout/) | Twelve seeds 701–712 × sensor/physical/combined conditions; reference 12/36 vs aware 36/36. Faults synchronized with observation updates; no holdout cruise corrections. |
| Quarantine v4 · [`46cac6c`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/46cac6c) | [asynchronous holdout](../development_results/asynchronous_fault_v4_2026-10-23_holdout/) | Twelve seeds 901–912 × three conditions; 20/36 vs 36/36 overall. Seeds 903/908 complete before fault onset: exposed counts are 14/30 vs 30/30. |
| Bidirectional · [`ff4d062`](https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller/commit/ff4d062) | [multiregime holdout](../development_results/multiregime_transfer_2026-10-23_holdout/) | Seeds 1201–1224, one task per seed, three methods per task: 72 records. Legacy 1/24, matched outward 11/24, bidirectional 24/24. The latter two share the optimizer/search contract and are the primary comparison. |

These are different distributions and interventions. Do not pool their runs into one success rate. Conditions sharing a seed are related, not independent replications. No confidence interval, global superiority or arbitrary-fault robustness claim follows from these small sets.

### Negative evidence and controls

- The [initial multiregime development](../development_results/multiregime_transfer_2026-10-23_development/) gave all methods the inherited 1,100-day deadline. All returned `no_plan` on eight tasks; no mission was executed.
- The [extended development](../development_results/multiregime_transfer_2026-10-23_extended_development/) gave every method 4,400 days. Legacy/matched-outward/bidirectional captures were 1/8, 3/8 and 7/8. This is a new mission horizon, not improved performance under the original deadline.
- Development seed 1104 remains `no_plan` at 4,400 days. A [recorded diagnostic](../development_results/multiregime_transfer_2026-10-23_holdout/comparison.json) found a candidate at about 4,537.75 days under a separate 10,000-day search limit; it was not executed or included in success counts. Finite multi-start failure is not a proof of infeasibility.
- The legacy planner retains a 2,000-day internal prediction cutoff and different search details. Its 1/24 result is historical context, not a clean geometry-only baseline. On the eleven common captures of the matched and bidirectional planners, completion-time differences are below `4.88e-7` days and propellant differences below `2.61e-8` kg. The supported gain is coverage, with no demonstrated speed advantage.
- [V1 revision 3](../development_results/closed_loop_transfer_v1_2026-10-23_revision3/) records 9/12 → 11/12 at nominal 80 kg planning reserve, but feedback also replans departure with a larger effective reserve. Neither arm executes cruise corrections there. It does not isolate a midcourse-feedback benefit. Revision 5 is the stronger fixed-departure comparison.
- V3's [initial stress](../development_results/fault_aware_v3_2026-10-23_stress/) had 14/24 aware captures before refinement on the same cases reached 24/24. Those reused cases are development evidence; the separate final holdout is the appropriate validation set.
- In v4's exposed holdout, sensor-only capture is 10/10 in both arms, physical-impulse capture 0/10 → 10/10, and combined-fault capture 4/10 → 10/10. Both methods classify the exposed faults consistently with injected classes. Neither executes cruise corrections; quarantine protects estimator history rather than improving holdout replanning. The [sparse-observation set](../development_results/asynchronous_fault_v4_2026-10-23_sparse/) is a separate later robustness experiment, not pooled holdout evidence.

## Model and evaluation boundaries

The historical `envs/orbit_env.py` / Phase34 line and the October development line are distinct models and protocols. The latter uses [`dev_fuel_aware_transfer_optimizer_v0.py`](../scripts/dev_fuel_aware_transfer_optimizer_v0.py) with solar two-body gravity, target radius `7.5e12 m`, wet/dry mass 722/300 kg, nominal thrust 2,000 N, specific impulse 450 s and 100 s receding guidance. Search uses ideal impulses; execution integrates finite burns and mass loss with DOP853. An independent numerical execution of an impulsive plan is still simplified simulation, not physical hardware.

[`multiregime_transfer_research.py`](../scripts/multiregime_transfer_research.py) evaluates a bounded prograde elliptic transfer family. The paired holdout has initial radius ratios 0.94–0.99 or 1.01–1.06, radial velocities from −350 to 350 m/s, tangential speeds 0.85–1.08 times local circular speed, random planar orientation and seeded actual thrust 50–100% of nominal. Navigation velocity/position biases are bounded by 0.2 m/s and 100 km per component; mass is known. Planning reserve is 110 kg and operational reserve 80 kg. These relatively small navigation errors do not establish a fault-adaptation benefit for this experiment.

The latest capture predicate is guidance convergence, eccentricity `<0.001`, relative radius error `≤0.0025` and completion no later than the 4,400-day mission deadline. Successful bidirectional completion spans about 765.85–4,377.00 days. The minimum final propellant is about 110.498 kg; no recorded reserve violation occurs. Survival requires all 101 samples across the following 100 unpowered days to remain within the radius-error threshold. Sampled survival is not continuous invariance or a safety certificate.

**V2–v4 deadline caveat:** their coast logic uses a 1,100-day horizon, but their final capture predicates do not explicitly reject a terminal burn finishing after it. This is an implementation boundary, not an enforced hard deadline guarantee. In the recorded successful holdouts, the maximum completion times across both arms are 804.291 days (v2), 789.436 days (v3) and 792.680 days (v4); the missing check does not change those particular reported counts. No algorithm was modified for this documentation refresh.

The estimator consumes measurements, timestamps and a declared model. Injected faults and true state are used by the simulated plant and scoring. Shared model assumptions, exact mass, development detection gates, clean-coast requirements and long confirmation latencies remain limitations. `physical_execution` in a result record means a numerically integrated simulated mission; it does not mean a hardware trial.

## Historical evidence

- [Phase34 matched comparison](../analysis/phase34_post_cross_sync/phase34_vs_phase31_comparison.md): 24 cases per arm; eight crossings and eight simulator success labels in each. Recoverable crossings change 0 → 8. This is not a success-count gain or evidence of immediate first-crossing recoverability.
- [Final Veto](../analysis/final_veto_ablation_v0/summary.md): July 16 frozen milestone, eight preservation and five diagnostic stress cases. Overspeed changes 5/5 → 0/5 while task recovery stays 0/5. Step-limit endings are retained.
- [Stage 2A eligibility](../analysis/stage2a_q1_e_eligibility_audit_v0/summary.md): no eligible boundary; no active Stage 2A-E experiment. Offline alternatives and proposal classification do not confer active control authority.
- [Stage 2B](../analysis/stage2b_numerical_validation_v0/summary.md): Euler/DOP853 overspeed classifications agree for fourteen one-step proposals. This is a local numerical cross-check, not global integration independence or formal Runtime Assurance.

## Reproduction boundary

The focused numerical test command in the root README executes existing analytical/numerical checks, finite-burn examples and recorded artifact/pairing checks. It is not a rerun of every research campaign. The smoke tests exercise the environment and a short `quickrun`; the phase guard checks stored historical aggregates. Passing these does not establish hardware, full repository readiness or universal robustness.

Experiment entry points are `scripts.closed_loop_transfer_dev_v1`, `scripts.estimated_capture_dev_v2`, `scripts.fault_aware_transfer_dev_v3`, `scripts.asynchronous_fault_dev_v4` and `scripts.multiregime_transfer_research`. Their `--experiment` modes write under `development_results/` and refuse existing output directories. Historical snapshots can differ from current CLI scaffolding; use the saved implementation and matching configuration for a historical rerun in an expendable checkout. Do not delete or rewrite frozen evidence in the main research checkout.

Reporting entry points `scripts.report_*` write result/report files; some perform extra optimization. In particular, `report_multiregime_transfer_research` launches a longer-deadline development diagnostic. It must not be described as a read-only validator. The README's existing figure was reused and its hashes checked without executing that writer.

### Checkout line endings

The manifests hash original bytes, not normalized parsed data. On the audited default Windows checkout, 91 of 112 development-result manifest entries matched exactly; 21 Python snapshots differed solely by LF→CRLF conversion. Their committed LF blobs matched the original hashes. Several JSON/JSONL records instead have original CRLF hashes, so an all-LF checkout is not a universal fix.

For a fresh Windows checkout, prepare Python's LF exception before populating files:

```powershell
git -c core.autocrlf=true clone --no-checkout https://github.com/Sean-ZhiXin-Li/spacecraft-ai-controller.git
cd spacecraft-ai-controller
git config core.autocrlf true
python -c "from pathlib import Path; Path('.git/info/attributes').write_text('*.py text eol=lf\n', encoding='ascii')"
git checkout main
```

This is local checkout configuration, not a change to scientific content. Do not overwrite an existing `.git/info/attributes` in a working research checkout; the command is for a new clone. In the audit's already-populated independent clone, the 21 snapshots were restored from their committed LF blobs and identical index entries refreshed; the full 56-test numerical set then passed. Raw public scientific files and manifests have no content diff. Fresh Linux dependency resolution and byte-perfect historical hashes on arbitrary platforms remain separate verification tasks.

## Media audit

| Asset / placement | Real source and generation | Scientific support and limits |
|---|---|---|
| [Reused geometry comparison](../development_results/multiregime_transfer_2026-10-23_holdout/comparison.png), near README top | The 72 `runs.jsonl` records and `configuration.json` in that directory, source `ff4d062`. Existing renderer: [`scripts/report_multiregime_transfer_research.py`](../scripts/report_multiregime_transfer_research.py), originally invoked as `python -m scripts.report_multiregime_transfer_research`; it writes reports and runs a diagnostic. | All 24 tasks shown for each matched method, common axes: initial radius/target radius (dimensionless), radial velocity (m/s). Marker shape/color distinguishes capture plus sampled survival from no plan. Includes failures; not a measured trajectory, independent-task distribution estimate or flight result. Image and data hashes verified against supplied public manifests; previewed at README size. |
| [Historical insertion image](../analysis/demo/orbit_demo_trajectory.png) and [GIF](../analysis/demo/orbit_demo_zoom.gif), text links only | Existing [`scripts/generate_orbit_demo.py`](../scripts/generate_orbit_demo.py) and [`orbit_demo_summary.json`](../analysis/demo/orbit_demo_summary.json). | Historical explicit-controller illustration, not a Phase34 benchmark replay or latest fuel-constrained result. Uses `r0/target=1.00005`, `dt=100 s` and `thrust_scale=10000`; selected demonstration, not realistic propulsion evidence. No rerender or trajectory reconstruction performed. |

No new scientific asset was generated: the existing geometry figure already answers the README's coverage question. No private laboratory data, external imagery or generated decorative spacecraft images are included. The root README's bounded counts are the accessible textual equivalent of the holdout plot.
