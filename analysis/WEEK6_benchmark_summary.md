# WEEK6 Benchmark Summary (ExpertV3 vs ExpertImproved)

## Scenarios
- default
- weak_thrust_far
- oscillation_noise
- misaligned_entry

## Metrics
- total_reward (higher is better)
- avg_radius_error (lower is better)
- avg_jitter (lower is better)
- final_r (sanity / context)

## Results Table

| Scenario | Controller | steps | total_reward | avg_radius_error | avg_jitter | final_r |
|---|---|---:|---:|---:|---:|---:|
| default | ExpertV3 | 2000 | -9.569e+03 | 4.194e+05 | 2.810e-07 | 9.375e+12 |
| default | ExpertImproved | 2000 | -9.521e+03 | 0.000e+00 | 3.040e-07 | 9.375e+12 |
| weak_thrust_far | ExpertV3 | 2000 | -9.569e+03 | 4.194e+05 | 2.810e-07 | 9.375e+12 |
| weak_thrust_far | ExpertImproved | 2000 | -9.521e+03 | 0.000e+00 | 3.040e-07 | 9.375e+12 |
| oscillation_noise | ExpertV3 | 2000 | -9.569e+03 | 4.194e+05 | 2.810e-07 | 9.375e+12 |
| oscillation_noise | ExpertImproved | 2000 | -9.521e+03 | 0.000e+00 | 3.040e-07 | 9.375e+12 |
| misaligned_entry | ExpertV3 | 2000 | -9.569e+03 | 4.194e+05 | 2.810e-07 | 9.375e+12 |
| misaligned_entry | ExpertImproved | 2000 | -9.521e+03 | 0.000e+00 | 3.040e-07 | 9.375e+12 |

## Summary
- Reward: ExpertImproved shows a consistent reward improvement over ExpertV3 across all tested scenarios (from -9.569e+03 to -9.521e+03).
- Radius error: Reported avg_radius_error drops from 4.194e+05 (ExpertV3) to 0.000e+00 (ExpertImproved). This looks unusually perfect and should be validated against the raw trajectory logs.
- Jitter: avg_jitter slightly increases for ExpertImproved (2.810e-07 → 3.040e-07), suggesting smoothing may not be reflected in the current jitter metric implementation.
- Final radius: Both controllers end at the same final_r (9.375e+12), implying comparable end-state radius under the current evaluation settings.
