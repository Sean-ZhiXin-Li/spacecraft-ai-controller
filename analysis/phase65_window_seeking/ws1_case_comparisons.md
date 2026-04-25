# WS-1 Case Comparisons

The table below compares baseline and WS-1 on four representative regimes using the same local 2D Euler rollout evaluator as the main WS-1 sweep.

| case | controller | final_radius_error | min_abs_radius_error | final_v_r | final_energy | steps |
|---|---|---:|---:|---:|---:|---:|
| validated_success (validated baseline success) | baseline | 3.907e+03 | 2.629e+03 | 1.218e-06 | -8.850e+06 | 48470 |
| validated_success (validated baseline success) | ws1 | 4.545e+03 | 6.935e+00 | 1.416e-06 | -8.850e+06 | 76967 |
| near_miss (Phase 4 near-miss) | baseline | 2.944e+08 | 3.892e+07 | 1.752e+02 | -1.681e+07 | 100000 |
| near_miss (Phase 4 near-miss) | ws1 | 1.151e+03 | 5.464e+02 | 3.587e-07 | -8.850e+06 | 30913 |
| energy_limited (Phase 5 energy-limited failure) | baseline | 2.204e+09 | 5.679e+07 | 8.752e+02 | -1.641e+07 | 100000 |
| energy_limited (Phase 5 energy-limited failure) | ws1 | 5.481e+02 | 3.980e+02 | -1.658e-07 | -8.677e+06 | 31544 |
| geometry_miss (geometry-miss style failure) | baseline | 1.123e+09 | 7.575e+08 | 1.220e+02 | -1.688e+07 | 100000 |
| geometry_miss (geometry-miss style failure) | ws1 | 1.123e+09 | 7.575e+08 | 1.220e+02 | -1.688e+07 | 100000 |

WS-1 trace comparison is kept tabular here to stay focused on mechanism-level differences without producing another large bundle of per-case figure files.