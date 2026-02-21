import os
import pandas as pd

CSV_PATH = "analysis/results/whpl14_variant_x_thrust_x_r0.csv"
OUT_PATH = "analysis/results/whpl15_boundary_summary.csv"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(CSV_PATH)

DELTA = 3000.0
thrusts = [800.0, 2000.0]

rows = []

for thrust in thrusts:
    sub = df.loc[df["thrust_newton"] == thrust].copy()
    if sub.empty:
        continue

    # Aggregate per point
    g = (
        sub.groupby(["controller_variant", "r0_over_target"], as_index=False)
        .agg(
            total_reward=("total_reward", "mean"),
            saturation_rate_mean=("saturation_rate_mean", "mean"),
            avg_radius_error=("avg_radius_error", "mean"),
            final_radius_error=("final_radius_error", "mean"),
        )
    )

    max_reward_all = float(g["total_reward"].max())
    stable_thr = max_reward_all - DELTA

    for variant, vdf in g.groupby("controller_variant"):
        vdf = vdf.sort_values("r0_over_target")
        best_idx = vdf["total_reward"].idxmax()
        best_r0 = float(vdf.loc[best_idx, "r0_over_target"])
        best_reward = float(vdf.loc[best_idx, "total_reward"])

        stable_points = vdf.loc[vdf["total_reward"] >= stable_thr, "r0_over_target"].tolist()
        stable_points = [float(x) for x in stable_points]

        rows.append({
            "thrust_newton": float(thrust),
            "variant": str(variant),
            "global_max_reward_this_thrust": max_reward_all,
            "near_optimal_delta": DELTA,
            "stable_threshold": stable_thr,
            "best_r0": best_r0,
            "best_reward": best_reward,
            "stable_r0_points": stable_points,
        })

out = pd.DataFrame(rows).sort_values(["thrust_newton", "variant"])
out.to_csv(OUT_PATH, index=False)
print(f"[OK] wrote: {OUT_PATH}")
print(out.to_string(index=False))