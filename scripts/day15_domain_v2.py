import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "analysis/results/whpl14_variant_x_thrust_x_r0.csv"
OUT_DIR = "analysis/figs"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

DELTA = 3000.0  # Near-optimal band width

for thrust in [800.0, 2000.0]:
    sub = df.loc[df["thrust_newton"] == thrust].copy()
    if sub.empty:
        print(f"[WARN] no rows for thrust={thrust}")
        continue

    max_reward = float(sub["total_reward"].max())
    stable_threshold = max_reward - DELTA

    sub.loc[:, "is_stable"] = sub["total_reward"] >= stable_threshold

    plt.figure()

    for variant in sorted(sub["controller_variant"].unique()):
        vdf = sub.loc[sub["controller_variant"] == variant].copy()
        vdf = vdf.sort_values("r0_over_target")

        # Base curve
        plt.plot(
            vdf["r0_over_target"],
            vdf["total_reward"],
            marker="o",
            label=variant
        )

        # Highlight stable points
        stable_pts = vdf.loc[vdf["is_stable"]]
        plt.scatter(
            stable_pts["r0_over_target"],
            stable_pts["total_reward"],
            marker="o",
            facecolors="none",
            edgecolors="black",
            s=120
        )

        # Print stable r0 list for this variant
        stable_r0s = stable_pts["r0_over_target"].tolist()
        print(f"[{int(thrust)}N | {variant}] stable r0_over_target =", [float(x) for x in stable_r0s])

    plt.axhline(stable_threshold, linestyle="--")
    plt.title(f"WHPL15 Near-Optimal Stability Domain | thrust={int(thrust)}N")
    plt.xlabel("r0_over_target")
    plt.ylabel("total_reward")
    plt.legend()

    out_path = f"{OUT_DIR}/whpl15_domain_{int(thrust)}N.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"{int(thrust)}N max_reward = {max_reward}")
    print(f"{int(thrust)}N stable_threshold = {stable_threshold}")
    print(f"[OK] saved: {out_path}\n")