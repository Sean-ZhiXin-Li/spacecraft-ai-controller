import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "analysis/results/whpl14_variant_x_thrust_x_r0.csv"
OUT_DIR = "analysis/figs"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for thrust in [800.0, 2000.0]:
    sub = df.loc[df["thrust_newton"] == thrust].copy()
    if sub.empty:
        print(f"[WARN] no rows for thrust={thrust}")
        continue

    # Aggregate in case of duplicates (e.g., multiple runs/seeds)
    g = (
        sub.groupby(["controller_variant", "r0_over_target"], as_index=False)["saturation_rate_mean"]
        .mean()
        .sort_values(["controller_variant", "r0_over_target"])
    )

    plt.figure()
    for variant in sorted(g["controller_variant"].unique()):
        vdf = g.loc[g["controller_variant"] == variant].sort_values("r0_over_target")
        plt.plot(vdf["r0_over_target"], vdf["saturation_rate_mean"], marker="o", label=variant)

    plt.title(f"WHPL15 Saturation vs r0 | thrust={int(thrust)}N")
    plt.xlabel("r0_over_target")
    plt.ylabel("saturation_rate_mean (mean)")
    plt.legend()

    out_path = f"{OUT_DIR}/whpl15_saturation_vs_r0_{int(thrust)}N.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[OK] saved: {out_path}")