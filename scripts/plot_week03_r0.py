import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis/results/week03_r0_sweep_summary.csv")

# split controllers
df_always = df[df["controller_variant"] == "always_on"]
df_gated = df[df["controller_variant"] == "gated"]

plt.figure()

plt.plot(
    df_always["r0_over_target"],
    df_always["avg_radius_error"],
    marker="o",
    label="always_on"
)

plt.plot(
    df_gated["r0_over_target"],
    df_gated["avg_radius_error"],
    marker="o",
    label="gated"
)

plt.xlabel("r0_over_target")
plt.ylabel("avg_radius_error")
plt.title("W03: r0 Sensitivity")
plt.legend()

plt.grid()
plt.savefig("analysis/figs/week03_r0_curve.png")
plt.show()