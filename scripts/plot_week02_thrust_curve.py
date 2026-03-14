import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULT_FILE = Path("analysis/results/week02_thrust_sweep_summary.csv")
FIG_DIR = Path("analysis/figs")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(RESULT_FILE)

# Separate controllers
always_on = df[df["controller"] == "always_on"].sort_values("thrust")
gated = df[df["controller"] == "gated"].sort_values("thrust")

# Plot
plt.figure(figsize=(8,6))

plt.plot(
    always_on["thrust"],
    always_on["min_vr"],
    marker="o",
    label="always_on"
)

plt.plot(
    gated["thrust"],
    gated["min_vr"],
    marker="s",
    label="gated"
)

# Labels
plt.xlabel("Thrust (N)")
plt.ylabel("Minimum Radial Velocity (min_vr)")
plt.title("W02: Thrust vs Minimum Radial Velocity")

plt.legend()
plt.grid(True)

# Save
save_path = FIG_DIR / "thrust_vs_min_vr_week02.png"
plt.savefig(save_path, dpi=200)

print(f"Saved: {save_path}")