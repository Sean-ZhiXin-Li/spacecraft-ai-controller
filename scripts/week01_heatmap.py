from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "analysis/results/day17_dynamics_summary.csv"
OUTPUT_PNG = ROOT / "analysis" / "figs" / "stability_surface_min_vr_v1.png"

METRIC = "min_vr"


# mapping from run id to experiment configuration
run_config_map = {
    "run_200825": ("always_on", 800),
    "run_305542": ("always_on", 2000),
    "run_750091": ("gated", 800),
    "run_804533": ("gated", 2000),
}


def main():

    df = pd.read_csv(INPUT_CSV)

    controllers = []
    thrusts = []

    for run in df["run"]:
        controller, thrust = run_config_map[run]
        controllers.append(controller)
        thrusts.append(thrust)

    df["controller"] = controllers
    df["thrust"] = thrusts

    plot_df = df[["controller", "thrust", METRIC]]

    pivot = plot_df.pivot(
        index="controller",
        columns="thrust",
        values=METRIC
    )

    controller_order = ["gated", "always_on"]
    thrust_order = sorted(plot_df["thrust"].unique())

    pivot = pivot.reindex(index=controller_order)
    pivot = pivot.reindex(columns=thrust_order)

    data = pivot.values

    fig, ax = plt.subplots(figsize=(6,3))

    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(thrust_order)))
    ax.set_xticklabels([f"{t}N" for t in thrust_order])

    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels(controller_order)

    ax.set_xlabel("Thrust")
    ax.set_ylabel("Controller")

    ax.set_title("Stability Surface v1")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i,j]
            ax.text(j, i, f"{val:.2e}", ha="center", va="center")

    plt.colorbar(im)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_PNG, dpi=200)

    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()