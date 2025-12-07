"""
Batch process all replay.npz files under new_week_3/*/*/replay.npz
and generate radius-vs-time plots into analysis/figures/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "logs/new_week_3"
OUTPUT_DIR = "analysis/figures"


def load_radius(replay_path):
    """Load x,y positions from obs and compute radius."""
    data = np.load(replay_path)
    obs = data["obs"]  # shape (N, 4) = [x, y, vx, vy]
    x = obs[:, 0]
    y = obs[:, 1]
    r = np.sqrt(x * x + y * y)
    t = np.arange(len(r))
    return t, r


def plot_radius(t, r, out_path, title):
    """Save radius vs time plot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(t, r)
    plt.xlabel("Time step")
    plt.ylabel("Radius")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved: {out_path}")


def main():
    for scenario in os.listdir(ROOT):
        scenario_dir = os.path.join(ROOT, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        for run in os.listdir(scenario_dir):
            run_dir = os.path.join(scenario_dir, run)
            replay_path = os.path.join(run_dir, "replay.npz")

            if os.path.isfile(replay_path):
                # output file name example:
                # misaligned_entry_run_01_radius.png
                out_name = f"{scenario}_{run}_radius.png"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                t, r = load_radius(replay_path)
                title = f"{scenario} / {run}"
                plot_radius(t, r, out_path, title)


if __name__ == "__main__":
    main()
