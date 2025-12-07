"""
Plot radius vs time from a replay (.npz) that stores obs and actions.

Expected structure:
    obs: shape (N, 4) with [x, y, vx, vy]
    actions: shape (N, 2) with thrust vector (not used here)

Usage (from project root):

    python tools/plot/plot_radius_vs_time.py \
        --input path/to/replay.npz \
        --output analysis/figures/replay_radius.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_radius_from_replay(path: str):
    """Load time index and radius from a replay npz file.

    We assume:
        obs[:, 0] = x
        obs[:, 1] = y
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Replay file not found: {path}")

    data = np.load(path)

    if "obs" not in data:
        available = list(data.keys())
        raise KeyError(
            f"'obs' not found in replay file. "
            f"Available keys: {available}"
        )

    obs = data["obs"]
    if obs.ndim != 2 or obs.shape[1] < 2:
        raise ValueError(
            f"Expected obs to have shape (N, >=2), got {obs.shape}"
        )

    x = obs[:, 0]
    y = obs[:, 1]
    r = np.sqrt(x * x + y * y)

    # Use simple index as time axis (0, 1, 2, ...)
    t = np.arange(len(r))

    return t, r


def plot_radius_vs_time(t, r, output_path: str, title: str):
    """Plot radius as a function of time and save to a PNG file."""
    # Make sure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(t, r)
    plt.xlabel("Time step")
    plt.ylabel("Radius")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot radius vs time from a replay npz file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to replay .npz file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help=(
            "Path to output PNG file "
            "(default: analysis/figures/<replay_name>_radius.png)"
        ),
    )

    args = parser.parse_args()

    replay_path = args.input

    # Default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(replay_path))[0]
        args.output = os.path.join(
            "analysis", "figures", f"{base_name}_radius.png"
        )

    t, r = load_radius_from_replay(replay_path)
    title = f"Radius vs Time ({os.path.basename(replay_path)})"
    plot_radius_vs_time(t, r, args.output, title)


if __name__ == "__main__":
    main()
