import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Config: edit here if needed
RUNS = [
    ("800 always_on (base)", r"analysis/runs/run_200825/traj.npz"),
    ("2000 always_on (A)",   r"analysis/runs/run_804533/traj.npz"),
    ("800 gated (B)",        r"analysis/runs/run_305542/traj.npz"),
    ("2000 gated (C)",       r"analysis/runs/run_750091/traj.npz"),
]

OUT_DIR = Path("analysis/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_traj(npz_path: str):
    d = np.load(npz_path)
    r = d["r"].astype(float)
    vr = d["vr"].astype(float)
    target_r = float(d["target_r"])
    r_err = r - target_r
    t = np.arange(len(r), dtype=int)
    return t, r_err, vr


def main():
    series = []
    for label, p in RUNS:
        t, r_err, vr = load_traj(p)
        series.append((label, t, r_err, vr))

    # Plot 1: r_err(t) overlay
    plt.figure()
    for label, t, r_err, _vr in series:
        plt.plot(t, r_err, label=label)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("r - target_r")
    plt.title("Day16 overlay: radius error r_err(t)")
    plt.legend()
    plt.tight_layout()
    out1 = OUT_DIR / "day16_overlay_rerr_t.png"
    plt.savefig(out1, dpi=200)

    # Plot 2: v_r(t) overlay
    plt.figure()
    for label, t, _r_err, vr in series:
        plt.plot(t, vr, label=label)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("radial velocity v_r")
    plt.title("Day16 overlay: radial velocity v_r(t)")
    plt.legend()
    plt.tight_layout()
    out2 = OUT_DIR / "day16_overlay_vr_t.png"
    plt.savefig(out2, dpi=200)

    print(f"saved: {out1}")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()