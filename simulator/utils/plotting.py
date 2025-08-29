# simulator/utils/plotting.py
from __future__ import annotations
from typing import List, Dict

def debug_plot_trajectory(traj: List[Dict[str, float]]) -> None:
    """
    Lightweight hook: only import matplotlib if actually needed to avoid CI headless issues.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [s["x"] for s in traj]
    ys = [s["y"] for s in traj]
    plt.figure()
    plt.plot(xs, ys, linewidth=1.0)
    plt.title("Trajectory (x-y)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def debug_plot_radius(traj: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    import math
    rs = [math.hypot(s["x"], s["y"]) for s in traj]
    plt.figure()
    plt.plot(rs, linewidth=1.0)
    plt.title("Radius vs. step")
    plt.xlabel("step")
    plt.ylabel("r (m)")
    plt.tight_layout()
    plt.show()
