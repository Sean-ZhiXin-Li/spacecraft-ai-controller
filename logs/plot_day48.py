"""
Day 48 – Quick visualization
Reads logs/day48/test_log_day48.csv and makes a bar chart of total_reward per scenario,
with annotations for dv1 and final_orbit_error.

Usage:
    python plot_day48.py
Output:
    logs/day48/day48_summary.png
"""

from pathlib import Path
import math

# Try pandas first; fall back to csv.DictReader if not installed
def load_csv(csv_path):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # ensure required columns exist
        required = ["scenario", "total_reward", "dv1", "final_orbit_error"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")
        return [
            {
                "scenario": str(r["scenario"]),
                "total_reward": float(r["total_reward"]),
                "dv1": float(r["dv1"]),
                "final_orbit_error": float(r["final_orbit_error"]),
            }
            for _, r in df.iterrows()
        ]
    except ImportError:
        import csv
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        # normalize
        out = []
        for r in rows:
            out.append({
                "scenario": str(r["scenario"]),
                "total_reward": float(r["total_reward"]),
                "dv1": float(r["dv1"]),
                "final_orbit_error": float(r["final_orbit_error"]),
            })
        return out


def main():
    import matplotlib.pyplot as plt

    csv_path = Path("logs/day48/test_log_day48.csv")
    out_path = Path("logs/day48/day48_summary.png")

    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}\n"
                         f"Please run: python -m orbit_env.tests.test_day48")

    data = load_csv(csv_path)

    # Sort by a fixed scenario order for readability
    order = ["circular", "elliptic", "transfer"]
    data = sorted(data, key=lambda r: order.index(r["scenario"]) if r["scenario"] in order else 999)

    scenarios = [d["scenario"] for d in data]
    rewards = [d["total_reward"] for d in data]
    dv1s = [d["dv1"] for d in data]
    errs = [d["final_orbit_error"] for d in data]

    # Create bar chart for rewards
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x = list(range(len(scenarios)))
    bars = ax.bar(x, rewards)  # do not set explicit colors (keep defaults)

    # Labels & title
    ax.set_title("Day 48 – Total Reward by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Total Reward")

    # Annotate dv1 and final_orbit_error on each bar
    for i, (bx, r, dv, err) in enumerate(zip(bars, rewards, dv1s, errs)):
        # place text slightly above the bar top (or below if negative)
        y = bx.get_height()
        offset = 0.01 * (max(rewards) - min(rewards) if len(rewards) > 1 else (abs(y) + 1.0))
        y_text = y + (offset if y >= 0 else -offset)
        txt = f"dv1={dv:.1f} m/s\nerr={err:.3f}"
        ax.text(bx.get_x() + bx.get_width()/2.0, y_text, txt,
                ha="center", va="bottom" if y >= 0 else "top", fontsize=9)

    # Nice layout
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved plot -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
