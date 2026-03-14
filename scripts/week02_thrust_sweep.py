from pathlib import Path
from typing import Any
from day17_batch_probe import summarize_one_run

import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import time


# PL_W02 fixed sweep config

THRUST_LEVELS = [500, 800, 1000, 1200, 1500, 1800, 2000, 2500]
CONTROLLERS = ["always_on", "gated"]

RESULTS_CSV = Path("analysis/results/week02_thrust_sweep_summary.csv")
FIG_PATH = Path("analysis/figs/stability_surface_week02_min_vr.png")
RUNS_DIR = Path("analysis/runs")
QUICK_COMPARE_SCRIPT = Path("src/quick_compare_v3_v4.py")


# Helpers

def get_existing_run_dirs() -> set[Path]:
    """Return current run directories under analysis/runs."""
    if not RUNS_DIR.exists():
        return set()
    return {p for p in RUNS_DIR.iterdir() if p.is_dir()}


def find_new_run_dir(before: set[Path], after: set[Path]) -> Path | None:
    """Find the newly created run directory by set difference."""
    new_dirs = list(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]

    if len(new_dirs) == 0:
        return None

    # If multiple new dirs appear, choose the most recently modified one
    new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return new_dirs[0]


def launch_one_experiment(controller: str, thrust: float) -> Path | None:
    """
    Launch one experiment by calling src/quick_compare_v3_v4.py
    with environment-variable overrides.

    Returns the new run directory if found, else None.
    """
    before = get_existing_run_dirs()

    env = os.environ.copy()
    env["CONTROLLER_VARIANT"] = controller
    env["THRUST_NEWTON"] = str(float(thrust))

    print(f"[EXEC] Launching quick_compare | controller={controller} | thrust={thrust}")

    result = subprocess.run(
        ["python", str(QUICK_COMPARE_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"Experiment failed | controller={controller} | thrust={thrust} | returncode={result.returncode}"
        )

    # Small pause in case filesystem timestamp/update is slightly delayed
    time.sleep(0.5)

    after = get_existing_run_dirs()
    new_run_dir = find_new_run_dir(before, after)
    return new_run_dir


# Core experiment interface

def run_single_config(controller: str, thrust: float) -> dict[str, Any]:
    """
    Run one configuration, locate the new run folder,
    summarize traj.npz, and return one CSV row.
    """
    try:
        run_dir = launch_one_experiment(controller=controller, thrust=thrust)

        if run_dir is None:
            return {
                "run": f"{controller}_{thrust}",
                "controller": controller,
                "thrust": thrust,
                "status": "new_run_not_found",
                "min_r_err": float("nan"),
                "max_r_err": float("nan"),
                "min_vr": float("nan"),
                "t_flip": float("nan"),
                "t_cross": float("nan"),
                "delta_r": float("nan"),
            }

        row = summarize_one_run(run_dir)

        if row is None:
            return {
                "run": run_dir.name,
                "controller": controller,
                "thrust": thrust,
                "status": "missing_traj",
                "min_r_err": float("nan"),
                "max_r_err": float("nan"),
                "min_vr": float("nan"),
                "t_flip": float("nan"),
                "t_cross": float("nan"),
                "delta_r": float("nan"),
            }

        row["controller"] = controller
        row["thrust"] = thrust
        return row

    except Exception as e:
        return {
            "run": f"{controller}_{thrust}",
            "controller": controller,
            "thrust": thrust,
            "status": f"error: {e}",
            "min_r_err": float("nan"),
            "max_r_err": float("nan"),
            "min_vr": float("nan"),
            "t_flip": float("nan"),
            "t_cross": float("nan"),
            "delta_r": float("nan"),
        }


# Sweep collection

def collect_runs() -> pd.DataFrame:
    rows = []

    for controller in CONTROLLERS:
        for thrust in THRUST_LEVELS:
            print(f"[RUN] controller={controller}, thrust={thrust}")
            row = run_single_config(controller=controller, thrust=thrust)
            rows.append(row)

    df = pd.DataFrame(rows)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print(f"[OK] Saved summary CSV to: {RESULTS_CSV}")
    return df


# Plotting

def plot_surface(df: pd.DataFrame) -> None:
    """
    Plot the stability surface using min_vr as the primary metric.
    """
    if df.empty:
        print("[WARN] DataFrame is empty. Skip plotting.")
        return

    plot_df = df.copy()
    plot_df["min_vr"] = pd.to_numeric(plot_df["min_vr"], errors="coerce")

    if plot_df["min_vr"].isna().all():
        print("[WARN] min_vr is all NaN. Skip plotting for now.")
        return

    plot_df["controller"] = pd.Categorical(
        plot_df["controller"],
        categories=["always_on", "gated"],
        ordered=True,
    )

    pivot = plot_df.pivot(index="controller", columns="thrust", values="min_vr")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(10, 3))
    plt.imshow(pivot.values.astype(float), aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="min_vr")
    plt.xlabel("thrust")
    plt.ylabel("controller")
    plt.title("PL_W02 Stability Surface (min_vr)")
    plt.tight_layout()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

    print(f"[OK] Saved figure to: {FIG_PATH}")


# Main

def main() -> None:
    df = collect_runs()
    print(df)
    plot_surface(df)


if __name__ == "__main__":
    main()