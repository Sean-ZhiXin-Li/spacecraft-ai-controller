from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "analysis" / "runs"
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
SESSION6_JSON = PROJECT_ROOT / "analysis" / "SESSION6_metrics_upgrade.json"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=np.nan):
    """Convert a value to float safely."""
    try:
        return float(x)
    except Exception:
        return default


def find_run_dirs(runs_dir: Path):
    """
    Find all run directories that contain both traj.npz and metrics.json.
    """
    if not runs_dir.exists():
        return []

    run_dirs = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        traj_path = p / "traj.npz"
        metrics_path = p / "metrics.json"
        if traj_path.exists() and metrics_path.exists():
            run_dirs.append(p)

    return sorted(run_dirs)


def load_run(run_dir: Path) -> dict:
    """
    Load one run directory and return structured data.
    """
    traj_path = run_dir / "traj.npz"
    metrics_path = run_dir / "metrics.json"

    traj = np.load(traj_path)
    meta = load_json(metrics_path)

    label = meta.get("label", run_dir.name)
    thrust_newton = safe_float(meta.get("thrust_newton", np.nan))
    r0_over_target = safe_float(meta.get("r0_over_target", np.nan))
    total_reward = safe_float(meta.get("total_reward", np.nan))
    saturation_rate_mean = safe_float(meta.get("saturation_rate_mean", np.nan))
    target_radius = safe_float(meta.get("target_radius", np.nan))

    data = {
        "run_dir": run_dir,
        "label": label,
        "thrust_newton": thrust_newton,
        "r0_over_target": r0_over_target,
        "total_reward": total_reward,
        "saturation_rate_mean": saturation_rate_mean,
        "target_radius": target_radius,
        "r": traj["r"] if "r" in traj.files else None,
        "vr": traj["vr"] if "vr" in traj.files else None,
        "cos_tr": traj["cos_tr"] if "cos_tr" in traj.files else None,
        "cos_tt": traj["cos_tt"] if "cos_tt" in traj.files else None,
    }

    return data


def summarize_direction_metrics(run: dict) -> dict:
    """
    Compute summary statistics from one run.
    """
    cos_tr = run["cos_tr"]
    cos_tt = run["cos_tt"]
    vr = run["vr"]
    r = run["r"]

    row = {
        "label": run["label"],
        "thrust_newton": run["thrust_newton"],
        "r0_over_target": run["r0_over_target"],
        "total_reward": run["total_reward"],
        "saturation_rate_mean": run["saturation_rate_mean"],
        "target_radius": run["target_radius"],
    }

    if cos_tr is not None:
        row["mean_cos_tr"] = float(np.nanmean(cos_tr))
        row["std_cos_tr"] = float(np.nanstd(cos_tr))
        row["min_cos_tr"] = float(np.nanmin(cos_tr))
        row["max_cos_tr"] = float(np.nanmax(cos_tr))
    else:
        row["mean_cos_tr"] = np.nan
        row["std_cos_tr"] = np.nan
        row["min_cos_tr"] = np.nan
        row["max_cos_tr"] = np.nan

    if cos_tt is not None:
        row["mean_cos_tt"] = float(np.nanmean(cos_tt))
        row["std_cos_tt"] = float(np.nanstd(cos_tt))
        row["min_cos_tt"] = float(np.nanmin(cos_tt))
        row["max_cos_tt"] = float(np.nanmax(cos_tt))
    else:
        row["mean_cos_tt"] = np.nan
        row["std_cos_tt"] = np.nan
        row["min_cos_tt"] = np.nan
        row["max_cos_tt"] = np.nan

    if vr is not None:
        row["mean_vr"] = float(np.nanmean(vr))
        row["min_vr"] = float(np.nanmin(vr))
        row["max_vr"] = float(np.nanmax(vr))
    else:
        row["mean_vr"] = np.nan
        row["min_vr"] = np.nan
        row["max_vr"] = np.nan

    if r is not None and np.isfinite(run["target_radius"]):
        radius_error = np.abs(r - run["target_radius"])
        row["mean_radius_error_from_traj"] = float(np.nanmean(radius_error))
        row["final_radius_error_from_traj"] = float(np.abs(r[-1] - run["target_radius"]))
    else:
        row["mean_radius_error_from_traj"] = np.nan
        row["final_radius_error_from_traj"] = np.nan

    return row


def save_summary_csv(runs: list, output_csv: Path):
    """
    Save a summary CSV for all discovered runs.
    """
    rows = [summarize_direction_metrics(run) for run in runs]
    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(["thrust_newton", "label"], na_position="last")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Wrote summary CSV: {output_csv}")

    return df


def plot_cos_vs_time(runs: list, thrust_filter=None, output_path: Path = None):
    """
    Plot cos(thrust, radial) over time for multiple controllers.
    """
    plt.figure(figsize=(10, 6))

    plotted = False
    for run in runs:
        if thrust_filter is not None and not math.isclose(run["thrust_newton"], thrust_filter, rel_tol=0.0, abs_tol=1e-9):
            continue
        if run["cos_tr"] is None:
            continue

        x = np.arange(len(run["cos_tr"]))
        plt.plot(x, run["cos_tr"], label=run["label"])
        plotted = True

    plt.xlabel("Timestep")
    plt.ylabel("cos(thrust, radial)")
    plt.title(f"Direction Structure Over Time (cos_tr) | thrust={thrust_filter}")
    plt.legend()
    plt.grid(True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved figure: {output_path}")

    if plotted:
        plt.show()
    else:
        plt.close()
        print("[WARN] No matching runs found for cos_tr plot.")


def plot_cos_tt_vs_time(runs: list, thrust_filter=None, output_path: Path = None):
    """
    Plot cos(thrust, tangential) over time for multiple controllers.
    """
    plt.figure(figsize=(10, 6))

    plotted = False
    for run in runs:
        if thrust_filter is not None and not math.isclose(run["thrust_newton"], thrust_filter, rel_tol=0.0, abs_tol=1e-9):
            continue
        if run["cos_tt"] is None:
            continue

        x = np.arange(len(run["cos_tt"]))
        plt.plot(x, run["cos_tt"], label=run["label"])
        plotted = True

    plt.xlabel("Timestep")
    plt.ylabel("cos(thrust, tangential)")
    plt.title(f"Tangential Structure Over Time (cos_tt) | thrust={thrust_filter}")
    plt.legend()
    plt.grid(True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved figure: {output_path}")

    if plotted:
        plt.show()
    else:
        plt.close()
        print("[WARN] No matching runs found for cos_tt plot.")


def plot_direction_signature(runs: list, thrust_filter=None, output_path: Path = None):
    """
    Plot scatter of (cos_tr, cos_tt) for each controller.
    """
    plt.figure(figsize=(7, 7))

    plotted = False
    for run in runs:
        if thrust_filter is not None and not math.isclose(run["thrust_newton"], thrust_filter, rel_tol=0.0, abs_tol=1e-9):
            continue
        if run["cos_tr"] is None or run["cos_tt"] is None:
            continue

        plt.scatter(
            run["cos_tr"],
            run["cos_tt"],
            s=6,
            alpha=0.35,
            label=run["label"],
        )
        plotted = True

    plt.xlabel("cos(thrust, radial)")
    plt.ylabel("cos(thrust, tangential)")
    plt.title(f"Direction Signature Scatter | thrust={thrust_filter}")
    plt.grid(True)
    plt.legend()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved figure: {output_path}")

    if plotted:
        plt.show()
    else:
        plt.close()
        print("[WARN] No matching runs found for direction signature plot.")


def plot_vr_vs_time(runs: list, thrust_filter=None, output_path: Path = None):
    """
    Plot radial velocity over time for multiple controllers.
    """
    plt.figure(figsize=(10, 6))

    plotted = False
    for run in runs:
        if thrust_filter is not None and not math.isclose(run["thrust_newton"], thrust_filter, rel_tol=0.0, abs_tol=1e-9):
            continue
        if run["vr"] is None:
            continue

        x = np.arange(len(run["vr"]))
        plt.plot(x, run["vr"], label=run["label"])
        plotted = True

    plt.xlabel("Timestep")
    plt.ylabel("Radial velocity (v_r)")
    plt.title(f"Radial Velocity Over Time | thrust={thrust_filter}")
    plt.legend()
    plt.grid(True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved figure: {output_path}")

    if plotted:
        plt.show()
    else:
        plt.close()
        print("[WARN] No matching runs found for v_r plot.")


def load_session6_results(path: Path) -> pd.DataFrame:
    """
    Load SESSION6_metrics_upgrade.json and return a DataFrame.
    """
    if not path.exists():
        print(f"[WARN] SESSION6 JSON not found: {path}")
        return pd.DataFrame()

    payload = load_json(path)
    rows = payload.get("results", [])
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    keep_cols = [
        "label",
        "steps",
        "total_reward",
        "final_r",
        "avg_radius_error",
        "avg_jitter",
        "saturation_rate_mean",
        "reward_minus_lambda_sat",
        "target_radius",
        "r0_over_target",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def print_interpretation(df: pd.DataFrame):
    """
    Print a compact text interpretation for quick inspection.
    """
    if df.empty:
        print("[WARN] No summary dataframe available.")
        return

    print("\n=== DAY 6 SUMMARY TABLE ===")
    print(df.to_string(index=False))

    if {"label", "mean_cos_tr", "mean_cos_tt", "total_reward", "saturation_rate_mean"}.issubset(df.columns):
        print("\n=== QUICK INTERPRETATION ===")
        for _, row in df.iterrows():
            print(
                f"{row['label']:>10} | "
                f"mean_cos_tr={row['mean_cos_tr']:+.4f} | "
                f"mean_cos_tt={row['mean_cos_tt']:+.4f} | "
                f"reward={row['total_reward']:.3e} | "
                f"sat={row['saturation_rate_mean']:.6f}"
            )
def filter_runs_by_run_dir_name(runs: list, keep_names: set[str]) -> list:
    """
    Keep only runs whose folder names are explicitly listed.
    """
    filtered = []
    for run in runs:
        run_name = run["run_dir"].name
        if run_name in keep_names:
            filtered.append(run)

    print(f"[OK] Explicit run-name filter: kept {len(filtered)} / {len(runs)}")
    print(f"[OK] Keep names: {sorted(keep_names)}")
    return filtered

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(RUNS_DIR)
    if not run_dirs:
        print(f"[WARN] No run directories found under: {RUNS_DIR}")
        return

    runs_all = [load_run(run_dir) for run_dir in run_dirs]

    session6_df = load_session6_results(SESSION6_JSON)
    if session6_df.empty:
        print(f"[WARN] SESSION6 JSON is empty or missing: {SESSION6_JSON}")
        return

    session6_csv = RESULTS_DIR / "pl_w03_day6_session6_metrics.csv"
    session6_df.to_csv(session6_csv, index=False)
    print(f"[OK] Wrote session6 CSV: {session6_csv}")
    print("\n=== SESSION6 METRICS ===")
    print(session6_df.to_string(index=False))

    high_thrust = 100000.0

    # Explicit whitelist for today's confirmed run folders
    today_run_names = {
        "run_964254",
        "run_854705",
        "run_835152",
    }

    runs = filter_runs_by_run_dir_name(runs_all, today_run_names)

    if not runs:
        print("[WARN] No runs left after explicit run-name filtering.")
        return

    summary_csv = RESULTS_DIR / "pl_w03_day6_structure_summary.csv"
    df_summary = save_summary_csv(runs, summary_csv)
    print_interpretation(df_summary)

    plot_cos_vs_time(
        runs,
        thrust_filter=high_thrust,
        output_path=RESULTS_DIR / "pl_w03_day6_cos_tr_vs_time_100000.png",
    )

    plot_cos_tt_vs_time(
        runs,
        thrust_filter=high_thrust,
        output_path=RESULTS_DIR / "pl_w03_day6_cos_tt_vs_time_100000.png",
    )

    plot_direction_signature(
        runs,
        thrust_filter=high_thrust,
        output_path=RESULTS_DIR / "pl_w03_day6_direction_signature_100000.png",
    )

    plot_vr_vs_time(
        runs,
        thrust_filter=high_thrust,
        output_path=RESULTS_DIR / "pl_w03_day6_vr_vs_time_100000.png",
    )

if __name__ == "__main__":
    main()
