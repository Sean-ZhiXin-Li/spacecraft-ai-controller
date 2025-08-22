"""
Day 37 quick baseline runner
- Runs 2–3 experts on a small fixed task set
- Exports baseline CSV
- Summarizes metrics -> summary.csv
- (Optional) Replays worst-3 and saves small figures

Usage:
    python tools/day38_quickrun.py --tasks_dir ab/day38/task_specs_fast \
        --out_dir ab/day38 --controllers elliptic_strong transfer_2phase spiral_in \
        --do_replay

Notes:
    - This script assumes your project already provides:
        - script.run_baseline_complex
        - eval.summary
        - script.replay_worst
    - If these modules are missing, the script will raise a helpful error.
"""

import argparse
import os
import sys
import subprocess
import csv
import pandas as pd
import matplotlib.pyplot as plt


def sh(cmd: list, cwd: str = None):
    """Run a command and stream output; raise if non-zero."""
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit={ret})")


def ensure_dirs(*paths: str):
    """Create folders if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def detect_python():
    """Detect python executable on Windows/Unix."""
    # Prefer current interpreter
    return sys.executable or "python"


def build_controller_flags(ctrls):
    """Map simple names to your project's expert keys."""
    # You can directly pass "expert:<name>" if your CLI expects that.
    mapped = []
    name_map = {
        "elliptic_strong": "expert:elliptic_strong",
        "transfer_2phase": "expert:transfer_2phase",
        "spiral_in": "expert:spiral_in",
        "elliptic": "expert:elliptic",
        "transfer": "expert:transfer"
    }
    for c in ctrls:
        mapped.append(name_map.get(c, c if c.startswith("expert:") else f"expert:{c}"))
    return mapped


def try_import(module_name: str):
    """Check if a module is importable; return bool."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def plot_small_charts(summary_csv: str, fig_dir: str):
    """Create minimal bar charts for r_err and ret by controller."""
    # Read CSV and normalize column names
    df = pd.read_csv(summary_csv)

    # Try to guess columns
    # Common columns: 'controller', 'name', 'r_err', 'ret' (sometimes 'return')
    if 'controller' not in df.columns:
        # Fallback: try to split 'name' into controller prefix if present
        if 'name' in df.columns:
            df['controller'] = df['name'].apply(lambda s: str(s).split('|')[0].strip() if isinstance(s, str) else 'unknown')
        else:
            df['controller'] = 'unknown'

    # Normalize metric column names
    if 'ret' not in df.columns and 'return' in df.columns:
        df['ret'] = df['return']

    # Aggregate by controller
    agg = df.groupby('controller', as_index=False).agg({
        'r_err': 'mean',
        'ret': 'mean'
    }).sort_values('r_err', ascending=True)

    ensure_dirs(fig_dir)

    # Plot r_err
    plt.figure()
    plt.bar(agg['controller'], agg['r_err'])
    plt.xticks(rotation=15, ha='right')
    plt.title('Mean r_err by Controller (Day 37)')
    plt.ylabel('Mean r_err')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'day37_r_err_by_controller.png'), dpi=160)

    # Plot ret
    plt.figure()
    plt.bar(agg['controller'], agg['ret'])
    plt.xticks(rotation=15, ha='right')
    plt.title('Mean return by Controller (Day 37)')
    plt.ylabel('Mean return')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'day37_return_by_controller.png'), dpi=160)

    print(f"[OK] Figures saved under: {fig_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", type=str, default="ab/day38/task_specs_fast", help="Directory of task specs")
    parser.add_argument("--out_dir", type=str, default="ab/day38", help="Day 38 output root")
    parser.add_argument("--controllers", nargs="+", default=["elliptic_strong", "transfer_2phase", "spiral_in"])
    parser.add_argument("--do_replay", action="store_true", help="Run worst_3 replay")
    args = parser.parse_args()

    py = detect_python()

    # Define paths
    csv_dir = os.path.join(args.out_dir, "csv")
    replay_dir = os.path.join(args.out_dir, "replay")
    fig_dir = os.path.join(args.out_dir, "figs")
    ensure_dirs(args.out_dir, csv_dir, replay_dir, fig_dir)

    baseline_csv = os.path.join(csv_dir, "baseline_fast.csv")
    summary_csv = os.path.join(csv_dir, "summary.csv")

    # --- Step 0: Ensure tasks_dir exists (fallback to day36 fast tasks if present) ---
    if not os.path.isdir(args.tasks_dir):
        # Try fallback from day36
        fallback = "ab/day36/task_specs_fast"
        if os.path.isdir(fallback):
            print(f"[WARN] {args.tasks_dir} not found. Using fallback: {fallback}")
            args.tasks_dir = fallback
        else:
            raise FileNotFoundError(
                f"Tasks dir not found: {args.tasks_dir}. "
                f"Please prepare a small fast task set under ab/day38/task_specs_fast or ab/day36/task_specs_fast."
            )

    # --- Step 1: Run baseline with selected experts ---
    if not try_import("script.run_baseline_complex"):
        raise ImportError(
            "Missing module 'script.run_baseline_complex'. "
            "Please ensure your project environment is activated and PYTHONPATH is set."
        )
    ctrl_flags = build_controller_flags(args.controllers)

    sh([
        py, "-m", "script.run_baseline_complex",
        "--tasks_dir", args.tasks_dir,
        "--out_csv", baseline_csv,
        "--controllers", *ctrl_flags
    ])

    # --- Step 2: Summarize results -> summary.csv ---
    if not try_import("eval.summary"):
        raise ImportError(
            "Missing module 'eval.summary'. "
            "Please ensure your project provides eval.summary."
        )
    sh([
        py, "-m", "eval.summary",
        "--in_csv", baseline_csv,
        "--out_csv", summary_csv
    ])

    # --- Step 3 (Optional): Replay worst-3 and save figs ---
    if args.do_replay:
        if not try_import("script.replay_worst"):
            raise ImportError(
                "Missing module 'script.replay_worst'. "
                "Please ensure your project provides script.replay_worst."
            )
        sh([
            py, "-m", "script.replay_worst",
            "--in_csv", summary_csv,
            "--out_dir", replay_dir,
            "--top_k", "3"
        ])

    # --- Step 4: Make small charts from summary.csv ---
    if os.path.isfile(summary_csv):
        plot_small_charts(summary_csv, fig_dir)
    else:
        print(f"[WARN] summary.csv not found at {summary_csv}, skip plotting.")

    print("\n[OK] Day 37 quick run complete.")
    print(f"CSV: {baseline_csv}")
    print(f"Summary: {summary_csv}")
    if args.do_replay:
        print(f"Replay dir: {replay_dir}")
    print(f"Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
