"""
Day 39 quickrun — reuse existing pipeline with new out paths.

Pipeline:
1) run script.run_baseline_complex on a fast task set
2) summarize with make_summary_day37.py (fallback summarizer you already have)
3) plot two tiny charts (r_err mean, return mean)

This file intentionally mirrors your Day38 quickrun structure but with day39
defaults and fewer dependencies (no replay by default).
"""
import argparse
import os
import sys
import subprocess

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
    """Prefer current interpreter."""
    return sys.executable or "python"


def build_controller_flags(ctrls):
    """Map simple names to your project's expert keys (same as Day38)."""
    mapped = []
    name_map = {
        "elliptic_strong": "expert:elliptic_strong",
        "transfer_2phase": "expert:transfer_2phase",
        "spiral_in": "expert:spiral_in",
        "elliptic": "expert:elliptic",
        "transfer": "expert:transfer",
        "elliptic_ecc": "expert:elliptic_ecc",
        "random": "random",
    }
    for c in ctrls:
        mapped.append(name_map.get(c, c if c.startswith("expert:") else f"expert:{c}"))
    return mapped


def plot_small_charts(summary_csv: str, fig_dir: str, title_suffix="(Day 39)"):
    """Two bar charts: mean r_err and mean return by controller."""
    df = pd.read_csv(summary_csv)

    # Normalize columns
    if "controller" not in df.columns:
        raise ValueError("summary.csv must contain 'controller' column")
    if "r_err" not in df.columns:
        raise ValueError("summary.csv must contain 'r_err' column")
    if "ret" not in df.columns:
        if "return" in df.columns:
            df["ret"] = df["return"]
        else:
            raise ValueError("summary.csv must contain 'ret' (or 'return') column")

    agg = df.groupby("controller", as_index=False).agg({"r_err": "mean", "ret": "mean"}).sort_values("r_err")

    ensure_dirs(fig_dir)

    # r_err figure
    plt.figure()
    plt.bar(agg["controller"], agg["r_err"])
    plt.xticks(rotation=15, ha="right")
    plt.title(f"Mean r_err by Controller {title_suffix}")
    plt.ylabel("Mean r_err")
    plt.tight_layout()
    f1 = os.path.join(fig_dir, "day39_r_err_by_controller.png")
    plt.savefig(f1, dpi=160)

    # return figure
    plt.figure()
    plt.bar(agg["controller"], agg["ret"])
    plt.xticks(rotation=15, ha="right")
    plt.title(f"Mean return by Controller {title_suffix}")
    plt.ylabel("Mean return")
    plt.tight_layout()
    f2 = os.path.join(fig_dir, "day39_return_by_controller.png")
    plt.savefig(f2, dpi=160)

    print(f"[OK] Saved figures:\n  {f1}\n  {f2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_dir", type=str, default="ab/day36/task_specs_fast",
                    help="Directory of task specs (fast set).")
    ap.add_argument("--out_dir", type=str, default="ab/day39", help="Day39 output root.")
    ap.add_argument("--controllers", nargs="+",
                    default=["elliptic_strong", "transfer_2phase", "spiral_in"],
                    help="Controller short names or expert:*")
    ap.add_argument("--mu", type=float, default=3.986004418e14, help="Gravitational parameter.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of tasks.")
    args = ap.parse_args()

    py = detect_python()

    # Paths
    csv_dir = os.path.join(args.out_dir, "csv")
    fig_dir = os.path.join(args.out_dir, "figs")
    ensure_dirs(args.out_dir, csv_dir, fig_dir)

    baseline_csv = os.path.join(csv_dir, "baseline_fast.csv")
    summary_csv = os.path.join(csv_dir, "summary.csv")

    # Step 1: baseline run (reuse your complex runner)
    ctrl_flags = build_controller_flags(args.controllers)
    cmd = [
        py, "-m", "script.run_baseline_complex",
        "--tasks_dir", args.tasks_dir,
        "--out_csv", baseline_csv,
        "--controllers", *ctrl_flags,
        "--mu", str(args.mu),
        "--overwrite"
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    sh(cmd)

    # Step 2: summarize (use your Day37 fallback summarizer)
    sh([
        py, "tools/make_summary_day37.py",  # local path call to avoid module import issues
        "--in_csv", baseline_csv,
        "--out_csv", summary_csv
    ])

    # Step 3: small charts
    plot_small_charts(summary_csv, fig_dir, title_suffix="(Day 39)")

    print("\n[OK] Day 39 quickrun complete.")
    print(f"CSV: {baseline_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
