"""
Make two small bar charts from Day 37 summary:
- Mean r_err by controller
- Mean return by controller
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--fig_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    if "controller" not in df.columns:
        raise ValueError("summary.csv must contain 'controller' column")

    if "r_err" not in df.columns:
        raise ValueError("summary.csv must contain 'r_err' column")

    if "ret" not in df.columns:
        # Fallback: rename if user placed 'return' instead
        if "return" in df.columns:
            df["ret"] = df["return"]
        else:
            raise ValueError("summary.csv must contain 'ret' (or 'return') column")

    agg = df.groupby("controller", as_index=False).agg({
        "r_err": "mean",
        "ret": "mean"
    }).sort_values("r_err", ascending=True)

    # r_err figure
    plt.figure()
    plt.bar(agg["controller"], agg["r_err"])
    plt.xticks(rotation=15, ha="right")
    plt.title("Mean r_err by Controller (Day 37)")
    plt.ylabel("Mean r_err")
    plt.tight_layout()
    f1 = os.path.join(args.fig_dir, "day37_r_err_by_controller.png")
    plt.savefig(f1, dpi=160)

    # return figure
    plt.figure()
    plt.bar(agg["controller"], agg["ret"])
    plt.xticks(rotation=15, ha="right")
    plt.title("Mean return by Controller (Day 37)")
    plt.ylabel("Mean return")
    plt.tight_layout()
    f2 = os.path.join(args.fig_dir, "day37_return_by_controller.png")
    plt.savefig(f2, dpi=160)

    print(f"[OK] Saved figures:\n  {f1}\n  {f2}")

if __name__ == "__main__":
    main()
