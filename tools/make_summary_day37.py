"""
Fallback summarizer for Day 37.
- Reads baseline CSV (from script.run_baseline_complex)
- Ensures columns: name, controller, r_err, ret, agent_success
- Writes a normalized summary.csv compatible with later steps
"""

import argparse
import os
import pandas as pd
import re

def infer_controller(row):
    """
    Try to infer 'controller' from various possible fields.
    Priority:
      1) 'controller' column if present
      2) parse from 'name' like 'expert:xxx | task...' or 'xxx | task...'
    """
    # direct
    for key in ["controller", "ctrl", "agent", "policy"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()

    # from name prefix
    name = str(row.get("name", ""))
    if "|" in name:
        left = name.split("|", 1)[0].strip()
        return left
    # sometimes controller is embedded like [expert:xxx] in logs -> keep as-is if present
    m = re.search(r"(expert:[\w\-]+)", name)
    if m:
        return m.group(1)
    return "unknown"

def pick_first(df, candidates, default=None):
    """Return the first column from candidates that exists in df; else default."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.read_csv(args.in_csv)

    # Normalize metric columns
    # r_err could be named r_err / r_error / radius_err
    rerr_col = pick_first(df, ["r_err", "r_error", "radius_err", "radial_err"], None)
    if rerr_col is None:
        raise ValueError("Cannot find r_err column in baseline CSV.")

    # return could be named ret / return / reward
    ret_col = pick_first(df, ["ret", "return", "reward", "episode_return"], None)
    if ret_col is None:
        raise ValueError("Cannot find return/ret column in baseline CSV.")

    # success could be named agent_success / success / succ
    succ_col = pick_first(df, ["agent_success", "success", "succ"], None)
    if succ_col is None:
        # if none, create a dummy 0/1 success based on r_err threshold (very loose fallback)
        df["agent_success"] = (df[rerr_col] < 1e-3).astype(int)
        succ_col = "agent_success"

    # name column
    name_col = pick_first(df, ["name", "task", "id"], None)
    if name_col is None:
        # create a synthetic name if missing
        df["name"] = [f"task_{i}" for i in range(len(df))]
        name_col = "name"

    # Build normalized frame
    out = pd.DataFrame({
        "name": df[name_col],
        "controller": [infer_controller(r) for _, r in df.iterrows()],
        "r_err": df[rerr_col].astype(float),
        "ret": df[ret_col].astype(float),
        "agent_success": df[succ_col].astype(int)
    })

    # Sort for convenience (best first by r_err)
    out = out.sort_values(["controller", "r_err"], ascending=[True, True]).reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote summary to {args.out_csv}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
