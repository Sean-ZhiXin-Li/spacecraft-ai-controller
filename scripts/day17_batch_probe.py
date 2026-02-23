"""
Day17: Batch dynamics probe summarizer.

Input:
  - analysis/runs/*/traj.npz (for each run folder)

Output:
  - analysis/results/day17_dynamics_summary.csv

Metrics (minimal set):
  - min_r_err
  - max_r_err
  - min_vr
  - t_flip   (first time v_r changes sign)
  - t_cross  (first time r_err crosses zero)
  - delta_r  (r[-1] - r[0] if r exists; else r_err[-1] - r_err[0])
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


RUNS_DIR = Path("analysis/runs")
OUT_CSV = Path("analysis/results/day17_dynamics_summary.csv")


def _first_sign_change_time(t: np.ndarray, x: np.ndarray) -> float:
    """Return the first time when x changes sign (ignoring zeros). NaN if never."""
    if t.size == 0 or x.size == 0:
        return float("nan")

    # Normalize lengths
    n = min(t.size, x.size)
    t = t[:n]
    x = x[:n]

    # Compute sign ignoring zeros by forward-filling last non-zero sign
    s = np.sign(x)
    last = 0.0
    for i in range(n):
        if s[i] != 0:
            last = s[i]
            break
    if last == 0.0:
        return float("nan")

    prev = last
    for i in range(1, n):
        si = s[i]
        if si == 0:
            continue
        if si != prev:
            return float(t[i])
        prev = si
    return float("nan")


def _first_zero_cross_time(t: np.ndarray, x: np.ndarray) -> float:
    """Return the first time when x crosses 0 (sign change). NaN if never."""
    if t.size == 0 or x.size == 0:
        return float("nan")

    n = min(t.size, x.size)
    t = t[:n]
    x = x[:n]

    s = np.sign(x)
    # Find first pair with opposite non-zero signs or direct hit near zero
    for i in range(1, n):
        if abs(x[i]) == 0:
            return float(t[i])
        if s[i] == 0 or s[i - 1] == 0:
            continue
        if s[i] != s[i - 1]:
            return float(t[i])
    return float("nan")


def _load_traj_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load traj.npz into a dict of arrays."""
    data = np.load(npz_path, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k])
    return out


def _pick_key(d: Dict[str, np.ndarray], candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def _safe_min(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.min(x))


def _safe_max(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.max(x))


def summarize_one_run(run_dir: Path) -> Optional[dict]:
    npz_path = run_dir / "traj.npz"
    if not npz_path.exists():
        return None

    d = _load_traj_npz(npz_path)

    # Common key fallbacks
    t_key = _pick_key(d, ("t", "time", "ts"))  # may be missing
    r_key = _pick_key(d, ("r", "radius", "r_arr"))
    target_r_key = _pick_key(d, ("target_r", "r_target", "target_radius"))
    vr_key = _pick_key(d, ("v_r", "vr", "radial_velocity"))

    if r_key is None or target_r_key is None or vr_key is None:
        print(f"[WARN] {run_dir.name}: missing_keys | keys={sorted(list(d.keys()))}")
        # Minimal requirement not met -> still output row with NaNs but mark missing
        return {
            "run": run_dir.name,
            "status": "missing_keys",
            "min_r_err": float("nan"),
            "max_r_err": float("nan"),
            "min_vr": float("nan"),
            "t_flip": float("nan"),
            "t_cross": float("nan"),
            "delta_r": float("nan"),
        }

    # Load minimal signals present in traj.npz
    r = np.asarray(d[r_key]).astype(float).reshape(-1)
    target_r = np.asarray(d[target_r_key]).astype(float).reshape(-1)
    vr = np.asarray(d[vr_key]).astype(float).reshape(-1)

    # Align lengths
    n = min(r.size, target_r.size, vr.size)
    r = r[:n]
    target_r = target_r[:n]
    vr = vr[:n]

    # Derived minimal metrics
    r_err = r - target_r

    # Time fallback: use step index (traj.npz has no explicit t)
    t = np.arange(n, dtype=float)

    # delta_r (real radius delta)
    delta_r = float(r[-1] - r[0]) if n >= 2 else float("nan")

    row = {
        "run": run_dir.name,
        "status": "ok",
        "min_r_err": _safe_min(r_err),
        "max_r_err": _safe_max(r_err),
        "min_vr": _safe_min(vr),
        "t_flip": _first_sign_change_time(t, vr),
        "t_cross": _first_zero_cross_time(t, r_err),
        "delta_r": delta_r,
    }
    return row


def main() -> int:
    run_dirs = sorted([p for p in RUNS_DIR.glob("*") if p.is_dir()])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rd in run_dirs:
        row = summarize_one_run(rd)
        if row is not None:
            rows.append(row)

    fieldnames = ["run", "status", "min_r_err", "max_r_err", "min_vr", "t_flip", "t_cross", "delta_r"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote: {OUT_CSV} | rows={len(rows)}")
    # Optional: show quick stats without plotting
    ok_cnt = sum(1 for r in rows if r["status"] == "ok")
    miss_cnt = sum(1 for r in rows if r["status"] != "ok")
    print(f"[INFO] status: ok={ok_cnt} non_ok={miss_cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
