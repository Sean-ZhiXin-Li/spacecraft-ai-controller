import os, sys, json, argparse
import numpy as np

# Ensure local imports work when running from repo root
CUR = os.path.dirname(__file__)
if CUR not in sys.path:
    sys.path.insert(0, CUR)

from metrics_core import (
    radial_distance, radial_error, norm_error,
    convergence_step, steady_state_error, oscillation_index,
    saturation_rate, under_power_ratio, drift_percent
)

def load_npz(npz_path: str):
    d = np.load(npz_path)
    # Expect keys: ['obs','actions', ...]
    obs = d["obs"]       # shape (T, obs_dim) -> assume [x,y,vx,vy,...]
    act = d["actions"]   # shape (T, act_dim)
    return obs, act

def infer_target_radius(meta_json: str, fallback: float = None):
    if meta_json and os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        extra = meta.get("extra", {})
        rt = extra.get("target_radius")  # e.g., 9.375e12
        if rt is not None:
            return float(rt)
    return float(fallback) if fallback is not None else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)  # metrics.json path
    ap.add_argument("--r-target", type=float, default=None, help="Override target radius if meta not provided")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--win", type=int, default=20)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--sat-high", type=float, default=0.95)
    ap.add_argument("--under-e", type=float, default=0.01)
    ap.add_argument("--under-a", type=float, default=0.2)
    args = ap.parse_args()

    obs, act = load_npz(args.npz)
    xy = obs[:, :2]
    r = radial_distance(xy)

    r_target = args.r_target or infer_target_radius(args.meta)
    if r_target is None:
        raise ValueError("Target radius is required. Provide --r-target or store in meta.extra.target_radius")

    e = radial_error(r, r_target)
    ne = norm_error(e, r_target)

    T = min(len(ne), act.shape[0])
    e = e[:T];
    ne = ne[:T];
    act = act[:T]

    metrics = {
        "r_target": r_target,
        "convergence_step": convergence_step(ne, eps=args.eps, window=args.win),
        "steady_state_error": steady_state_error(ne, tail=args.tail, reducer="mean"),
        "oscillation_index": oscillation_index(e, tail=args.tail),
        "saturation_rate": saturation_rate(act, high=args.sat_high),
        "under_power_ratio": under_power_ratio(act, ne, e_thresh=args.under_e, a_low=args.under_a),
        "drift_percent": drift_percent(ne)
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[metrics] wrote", args.out)
    for k, v in metrics.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
