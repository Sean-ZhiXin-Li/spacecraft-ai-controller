# -*- coding: utf-8 -*-
"""Record a deterministic trajectory for later replay (ExpertController(t,pos,vel))."""
import argparse, math
from typing import Any, Dict, List, Optional

import numpy as np
from utils.seed_utils import set_global_seed, get_seed_from_env
from utils.replay_io import write_jsonl

from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController

def make_env(preset: str, deterministic: bool = True):
    """Try multiple __init__ signatures of OrbitEnv."""
    try:
        return OrbitEnv(preset=preset, deterministic=deterministic)
    except TypeError:
        try:
            return OrbitEnv(preset=preset)
        except TypeError:
            try:
                return OrbitEnv(deterministic=deterministic)
            except TypeError:
                return OrbitEnv()

def safe_reset(env, seed: int):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}

def safe_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out  # obs, r, done, truncated, info
    if isinstance(out, tuple) and len(out) == 4:
        obs, r, done, info = out
        return obs, r, done, False, info
    raise RuntimeError("Unsupported env.step return format")

def extract_pos_vel(obs):
    """Assume obs = [x, y, vx, vy, ...]."""
    arr = np.array(list(obs), dtype=float)
    if arr.size < 4:
        # best-effort fallback
        x = arr[0] if arr.size > 0 else 0.0
        y = arr[1] if arr.size > 1 else 0.0
        vx = arr[2] if arr.size > 2 else 0.0
        vy = arr[3] if arr.size > 3 else 0.0
        return np.array([x, y]), np.array([vx, vy])
    return arr[:2], arr[2:4]

def dig_radius_like(x) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, dict):
        for k in ("target_radius", "target_r", "radius", "R", "r"):
            v = x.get(k)
            if isinstance(v, (int, float)): return float(v)
    return None

def discover_target_radius(env, info: dict, preset: str) -> float:
    # 1) from reset info
    r = dig_radius_like(info)
    if r is not None:
        return r
    # 2) common env attributes
    for attr in ("target_radius", "TARGET_RADIUS", "target_r", "radius"):
        if hasattr(env, attr):
            v = getattr(env, attr)
            if isinstance(v, (int, float)):
                return float(v)
    # 3) fallback by preset
    if (preset or "").lower() == "voyager1":
        return 9.375e12
    # 4) ultimate fallback: use initial |pos|
    try:
        obs0, _ = safe_reset(env, seed=None)
        pos0, _ = extract_pos_vel(obs0)
        return float(np.linalg.norm(pos0))
    except Exception:
        return 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=get_seed_from_env(51))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--preset", type=str, default="voyager1")
    ap.add_argument("--out", type=str, default="replays/day51_v1.jsonl")
    args = ap.parse_args()

    set_global_seed(args.seed)
    env = make_env(args.preset, deterministic=True)

    # reset once to get info (and also to compute fallback target radius if needed)
    obs, info = safe_reset(env, seed=args.seed)
    target_radius = discover_target_radius(env, info, args.preset)

    # instantiate ExpertController with required target_radius
    ctrl = ExpertController(target_radius=target_radius)

    records: List[Dict[str, Any]] = []
    ep_return = 0.0
    t = 0.0

    for step in range(args.steps):
        # derive time t if env exposes it; else use step index
        if isinstance(info, dict):
            t = float(info.get("t", info.get("time", step)))
        else:
            t = float(step)

        pos, vel = extract_pos_vel(obs)
        action = ctrl(t, pos, vel)  # >>> your signature (t, pos, vel)

        nobs, reward, done, truncated, ninfo = safe_step(env, action)
        ep_return += float(reward)

        records.append({
            "t": float(t),
            "obs": list(map(float, np.array(obs, dtype=float))),
            "action": list(map(float, np.array(action, dtype=float))),
            "reward": float(reward),
            "done": bool(done),
        })

        obs, info = nobs, ninfo
        if done or truncated:
            break

    write_jsonl(args.out, records)
    print(f"[record] wrote {len(records)} steps -> {args.out}; return={ep_return:.6f}; target_radius={target_radius:.6g}")

if __name__ == "__main__":
    main()
