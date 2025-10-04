"""Quick smoke run (~1–2 minutes) using ExpertController(t,pos,vel)."""
import argparse, time
from typing import Any, Dict, Optional

import numpy as np
from utils.seed_utils import set_global_seed, get_seed_from_env

# --- adjust imports only if需要 ---
from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController
# ---------------------------------

def make_env(preset: str, deterministic: bool = True):
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
        return out
    if isinstance(out, tuple) and len(out) == 4:
        obs, r, done, info = out
        return obs, r, done, False, info
    raise RuntimeError("Unsupported env.step return format")

def extract_pos_vel(obs):
    arr = np.array(list(obs), dtype=float)
    if arr.size < 4:
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
    r = dig_radius_like(info)
    if r is not None: return r
    for attr in ("target_radius", "TARGET_RADIUS", "target_r", "radius"):
        if hasattr(env, attr):
            v = getattr(env, attr)
            if isinstance(v, (int, float)): return float(v)
    if (preset or "").lower() == "voyager1":
        return 9.375e12
    try:
        obs0, _ = safe_reset(env, seed=None)
        pos0, _ = extract_pos_vel(obs0)
        return float(np.linalg.norm(pos0))
    except Exception:
        return 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=get_seed_from_env(51))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--preset", type=str, default="voyager1")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    set_global_seed(args.seed)
    env = make_env(args.preset, deterministic=True)

    obs, info = safe_reset(env, seed=args.seed)
    target_radius = discover_target_radius(env, info, args.preset)
    ctrl = ExpertController(target_radius=target_radius)

    ep_return = 0.0
    t0 = time.time()
    steps_run = 0
    t = 0.0

    for _ in range(args.steps):
        if isinstance(info, dict):
            t = float(info.get("t", info.get("time", steps_run)))
        else:
            t = float(steps_run)
        pos, vel = extract_pos_vel(obs)
        action = ctrl(t, pos, vel)  # >>> your signature

        obs, r, done, truncated, info = safe_step(env, action)
        ep_return += float(r)
        steps_run += 1
        if done or truncated:
            break

    dt = time.time() - t0
    metrics = {
        "wall_time_s": dt,
        "steps_run": steps_run,
        "steps_per_sec": steps_run / max(dt, 1e-6),
        "ep_return": float(ep_return),
        "target_radius": float(target_radius),
    }
    thr = {"steps_per_sec": 300.0, "wall_time_s": 120.0}
    if args.strict:
        thr = {"steps_per_sec": 400.0, "wall_time_s": 90.0}

    ok_speed = (metrics["steps_per_sec"] >= thr["steps_per_sec"]) and (metrics["wall_time_s"] <= thr["wall_time_s"])
    print(f"[quickrun] {metrics} thresholds={thr}")
    raise SystemExit(0 if ok_speed else 1)

if __name__ == "__main__":
    main()
