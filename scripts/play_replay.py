"""Replay a recorded trajectory and compare metrics."""
import argparse
from utils.seed_utils import set_global_seed, get_seed_from_env
from utils.replay_io import read_jsonl
from utils.compare import l2, within_thresholds

# >>> adjust import if needed
from envs.orbit_env import OrbitEnv
# <<<

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=get_seed_from_env(51))
    ap.add_argument("--preset", type=str, default="voyager1")
    ap.add_argument("--replay", type=str, required=True)
    ap.add_argument("--pos_scale", type=float, default=1.0)
    ap.add_argument("--vel_scale", type=float, default=1.0)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    set_global_seed(args.seed)

    traj = read_jsonl(args.replay)
    env = make_env(args.preset, deterministic=True)

    obs, info = safe_reset(env, seed=args.seed)
    last_obs_ref = traj[-1]["obs"]
    ret_ref = sum(float(s["reward"]) for s in traj)

    ret_new = 0.0
    last_obs_new = obs

    for step in traj:
        action = step["action"]
        obs, reward, done, truncated, info = safe_step(env, action)
        ret_new += float(reward)
        last_obs_new = obs
        if done or truncated:
            break

    pos_l2 = l2(last_obs_new[:2], last_obs_ref[:2]) / max(args.pos_scale, 1.0)
    vel_l2 = l2(last_obs_new[2:4], last_obs_ref[2:4]) / max(args.vel_scale, 1.0)
    ret_abs = abs(ret_new - ret_ref)

    metrics = {"pos_l2": pos_l2, "vel_l2": vel_l2, "ret_abs": ret_abs}
    thresholds = {"pos_l2": 1e-3, "vel_l2": 1e-3, "ret_abs": 1e-3}
    if args.strict:
        thresholds = {"pos_l2": 5e-4, "vel_l2": 5e-4, "ret_abs": 5e-4}

    ok, detail = within_thresholds(metrics, thresholds)
    print(f"[replay] metrics={metrics} thresholds={thresholds} detail={detail}")
    raise SystemExit(0 if ok else 2)

if __name__ == "__main__":
    main()
