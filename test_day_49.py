# -*- coding: utf-8 -*-
"""
Day 49: Validate ExpertController (v3.1 behavior) on MultiOrbitEnv across 3 scenarios
- Scenarios: circular / elliptic / transfer
- Controller: controller.expert_controller.ExpertController (kept intact; we only adapt around it)
- Output CSV: logs/day49/test_log_day49.csv

Notes:
- All comments are in English.
- We DO NOT modify your ExpertController logic or defaults.
- We only pass required args (e.g., target_radius) and adapt obs -> (pos, vel).
"""

DEBUG_D49 = True  # set to False after things look good

import os
import sys
import csv
import time
import argparse
import inspect
from pathlib import Path
from datetime import datetime
import numpy as np
import math

# -----------------------------------------------------------------------------
# Action adaptation
# -----------------------------------------------------------------------------
def adapt_action_to_env(thrust_vec, env):
    """
    Adapt a 2D thrust vector [tx, ty] to the environment's expected action format.
    Rules:
      - If action_space is Box(2,), pass through (then clip).
      - If Box(3,), assume [tx, ty, enable] with enable=1.
      - If Box(1,), use thrust magnitude as a fallback.
      - If bounds look normalized (|high| <= ~1), normalize accordingly.
      - Always clip to action_space bounds if available.
    """
    tv = np.asarray(thrust_vec, dtype=float).reshape(-1)
    space = getattr(env, "action_space", None)
    a = tv.copy()

    if hasattr(space, "shape") and isinstance(space.shape, tuple):
        shp = space.shape
        if shp == (2,):
            a = tv
        elif shp == (3,):
            a = np.array([tv[0], tv[1], 1.0], dtype=float)
        elif shp == (1,):
            a = np.array([float(np.linalg.norm(tv))], dtype=float)
        else:
            # Best-effort: pad or trim to match shape size
            size = int(np.prod(shp))
            if tv.size >= size:
                a = tv[:size]
            else:
                a = np.zeros(size, dtype=float)
                a[:tv.size] = tv

        # Rough normalization for small ranges (e.g., [-1, 1])
        if hasattr(space, "high") and hasattr(space, "low"):
            try:
                hi = np.asarray(space.high, dtype=float)
                if np.all(np.isfinite(hi)) and np.all(np.abs(hi) <= 1.5):
                    max_abs = np.max(np.abs(a)) if a.size > 0 else 1.0
                    if max_abs > 1e-9:
                        a = a / max_abs
            except Exception:
                pass

        # Final clipping
        if hasattr(space, "low") and hasattr(space, "high"):
            try:
                a = np.clip(a, space.low, space.high)
            except Exception:
                pass
        try:
            a = a.reshape(space.shape)
        except Exception:
            pass
    else:
        a = tv

    return a


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent  # project root

for _p in [_REPO_ROOT, _REPO_ROOT / "envs", _REPO_ROOT / "controller"]:
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)


# -----------------------------------------------------------------------------
# Position / radius helpers
# -----------------------------------------------------------------------------
def extract_pos_from_any(obs, info, env):
    """
    Return position vector [x, y] from obs/info/env.
    Priority:
      1) info['pos'] or info['position'] (length >= 2)
      2) info['x'] and info['y']
      3) env.x and env.y
      4) obs[0:2]
    """
    # info vector
    for k in ["pos", "position"]:
        v = info.get(k, None)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return np.array([float(v[0]), float(v[1])], dtype=float)
    # info scalars
    if all(k in info for k in ["x", "y"]):
        return np.array([float(info["x"]), float(info["y"])], dtype=float)
    # env attributes
    if all(hasattr(env, k) for k in ["x", "y"]):
        return np.array([float(getattr(env, "x")), float(getattr(env, "y"))], dtype=float)
    # obs fallback
    if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 2:
        arr = np.asarray(obs, dtype=float)
        return np.array([arr[0], arr[1]], dtype=float)
    raise RuntimeError("Unable to recover position from obs/info/env.")


def current_radius_from_state(obs, info, env) -> float:
    """Compute current orbital radius from position vector."""
    pos = extract_pos_from_any(obs, info, env)
    return float(np.linalg.norm(pos))


def try_preset_radius_fallback(preset_name: str):
    """
    Try to read an initial radius from envs.orbit_presets.PRESET_MAP if available.
    Best-effort only; absence is fine.
    """
    try:
        from envs.orbit_presets import PRESET_MAP
        preset = PRESET_MAP.get(preset_name, None)
        if preset is None:
            return None
        for k in ["r0", "radius0", "initial_radius", "target_radius", "r_init"]:
            if hasattr(preset, k):
                return float(getattr(preset, k))
            if isinstance(preset, dict) and k in preset:
                return float(preset[k])
        return None
    except Exception:
        return None


def resolve_target_radius_for_scene(env, obs, info, scene: str, reset_opts: dict, preset_name: str | None) -> float:
    """
    Decide a target_radius without relying on the env to expose it.
    Policy:
      - circular / elliptic: use current radius (position stays on this orbit; velocity is perturbed)
      - transfer: if reset_opts has target_radius_multiplier -> r_now * multiplier;
                  else try preset; else fallback to r_now
    """
    # Prefer an explicit attribute if present
    for name in ["target_radius", "r_target", "target_r", "desired_radius"]:
        if hasattr(env, name):
            val = getattr(env, name)
            if val is not None:
                return float(val)

    r_now = current_radius_from_state(obs, info, env)
    scene_l = str(scene).lower()

    if scene_l in ["circular", "elliptic"]:
        return r_now

    if scene_l == "transfer":
        mult = reset_opts.get("target_radius_multiplier", None) if isinstance(reset_opts, dict) else None
        if mult is not None:
            try:
                return float(r_now) * float(mult)
            except Exception:
                pass
        r_preset = try_preset_radius_fallback(preset_name)
        if isinstance(r_preset, (int, float)) and r_preset > 0:
            return float(r_preset)
        return r_now

    return r_now


# -----------------------------------------------------------------------------
# Small filesystem helper
# -----------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Orbit error extraction
# -----------------------------------------------------------------------------
def pick_from_info(info: dict, keys, default=None):
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return default


def compute_orbit_error_fallback(info, env, default=None):
    """
    Try to get an orbit error scalar from info/env:
      - Prefer info keys: 'orbit_error', 'radius_error', 'r_err'
      - Fallback: if env has 'target_radius' and info has current 'r'/'radius',
        compute relative error |r - r_target| / r_target.
    """
    val = pick_from_info(info, ["orbit_error", "radius_error", "r_err"], None)
    if val is not None:
        try:
            return float(val)
        except Exception:
            return default

    r = pick_from_info(info, ["radius", "r"], None)
    r_target = getattr(env, "target_radius", None)
    if r is not None and r_target:
        try:
            return abs(float(r) - float(r_target)) / float(r_target)
        except Exception:
            return default

    return default


# -----------------------------------------------------------------------------
# Scenario reset options
# -----------------------------------------------------------------------------
def resolve_reset_kwargs(scene_name: str, init_speed_scale: float = 1.0, target_radius_multiplier: float | None = None):
    """
    Compose an options dict for env.reset(...).
    The env may ignore unused fields; that's fine.
    """
    options = {
        "scenario": scene_name,
        "init_speed_scale": init_speed_scale,
    }
    if target_radius_multiplier is not None:
        options["target_radius_multiplier"] = target_radius_multiplier
    return options


# -----------------------------------------------------------------------------
# Robust package-only import
# -----------------------------------------------------------------------------
def try_import():
    """
    Import MultiOrbitEnv and ExpertController via package names so that
    relative imports inside those modules (e.g., 'from .orbit_presets') work.
    """
    import importlib

    # Ensure 'envs' and 'controller' are packages
    for pkg in ["envs", "controller"]:
        pkg_init = _REPO_ROOT / pkg / "__init__.py"
        if not pkg_init.exists():
            pkg_init.parent.mkdir(parents=True, exist_ok=True)
            pkg_init.write_text("", encoding="utf-8")

    env_mod_candidates = ["envs.multi_orbit_env", "envs.multi_orbit"]
    ctrl_mod_candidates = [
        "controller.expert_controller",
        "controller.combined_controller",
        "controller.multi_orbit_expert_controller",
    ]

    last_err = None
    EnvClass, CtrlClass = None, None

    # Env
    for name in env_mod_candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "MultiOrbitEnv"):
                EnvClass = getattr(mod, "MultiOrbitEnv")
                break
        except Exception as e:
            last_err = e
    if EnvClass is None:
        print("[IMPORT-DEBUG] sys.path head:", sys.path[:5])
        print("[IMPORT-DEBUG] repo_root:", _REPO_ROOT)
        print("[IMPORT-DEBUG] envs exists?:", (_REPO_ROOT / "envs").exists(),
              "controller exists?:", (_REPO_ROOT / "controller").exists())
        raise ImportError(f"Unable to import MultiOrbitEnv. Last error: {last_err}")

    # Controller
    for name in ctrl_mod_candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "ExpertController"):
                CtrlClass = getattr(mod, "ExpertController")
                break
            if hasattr(mod, "ExpertControllerV3"):
                CtrlClass = getattr(mod, "ExpertControllerV3")
                break
        except Exception as e:
            last_err = e
    if CtrlClass is None:
        print("[IMPORT-DEBUG] Tried controller modules:", ctrl_mod_candidates)
        raise ImportError(f"Unable to import ExpertController. Last error: {last_err}")

    return EnvClass, CtrlClass


# -----------------------------------------------------------------------------
# Expert instantiation helpers
# -----------------------------------------------------------------------------
def extract_controller_kwargs_from_env(env, target_radius_value: float) -> dict:
    """
    Build kwargs for ExpertController using a provided target_radius_value.
    Also forward optional physical constants if exposed by the env.
    """
    kw = {"target_radius": float(target_radius_value)}
    for src_attr in ["G", "M", "mass"]:
        if hasattr(env, src_attr):
            kw[src_attr] = getattr(env, src_attr)
    return kw


def instantiate_with_supported_kwargs(cls, kwargs: dict):
    """
    Instantiate a class using only kwargs accepted by its __init__ signature.
    Prevents unexpected-kw errors without modifying the class itself.
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


# -----------------------------------------------------------------------------
# (obs, info, env) -> (pos, vel) mapping for Expert.__call__(t, pos, vel)
# -----------------------------------------------------------------------------
def extract_pos_vel_from_obs_info(obs, info, env):
    """
    Recover position and velocity vectors from obs/info/env.
    Returns (pos_vec, vel_vec) as numpy arrays of length 2.
    Priority:
      1) info['pos'/'position'] and info['vel'/'velocity']
      2) info scalars x,y and vx,vy
      3) env attributes x,y and vx,vy
      4) obs assumed as [x, y, vx, vy]
    """
    pos = None
    vel = None

    for k in ["pos", "position"]:
        v = info.get(k, None)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            pos = np.array(v[:2], dtype=float)
            break
    for k in ["vel", "velocity"]:
        v = info.get(k, None)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            vel = np.array(v[:2], dtype=float)
            break

    if pos is None and all(s in info for s in ["x", "y"]):
        pos = np.array([info["x"], info["y"]], dtype=float)
    if vel is None and all(s in info for s in ["vx", "vy"]):
        vel = np.array([info["vx"], info["vy"]], dtype=float)

    if pos is None and all(hasattr(env, s) for s in ["x", "y"]):
        pos = np.array([getattr(env, "x"), getattr(env, "y")], dtype=float)
    if vel is None and all(hasattr(env, s) for s in ["vx", "vy"]):
        vel = np.array([getattr(env, "vx"), getattr(env, "vy")], dtype=float)

    if (pos is None or vel is None) and isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 4:
        arr = np.array(obs, dtype=float)
        if pos is None:
            pos = np.array([arr[0], arr[1]], dtype=float)
        if vel is None:
            vel = np.array([arr[2], arr[3]], dtype=float)

    if pos is None or vel is None:
        raise RuntimeError("Unable to extract (pos, vel). Please expose x,y,vx,vy or make obs=[x,y,vx,vy].")

    return pos, vel


def expert_action_from_state(expert, step_idx, env, obs, info, accel_mode=True):
    """
    Call ExpertController which expects (t, pos, vel) -> thrust_vec (2,).
    Time t is approximated as step_idx * env.dt if available, otherwise 0.0.
    accel_mode=True will reinterpret Expert output as acceleration by dividing by env.mass.
    """
    t = float(step_idx) * float(getattr(env, "dt", 0.0))
    pos, vel = extract_pos_vel_from_obs_info(obs, info, env)
    thrust = expert(t, pos, vel)
    thrust = np.asarray(thrust, dtype=float).reshape(-1)
    if accel_mode and hasattr(env, "mass") and env.mass:
        thrust = thrust / env.mass
    return thrust


# -----------------------------------------------------------------------------
# Optional adaptive thrust scaling (kept but not enforced)
# -----------------------------------------------------------------------------
THRUST_ALPHA = 1e-4  # target dv per step is alpha * v_circ (safe & small)
MAX_GAIN = 1e8
MIN_GAIN = 1.0

def estimate_v_circular_from_expert(expert, target_radius_fallback: float) -> float:
    """
    Estimate circular speed using Expert's physical params if available:
      v_c = sqrt(G * M / r)
    """
    G = getattr(expert, "G", 6.67430e-11)
    M = getattr(expert, "M", 1.989e30)
    r = getattr(expert, "target_radius", target_radius_fallback) or target_radius_fallback
    return float(np.sqrt(G * M / float(r)))


def calibrate_thrust_gain(env, expert, raw_action, dt: float) -> float:
    """
    Compute a scale factor 'gain' so that dv_per_step ~ THRUST_ALPHA * v_circ.
    Interprets Expert output as FORCE: dv = (F/m) * dt.
    """
    a0 = float(np.linalg.norm(raw_action))
    if a0 < 1e-12:
        return 1.0

    mass = float(getattr(expert, "mass", getattr(env, "mass", 1.0)))
    v_circ = estimate_v_circular_from_expert(expert, target_radius_fallback=1.0)
    target_dv = THRUST_ALPHA * v_circ

    denom = (a0 / mass) * max(dt, 1e-9)
    if denom <= 0:
        return 1.0

    gain = float(target_dv / denom)
    return max(MIN_GAIN, min(gain, MAX_GAIN))


def apply_thrust_gain(vec, gain: float):
    """Scale a 2D vector by 'gain'."""
    return np.asarray(vec, dtype=float).reshape(-1) * float(gain)


# -----------------------------------------------------------------------------
# Env target extraction
# -----------------------------------------------------------------------------
def get_env_target_radius(env, info, fallback):
    """
    Extract the environment's notion of target_radius if possible.
    Priority:
      1) info['target_radius'] or info['r_target']
      2) env.target_radius attribute
      3) fallback provided by caller
    """
    if "target_radius" in info:
        return float(info["target_radius"])
    if "r_target" in info:
        return float(info["r_target"])
    if hasattr(env, "target_radius") and env.target_radius is not None:
        return float(env.target_radius)
    return float(fallback)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(args):
    np.random.seed(args.seed)

    # Import environment and ExpertController
    MultiOrbitEnv, ExpertController = try_import()

    # Output paths
    out_dir = Path("logs/day49")
    ensure_dir(out_dir)
    csv_path = out_dir / "test_log_day49.csv"
    write_header = not csv_path.exists()

    # Scenarios
    scenarios = [
        {"scene": "circular", "reset_opts": resolve_reset_kwargs("circular", init_speed_scale=1.0)},
        {"scene": "elliptic", "reset_opts": resolve_reset_kwargs("elliptic", init_speed_scale=0.9)},
        {"scene": "transfer", "reset_opts": resolve_reset_kwargs("transfer", init_speed_scale=1.0, target_radius_multiplier=1.5)},
    ]

    # CSV columns
    fieldnames = [
        "datetime", "scene", "controller", "preset", "seed", "steps",
        "total_reward", "orbit_error_mean", "orbit_error_final", "success", "notes",
    ]

    rows = []
    for sc in scenarios:
        scene = sc["scene"]
        reset_opts = sc["reset_opts"]

        # 1) Create environment
        try:
            env = MultiOrbitEnv(preset=args.preset)
        except TypeError:
            env = MultiOrbitEnv()

        # 2) Reset environment
        try:
            obs, info = env.reset(seed=args.seed, options=reset_opts)
        except TypeError:
            obs = env.reset()
            info = {}

        # 3) Resolve target radius for ExpertController
        fallback_val = resolve_target_radius_for_scene(
            env=env, obs=obs, info=info, scene=scene, reset_opts=reset_opts, preset_name=args.preset
        )
        target_radius_val = get_env_target_radius(env, info, fallback=fallback_val)

        if DEBUG_D49:
            print(f"[DEBUG] scene={scene} env_target={getattr(env, 'target_radius', None)} "
                  f"expert_target={target_radius_val}")

        # 4) Instantiate ExpertController
        ctrl_kwargs = extract_controller_kwargs_from_env(env, target_radius_val)
        expert = instantiate_with_supported_kwargs(ExpertController, ctrl_kwargs)

        # Optional reset hook on Expert
        if hasattr(expert, "reset"):
            try:
                expert.reset(env=env, scene=scene)
            except TypeError:
                expert.reset()

        total_reward = 0.0
        orbit_err_list = []
        steps = 0
        terminated = False
        truncated = False

        # 5) Rollout
        for t in range(args.steps):
            # Query Expert for action (acceleration mode enabled)
            raw_action = expert_action_from_state(expert, t, env, obs, info, accel_mode=False)
            action = adapt_action_to_env(raw_action, env)

            if DEBUG_D49 and t < 5:
                print(f"[DEBUG] raw_action={raw_action} -> adapted_action={action}")

            # Step environment (Gymnasium/Gym compatibility)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError:
                obs, reward, done, info = env.step(action)
                terminated, truncated = done, False

            total_reward += float(reward)
            steps += 1

            # Track orbit error if available
            oe = compute_orbit_error_fallback(info, env, default=None)
            if oe is not None:
                orbit_err_list.append(float(oe))

            if terminated or truncated:
                break

        # 6) Metrics
        orbit_err_mean = float(np.mean(orbit_err_list)) if orbit_err_list else np.nan
        orbit_err_final = float(orbit_err_list[-1]) if orbit_err_list else np.nan
        thr = 0.01 if scene == "circular" else 0.05
        success = "" if np.isnan(orbit_err_final) else ("Y" if orbit_err_final <= thr else "N")

        row = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "scene": scene,
            "controller": "ExpertController_v3.1",
            "preset": args.preset,
            "seed": args.seed,
            "steps": steps,
            "total_reward": round(total_reward, 6),
            "orbit_error_mean": round(orbit_err_mean, 8) if not np.isnan(orbit_err_mean) else "",
            "orbit_error_final": round(orbit_err_final, 8) if not np.isnan(orbit_err_final) else "",
            "success": success,
            "notes": "",
        }
        rows.append(row)

        print(f"[Day49] {scene:9s} | reward={row['total_reward']:.4f} | "
              f"err_mean={row['orbit_error_mean']} | err_final={row['orbit_error_final']} | steps={steps}")

        time.sleep(0.2)

    # 7) Save CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n Saved Day49 Expert results to: {csv_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Max steps per scenario")
    parser.add_argument("--seed", type=int, default=49, help="Random seed")
    parser.add_argument("--preset", type=str, default="voyager1", help="Orbit preset name if your env supports it")
    args = parser.parse_args()
    run(args)
