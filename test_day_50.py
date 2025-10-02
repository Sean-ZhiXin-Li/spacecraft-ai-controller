import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

# Controller import
try:
    # Adjust this import path if your controller lives elsewhere
    from controller.expert_controller import ExpertController  # v3.1 controller
except Exception as e:
    raise ImportError(
        "Cannot import ExpertController. Please adjust the import path in test_day_50.py."
    ) from e

# resolve_reset_kwargs (fallback)

try:
    from envs.orbit_presets import resolve_reset_kwargs
except Exception:
    def resolve_reset_kwargs(scene_name: str, **kwargs):
        """Fallback: pass-through with a scene hint; env/controller can ignore safely."""
        dd = dict(kwargs)
        dd["scene_hint"] = scene_name
        return dd

# Environment import

try:
    from envs.multi_orbit_env import MultiOrbitEnv
except Exception as e:
    raise ImportError(
        "Cannot import MultiOrbitEnv from envs.multi_orbit_env. "
        "Please adjust the import path near the top of test_day_50.py."
    ) from e


# Utilities

def set_global_seed(seed: int):
    """Make runs deterministic-ish."""
    random.seed(seed)
    np.random.seed(seed)


def build_env(scene: str, reset_opts: dict, preset: str):
    """
    Construct MultiOrbitEnv without passing unsupported kwargs, and stash reset options.
    Only pass 'scene' if supported; otherwise construct with no args and set attribute.
    Merge 'preset' into default_reset_options so reset(options=...) can use it.
    """
    full_opts = dict(reset_opts or {})
    full_opts.setdefault("preset", preset)

    try:
        env = MultiOrbitEnv(scene=scene)
    except TypeError:
        env = MultiOrbitEnv()
        setattr(env, "scene", scene)

    setattr(env, "default_reset_options", full_opts)
    return env


def safe_reset(env):
    """
    Call env.reset(); pass options if supported (Gymnasium-style).
    Also tolerate legacy Gym reset() that returns only obs.
    Returns (obs, info).
    """
    options = getattr(env, "default_reset_options", None)
    try:
        if options is not None:
            res = env.reset(options=options)
        else:
            res = env.reset()
    except TypeError:
        res = env.reset()

    if isinstance(res, tuple):
        if len(res) == 2:
            obs, info = res
        else:
            obs, info = res[0], {}
    else:
        obs, info = res, {}

    return obs, (info if isinstance(info, dict) else {})


def _get_nested_attr(obj, names):
    """
    Try a list of candidate attribute names or dotted paths and return the first found.
    Example names: ["target_radius", "goal_radius", "orbit_target.radius"]
    """
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    return None


def discover_target_radius(env, obs, info, reset_opts):
    """
    Best-effort discovery of target radius from info/env/opts.
    Priority:
      1) info dict keys
      2) env attributes (including nested)
      3) derive from reset_opts if possible
      4) fallback to 1.0 (warn)
    """
    # 1) Try info keys
    info_keys = [
        "target_radius", "goal_radius", "target_r", "expert_target",
        "env_target", "target"
    ]
    for k in info_keys:
        if isinstance(info, dict) and (k in info):
            try:
                val = float(info[k])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass

    # 2) Try env attributes (including nested)
    attr_candidates = [
        "target_radius", "goal_radius", "target_r",
        "expert_target", "env_target", "target",
        "orbit_target.radius", "target.radius", "goal.radius"
    ]
    val = _get_nested_attr(env, attr_candidates)
    try:
        if val is not None:
            val = float(val)
            if np.isfinite(val) and val > 0:
                return val
    except Exception:
        pass

    # 3) Try deriving from reset options if there is enough info
    mul = None
    try:
        mul = float((reset_opts or {}).get("target_radius_multiplier", None))
    except Exception:
        mul = None

    if mul and mul > 0:
        base_r_candidates = [
            "init_radius", "orbit_radius", "r0", "radius0",
            "state_radius", "current_radius"
        ]
        base_r = _get_nested_attr(env, base_r_candidates)
        try:
            if base_r is not None:
                base_r = float(base_r)
                if np.isfinite(base_r) and base_r > 0:
                    return base_r * mul
        except Exception:
            pass

    # 4) Fallback
    print("[WARN] target_radius not found; fallback to 1.0 (adjust discover_target_radius if needed).")
    return 1.0


def evaluate_failure(final_err: float,
                     terminated: bool,
                     truncated: bool,
                     steps_taken: int,
                     max_steps: int,
                     episode_return: float,
                     err_threshold: float,
                     unstable_reward_threshold: float):
    """
    Decide failure_reason string ('' means success).
    Priority:
    1) Large final error
    2) Did not converge (timeout)
    3) Unstable trajectory (very low return)
    """
    if np.isnan(final_err):
        return "Missing error metric"
    if final_err > err_threshold:
        return "Large final error"
    if not terminated and steps_taken >= max_steps:
        return "Did not converge"
    if episode_return < unstable_reward_threshold:
        return "Unstable trajectory"
    return ""  # success


def extract_error(info: dict, default=np.nan):
    """
    Try to read orbit/goal error from info dict.
    Adjust keys to match your env's info contract if needed.
    """
    if not isinstance(info, dict):
        return default
    for k in ("orbit_error", "goal_error", "radius_err", "normalized_orbit_error"):
        if k in info:
            try:
                return float(info[k])
            except Exception:
                continue
    return default


def split_pos_vel_from_obs(obs, info):
    """
    Best-effort extraction of (pos, vel) from either info or obs.
    Priority:
      1) info contains 'pos' & 'vel' or 'position' & 'velocity'
      2) obs is a dict with x,y,vx,vy
      3) obs is array-like: 3D [x,y,z, vx,vy,vz], 2D [x,y, vx,vy]
      4) generic even-length split: first half pos, second half vel
    """
    # 1) info vectors
    if isinstance(info, dict):
        for pkey, vkey in (("pos", "vel"), ("position", "velocity")):
            if pkey in info and vkey in info:
                pos = np.asarray(info[pkey], dtype=float).ravel()
                vel = np.asarray(info[vkey], dtype=float).ravel()
                return pos, vel

    # 2) obs dict components
    if isinstance(obs, dict) and all(k in obs for k in ("x", "y", "vx", "vy")):
        pos = np.array([obs["x"], obs["y"]], dtype=float)
        vel = np.array([obs["vx"], obs["vy"]], dtype=float)
        return pos, vel

    # 3) array-like
    arr = np.asarray(obs, dtype=float).ravel()
    n = arr.size
    if n >= 6:
        return arr[:3], arr[3:6]   # assume 3D
    if n >= 4:
        return arr[:2], arr[2:4]   # assume 2D

    # 4) generic even split
    if n > 0 and n % 2 == 0:
        half = n // 2
        return arr[:half], arr[half:]

    # cannot infer
    return None, None


def controller_action(controller, obs, info=None):
    """
    Robustly obtain an action from the controller.
    Strategy:
      1) Prefer (pos, vel) call for methods likely to be physics-based (e.g., __call__, step, action)
      2) Fallback to (obs) calls
      3) SB3-like predict kept obs-only
    Methods tried in order:
      - __call__, step, action, compute_action, act, get_action, predict
    """
    methods = []
    for name in ["__call__", "step", "action", "compute_action", "act", "get_action", "predict"]:
        m = getattr(controller, name, None)
        if callable(m):
            methods.append((name, m))

    # Precompute pos/vel once
    pos, vel = split_pos_vel_from_obs(obs, info)
    have_pv = (pos is not None) and (vel is not None)

    last_err = None

    for name, m in methods:
        try:
            # 1) Try pos/vel forms first for non-predict methods
            if name != "predict" and have_pv:
                try:
                    return m(pos, vel, info=info)
                except TypeError:
                    return m(pos, vel)
            # 2) Fallback to obs forms
            if name == "predict":
                try:
                    out = m(obs, deterministic=True)
                except TypeError:
                    out = m(obs)
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    return out[0]
                return out
            else:
                try:
                    return m(obs, info=info)
                except TypeError:
                    return m(obs)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to obtain action from controller with any known signature. Last error: {last_err}"
    )

# ---- helpers: must appear before VelocityBackfillAdapter ----
def extract_pos_only(obs, info):
    if isinstance(info, dict):
        for key in ("pos", "position"):
            if key in info:
                return np.asarray(info[key], dtype=float).ravel()
    if isinstance(obs, dict) and all(k in obs for k in ("x", "y")):
        if "z" in obs:
            return np.array([obs["x"], obs["y"], obs["z"]], dtype=float)
        return np.array([obs["x"], obs["y"]], dtype=float)
    try:
        arr = np.asarray(obs, dtype=float).ravel()
        if arr.size >= 6:
            return arr[:3]
        if arr.size >= 4:
            return arr[:2]
        if arr.size >= 2:
            return arr[:2]
    except Exception:
        pass
    return None

def estimate_dt(env, info):
    if isinstance(info, dict):
        for k in ("dt", "delta_t", "time_step"):
            if k in info:
                try:
                    v = float(info[k])
                    if np.isfinite(v) and v > 0:
                        return v
                except Exception:
                    pass
    for name in ("dt", "delta_t", "time_step", "config.dt"):
        v = _get_nested_attr(env, [name])
        if v is not None:
            try:
                v = float(v)
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return 1.0


class VelocityBackfillAdapter:
    """
    Wrap a controller that expects (pos, vel). If vel is absent, estimate via finite difference.
    Call-order strategy once (pos, vel) are available:
      1) controller(pos, vel)
      2) controller((pos, vel))
      3) controller(np.concatenate([pos, vel]))
      4) controller({"pos": pos, "vel": vel})
    If the object is not directly callable, try named methods with the same patterns, then obs fallbacks.
    """
    def __init__(self, controller, dt: float):
        self.controller = controller
        self.dt = float(dt) if (dt and np.isfinite(dt)) else 1.0
        self.prev_pos = None

        # Named methods fallback (when object isn't directly callable)
        self.named_methods = []
        for name in ["step", "action", "compute_action", "act", "get_action", "predict"]:
            m = getattr(controller, name, None)
            if callable(m):
                self.named_methods.append((name, m))

    # --- core dispatcher using (pos, vel) ---
    def _try_call_patterns(self, call_fn, pos, vel, info):
        """
        call_fn is a callable that receives exactly one argument:
          - either a pair of args (pos, vel), or a single packed arg,
        implemented here as small lambdas to try different shapes.
        """
        last_err = None
        patterns = [
            ("call_pv",      lambda: call_fn(("pv", pos, vel))),  # signal to unpack into two args
            ("call_tuple",   lambda: call_fn(("one", (pos, vel)))),
            ("call_concat",  lambda: call_fn(("one", np.concatenate([pos, vel])))),
            ("call_dict",    lambda: call_fn(("one", {"pos": pos, "vel": vel}))),
        ]
        for tag, thunk in patterns:
            try:
                return thunk()
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Controller call failed with all pv patterns. Last error: {last_err}")

    def _call_object(self, pos, vel, info):
        """
        Prefer calling the controller object itself if it's callable.
        We avoid passing unknown kwargs (like info) since your __call__ doesn't accept it.
        """
        if callable(self.controller):
            def call_fn(spec):
                mode, payload = spec
                if mode == "pv":
                    _, p, v = spec
                    return self.controller(p, v)
                else:
                    # single-arg call
                    _, one = spec
                    return self.controller(one)
            return self._try_call_patterns(call_fn, pos, vel, info)
        return None  # not callable -> try named methods

    def _call_named_methods(self, pos, vel, info):
        """
        Try named methods with the same pattern list.
        For predict(...), we keep it obs-oriented as a last resort.
        """
        last_err = None
        # 1) Non-predict methods with pv/packed patterns
        for name, m in self.named_methods:
            if name == "predict":
                continue
            def call_fn(spec, _m=m):
                mode, payload = spec
                if mode == "pv":
                    _, p, v = spec
                    try:
                        return _m(p, v, info=info)
                    except TypeError:
                        return _m(p, v)
                else:
                    _, one = spec
                    try:
                        return _m(one, info=info)
                    except TypeError:
                        return _m(one)
            try:
                return self._try_call_patterns(call_fn, pos, vel, info)
            except Exception as e:
                last_err = e
                continue

        # 2) predict(...) as last resort with obs-like vector
        for name, m in self.named_methods:
            if name != "predict":
                continue
            try:
                obs_vec = np.concatenate([pos, vel])
                try:
                    out = m(obs_vec, deterministic=True)
                except TypeError:
                    out = m(obs_vec)
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    return out[0]
                return out
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Controller named-method calls failed. Last error: {last_err}")

    def _call_with_pv(self, pos, vel, info):
        # 1) object call
        try:
            out = self._call_object(pos, vel, info)
            if out is not None:
                return out
        except Exception:
            pass
        # 2) named methods
        return self._call_named_methods(pos, vel, info)

    def action(self, obs, info=None):
        # direct pos/vel from info
        if isinstance(info, dict):
            for pkey, vkey in (("pos", "vel"), ("position", "velocity")):
                if pkey in info and vkey in info:
                    pos = np.asarray(info[pkey], dtype=float).ravel()
                    vel = np.asarray(info[vkey], dtype=float).ravel()
                    return self._call_with_pv(pos, vel, info)

        # dict obs with components
        if isinstance(obs, dict) and all(k in obs for k in ("x", "y", "vx", "vy")):
            pos = np.array([obs["x"], obs["y"]], dtype=float)
            vel = np.array([obs["vx"], obs["vy"]], dtype=float)
            return self._call_with_pv(pos, vel, info)

        # array-like split
        try:
            arr = np.asarray(obs, dtype=float).ravel()
            if arr.size >= 6:
                return self._call_with_pv(arr[:3], arr[3:6], info)
            if arr.size >= 4:
                return self._call_with_pv(arr[:2], arr[2:4], info)
        except Exception:
            pass

        # backfill velocity from position-only
        pos_only = extract_pos_only(obs, info)
        if pos_only is not None:
            if self.prev_pos is None:
                v0 = None
                if isinstance(info, dict):
                    for key in ("vel", "velocity"):
                        if key in info:
                            try:
                                v0 = np.asarray(info[key], dtype=float).ravel()
                                break
                            except Exception:
                                pass
                if v0 is None:
                    v0 = np.zeros_like(pos_only, dtype=float)
                self.prev_pos = pos_only.copy()
                return self._call_with_pv(pos_only, v0, info)
            else:
                vel_est = (pos_only - self.prev_pos) / self.dt
                self.prev_pos = pos_only.copy()
                return self._call_with_pv(pos_only, vel_est, info)

        # nothing worked
        raise RuntimeError("Cannot infer (pos, vel) from obs/info to call controller.")



def rollout(env, controller, max_steps: int, initial=None):
    """
    Run one rollout; return logs per step and episode summary.
    Compatible with Gymnasium (5-return) and legacy Gym (4-return) step API.
    If `initial` is provided, it must be a tuple (obs, info) to start without calling reset again.
    """
    if initial is None:
        obs, info = safe_reset(env)
    else:
        obs, info = initial

    error_series = []
    reward_sum = 0.0
    terminated = False
    truncated = False
    steps = 0

    while steps < max_steps:
        action = controller_action(controller, obs, info=info)

        # Prefer Gymnasium unpack; fallback to legacy Gym
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except ValueError:
            obs, reward, done, info = env.step(action)
            terminated = bool(done)
            truncated = False

        reward_sum += float(reward)
        steps += 1
        error_series.append(extract_error(info))

        if terminated or truncated:
            break

    final_err = error_series[-1] if len(error_series) else np.nan
    return {
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_sum": reward_sum,
        "final_error": final_err,
        "error_series": error_series
    }


def plot_error_curve(error_series, out_path: Path, title: str):
    """Save error curve for quick diagnostics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    x = np.arange(len(error_series))
    y = np.array(error_series, dtype=float)
    plt.plot(x, y)
    plt.xlabel("Step")
    plt.ylabel("Orbit/Goal Error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--preset", type=str, default="voyager1")
    parser.add_argument("--err_threshold", type=float, default=0.1, help="Failure if final_err > threshold")
    parser.add_argument("--unstable_reward_threshold", type=float, default=-50.0,
                        help="Failure if episode return < threshold")
    parser.add_argument("--logdir", type=str, default="logs/day50", help="Output directory")
    parser.add_argument("--target_radius", type=float, default=None,
                        help="Override controller target_radius (if auto-discovery fails or you want a specific value).")
    args = parser.parse_args()

    set_global_seed(args.seed)

    logdir = Path(args.logdir)
    figdir = logdir / "figs"
    logdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    csv_path = logdir / "test_log_day50.csv"
    write_header = not csv_path.exists()

    # Define four scenes, including the new 'transfer_2phase'
    scene_defs = [
        ("circular",
         resolve_reset_kwargs("circular", init_speed_scale=1.0, target_radius_multiplier=1.0)),
        ("elliptic",
         resolve_reset_kwargs("elliptic", init_speed_scale=1.0, target_radius_multiplier=1.5)),
        ("transfer",
         resolve_reset_kwargs("transfer", init_speed_scale=1.0, target_radius_multiplier=2.0)),
        ("transfer_2phase",
         resolve_reset_kwargs(
             "transfer",
             init_speed_scale=1.0,
             target_radius_multiplier=2.0,
             two_phase=True,
             phase1="raise_apogee",
             phase2="circularize",
         )),
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts", "preset", "scene", "seed",
                "max_steps", "steps", "terminated", "truncated",
                "reward_sum", "orbit_error_final",
                "failure_reason", "controller_target_radius"
            ],
        )
        if write_header:
            writer.writeheader()

        for scene, reset_opts in scene_defs:
            # 1) Build env and do a single reset to fetch info for target discovery
            env = build_env(scene=scene, reset_opts=reset_opts, preset=args.preset)
            obs0, info0 = safe_reset(env)

            # 2) Determine target_radius
            if args.target_radius is not None and np.isfinite(args.target_radius) and args.target_radius > 0:
                target_r = float(args.target_radius)
            else:
                target_r = discover_target_radius(env, obs0, info0, reset_opts)

            # 3) Construct controller (only passing target_radius, as required by your class)
            controller = ExpertController(target_radius=target_r)

            # 4) Rollout starting from the already-reset state
            ep = rollout(env, controller, args.steps, initial=(obs0, info0))

            # 5) Classify failure
            failure_reason = evaluate_failure(
                final_err=ep["final_error"],
                terminated=ep["terminated"],
                truncated=ep["truncated"],
                steps_taken=ep["steps"],
                max_steps=args.steps,
                episode_return=ep["reward_sum"],
                err_threshold=args.err_threshold,
                unstable_reward_threshold=args.unstable_reward_threshold
            )

            # 6) Save error curve if failed
            if failure_reason:
                fig_name = f"{scene}_seed{args.seed}_err.png"
                plot_error_curve(
                    ep["error_series"],
                    figdir / fig_name,
                    title=f"{scene} (seed={args.seed}) – {failure_reason}"
                )

            # 7) Write CSV row
            writer.writerow({
                "ts": int(time.time()),
                "preset": args.preset,
                "scene": scene,
                "seed": args.seed,
                "max_steps": args.steps,
                "steps": ep["steps"],
                "terminated": int(ep["terminated"]),
                "truncated": int(ep["truncated"]),
                "reward_sum": round(ep["reward_sum"], 6),
                "orbit_error_final": (np.nan if np.isnan(ep["final_error"]) else round(ep["final_error"], 8)),
                "failure_reason": failure_reason,
                "controller_target_radius": (np.nan if np.isnan(target_r) else round(float(target_r), 6))
            })

            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
