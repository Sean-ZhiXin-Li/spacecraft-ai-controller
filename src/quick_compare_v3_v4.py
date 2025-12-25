import os
import sys
import numpy as np
import inspect

# Make project root importable so we can import envs/, controller/, utils/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.action_interface import thrust_to_action, ActionInterfaceConfig

from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController as ExpertV3
from controller.expert_controller_improved import ExpertController as ExpertV4

import yaml

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_ai_mode(cfg: dict) -> str:
    return (cfg.get("action_interface", {}) or {}).get("mode", "raw")

def get_thrust_scale(cfg: dict, env: OrbitEnv) -> float:
    # config wins; fallback to env default
    return float((cfg.get("actuation", {}) or {}).get("thrust_scale", getattr(env, "thrust_scale", 1.0)))

def build_controller(ControllerCls, env, obs):
    """
    Build controller using env.target_radius if available.
    """
    tr = getattr(env, "target_radius", None)
    if tr is None:
        x = np.asarray(obs, dtype=float)
        n = x.size // 2
        pos0 = x[:n]
        tr = float(np.linalg.norm(pos0))
    return ControllerCls(target_radius=float(tr))


def make_env_for_scenario(scenario_name=None, cfg=None):
    """
    Emulate scenarios via:
    - reset(start_mode=...)
    - set_physical_params(...)
    """
    env = OrbitEnv()
    if cfg is not None:
        env.thrust_scale = get_thrust_scale(cfg, env)

    start_mode = "default"
    physical_kwargs = {}

    if scenario_name is None or scenario_name == "default":
        start_mode = "default"
    elif scenario_name == "spiral":
        start_mode = "spiral"
    elif scenario_name == "weak_thrust_far":
        physical_kwargs["thrust_newton"] = 800.0
    elif scenario_name == "misaligned_entry":
        start_mode = "default"
    elif scenario_name == "oscillation_noise":
        print("[SELF-CHECK][WARN] 'oscillation_noise' not supported by OrbitEnv (no noise API). Using default.")
        start_mode = "default"
    else:
        print(f"[SELF-CHECK][WARN] Unknown scenario '{scenario_name}'. Using default.")
        start_mode = "default"

    if physical_kwargs and hasattr(env, "set_physical_params"):
        env.set_physical_params(**physical_kwargs)

    env._qc_start_mode = start_mode
    env._qc_physical_kwargs = physical_kwargs
    return env


def clip_action_to_env(action, env):
    # NOTE: kept for legacy quick-check; CM Day2 uses action_interface instead.
    a = np.asarray(action, dtype=float)

    lo, hi = -1.0, 1.0
    if hasattr(env, "action_space") and env.action_space is not None:
        try:
            lo = float(np.min(env.action_space.low))
            hi = float(np.max(env.action_space.high))
        except Exception:
            pass

    a_clip = np.clip(a, lo, hi)

    overflow = float(np.mean((a < lo) | (a > hi)))
    return a_clip, lo, hi


def _obs_signature(obs, k=8):
    x = np.asarray(obs, dtype=float).ravel()
    return x[: min(k, x.size)]


def _unpack_step(step_out):
    """
    Gym (obs, reward, done, info) or Gymnasium (obs, reward, terminated, truncated, info)
    """
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    else:
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), info


def _try_shadow_step(env, action, atol_obs=1e-9, atol_reward=1e-9):
    """
    Compare step(action) vs step(0) from the exact same state using get_state/set_state.

    Restores env to the original state before returning.
    """
    has_get = hasattr(env, "get_state") and callable(getattr(env, "get_state"))
    has_set = hasattr(env, "set_state") and callable(getattr(env, "set_state"))

    if not (has_get and has_set):
        print("[SELF-CHECK][WARN] shadow_step skipped: env has no get_state/set_state API.")
        return

    saved = env.get_state()
    if saved is None:
        print("[SELF-CHECK][WARN] shadow_step skipped: env.get_state() returned None.")
        return

    a = np.asarray(action, dtype=np.float64).copy()
    z = np.zeros_like(a)

    try:
        out_a = env.step(a)
        obs_a, rew_a, done_a, info_a = _unpack_step(out_a)

        env.set_state(saved)
        out_z = env.step(z)
        obs_z, rew_z, done_z, info_z = _unpack_step(out_z)

        obs_a = np.asarray(obs_a, dtype=np.float64).ravel()
        obs_z = np.asarray(obs_z, dtype=np.float64).ravel()
        dr = float(np.linalg.norm(obs_a - obs_z))

        d_reward = float(abs(rew_a - rew_z))
        d_done = (done_a != done_z)

        print(f"[SELF-CHECK] shadow_step | ||obs(a)-obs(0)||={dr:.6e} | |rew(a)-rew(0)|={d_reward:.6e} | done_diff={d_done}")

        if dr < atol_obs and d_reward < atol_reward and (not d_done):
            print("[SELF-CHECK][WARN] shadow_step diffs are ~0 (obs/reward/done). "
                  "Strong evidence action is not affecting dynamics/reward at this state.")

        for k in ["thrust_vec", "acc_thrust", "acc_total"]:
            if isinstance(info_a, dict) and isinstance(info_z, dict) and (k in info_a or k in info_z):
                print(f"[SELF-CHECK] info[{k}] a={info_a.get(k, None)} | 0={info_z.get(k, None)}")

    finally:
        env.set_state(saved)


def run_episode(scenario, label, ControllerCls, cfg, max_steps=2000):
    env = make_env_for_scenario(scenario, cfg=cfg)

    start_mode = getattr(env, "_qc_start_mode", "default")

    reset_result = env.reset(start_mode=start_mode)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs, info = reset_result, {}

    controller = build_controller(ControllerCls, env, obs)

    print(f"[SELF-CHECK] {label} ControllerCls={ControllerCls} | file={inspect.getsourcefile(ControllerCls)}")

    target_radius = float(getattr(env, "target_radius", 0.0))
    if target_radius <= 0.0:
        x0 = np.asarray(obs, dtype=float)
        n = x0.size // 2
        pos0 = x0[:n]
        target_radius = float(np.linalg.norm(pos0))

    phys = getattr(env, "_qc_physical_kwargs", {})
    print(f"[SELF-CHECK] scenario={scenario or 'default'} | start_mode={start_mode} | phys={phys} | target_r={target_radius:.6e}")

    ctrl_tr = getattr(controller, "target_radius", None)
    if ctrl_tr is not None:
        try:
            ctrl_tr = float(ctrl_tr)
            rel = abs(ctrl_tr - target_radius) / max(1e-12, abs(target_radius))
            if rel > 1e-6:
                print(f"[SELF-CHECK][WARN] controller.target_radius != env.target_radius "
                      f"({ctrl_tr:.6e} vs {target_radius:.6e}, rel_diff={rel:.3e})")
        except Exception:
            pass

    total_reward = 0.0
    final_radius = None
    radius_errors = []
    jitter_terms = []
    prev_action_dir = None

    raw_norms = []   # ||thrust_intent|| (N)
    clip_norms = []  # ||action|| (dimensionless)
    sat_rates = []  # saturation_rate from action_interface (single source of truth)

    do_shadow_step = True
    steps = 0

    for _ in range(max_steps):
        # 1) Controller outputs thrust intent in Newtons
        thrust_intent = controller.act(obs)

        # 2) Day2: Action interface is the independent variable
        cfg_ai = ActionInterfaceConfig(mode=get_ai_mode(cfg))
        action, ainfo = thrust_to_action(thrust_intent, env.thrust_scale, cfg_ai)
        sat_rates.append(float(ainfo.get("saturation_rate", 0.0)))

        # 3) Stats
        ra = np.asarray(thrust_intent, dtype=float).ravel()
        aa_print = np.asarray(action, dtype=float).ravel()

        raw_norms.append(float(np.linalg.norm(ra)))
        clip_norms.append(float(np.linalg.norm(aa_print)))

        if do_shadow_step:
            do_shadow_step = False
            _try_shadow_step(env, action)

        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        # Day2: attach interface metrics to info
        if isinstance(info, dict):
            info.update(ainfo)

        total_reward += float(reward)

        if steps < 5:
            ts = info.get("thrust_scale", None) if isinstance(info, dict) else None
            m = info.get("mass", None) if isinstance(info, dict) else None
            dt = info.get("dt", None) if isinstance(info, dict) else None
            if ts is not None and m is not None and dt is not None:
                ts = float(ts)
                m = float(m)
                dt = float(dt)

                a_eff = (ts / max(1e-12, m)) * float(np.linalg.norm(aa_print))
                dv_eff = a_eff * dt

                dim = max(1, aa_print.size)
                a_max = (ts / max(1e-12, m)) * np.sqrt(dim)
                dv_max = a_max * dt

                print(
                    f"[SELF-CHECK] ts={ts:.3e}, mass={m:.3e}, dt={dt:.3e} | "
                    f"a_eff≈{a_eff:.3e} m/s^2, dv_eff≈{dv_eff:.3e} m/s | "
                    f"a_max≈{a_max:.3e}, dv_max≈{dv_max:.3e}"
                )

        x = np.asarray(obs, dtype=float)
        n = x.size // 2
        pos = x[:n]
        r = float(np.linalg.norm(pos))
        r_err = abs(r - target_radius)

        radius_errors.append(r_err)
        final_radius = r

        if steps % 200 == 0:
            rel_err = r_err / max(1e-12, target_radius)
            print(f"[SELF-CHECK] step={steps} r={r:.12e} r_err={r_err:.12e} rel_err={rel_err:.6e}")

        a_norm = np.linalg.norm(aa_print)
        if a_norm > 1e-12:
            cur_dir = aa_print / a_norm
            if prev_action_dir is not None:
                cos_theta = float(np.clip(np.dot(prev_action_dir, cur_dir), -1.0, 1.0))
                jitter_terms.append(1.0 - cos_theta)
            prev_action_dir = cur_dir

        steps += 1
        if done:
            break

    avg_radius_error = float(np.mean(radius_errors)) if radius_errors else 0.0
    avg_jitter = float(np.mean(jitter_terms)) if jitter_terms else 0.0

    if len(radius_errors) > 0:
        rmin = float(np.min(radius_errors))
        rmax = float(np.max(radius_errors))
        rmean = float(np.mean(radius_errors))
        print(f"[SELF-CHECK] radius_error stats ({label}/{scenario or 'default'}): min={rmin:.3e}, mean={rmean:.3e}, max={rmax:.3e}")

    sat_rate_mean = float(np.mean(sat_rates)) if sat_rates else 0.0
    print(
        f"[SELF-CHECK] saturation_rate_mean={sat_rate_mean:.3f} | "
        f"raw_norm_mean={np.mean(raw_norms):.3e} | "
        f"clip_norm_mean={np.mean(clip_norms):.3e}"
    )

    print(
        f"[{scenario or 'default'}] {label} | "
        f"steps={steps}, total_reward={total_reward:.3e}, "
        f"final_r={final_radius:.3e}, "
        f"avg_radius_error={avg_radius_error:.3e}, "
        f"avg_jitter={avg_jitter:.3e}"
    )

    return {
        "scenario": scenario or "default",
        "label": label,
        "steps": steps,
        "total_reward": float(total_reward),
        "final_r": float(final_radius) if final_radius is not None else None,
        "avg_radius_error": float(avg_radius_error),
        "avg_jitter": float(avg_jitter),
    }


def main():
    CFG_PATH = os.environ.get("CFG_PATH", os.path.join(PROJECT_ROOT, "config", "default.yaml"))
    cfg = load_cfg(CFG_PATH)

    scenarios = [
        "weak_thrust_far",
        "oscillation_noise",
        "misaligned_entry",
        None,
    ]

    results = []
    for s in scenarios:
        results.append(run_episode(s, "ExpertV3", ExpertV3, cfg))
        results.append(run_episode(s, "ExpertImproved", ExpertV4, cfg))

    key_fields = ["steps", "total_reward", "final_r", "avg_radius_error", "avg_jitter"]
    for label in ["ExpertV3", "ExpertImproved"]:
        subset = [r for r in results if r["label"] == label]
        sigs = {tuple(round(r[k], 12) if isinstance(r[k], float) else r[k] for k in key_fields) for r in subset}
        if len(sigs) == 1 and len(subset) >= 2:
            print(f"[SELF-CHECK][WARN] All scenarios produced identical metrics for {label}. "
                  f"This suggests scenario switching is not applied in OrbitEnv (or scenarios are equivalent).")


if __name__ == "__main__":
    main()
