import os
import sys
import numpy as np
import inspect
import json
import argparse
import csv
from pathlib import Path
import math
import hashlib
import subprocess
# WHPL_11: controller variant tracking (hard-coded, no auto-detect)
CONTROLLER_VARIANT = "whpl11_variant_tracking"


def dir_variation_unit(vecs: np.ndarray, eps: float = 1e-12) -> float:
    """
    vecs: (T, D)
    Returns mean(1 - cos(u_t, u_{t-1})) over valid consecutive pairs.
    0 means direction hardly changes; larger means more turning.
    """
    if vecs.shape[0] < 2:
        return 0.0

    v = vecs.astype(np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True)

    cos_list = []
    for t in range(1, v.shape[0]):
        if (n[t, 0] <= eps) or (n[t-1, 0] <= eps):
            continue
        u_t = v[t] / n[t, 0]
        u_p = v[t-1] / n[t-1, 0]
        c = float(np.clip(np.dot(u_t, u_p), -1.0, 1.0))
        cos_list.append(c)

    if len(cos_list) == 0:
        return 0.0

    cos_arr = np.array(cos_list, dtype=np.float64)
    return float(np.mean(1.0 - cos_arr))

def norm_mean(vecs: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(vecs.astype(np.float64), axis=1)))


def _parse_env_float(name: str):
    """Parse env var as float. Return None if missing/empty/invalid."""
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


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
    # Env override wins (used by ablation scripts)
    env_mode = os.environ.get("ACTION_IF_MODE", "").strip()
    if env_mode:
        return env_mode
    return (cfg.get("action_interface", {}) or {}).get("mode", "raw")


def get_thrust_scale(cfg: dict, env: OrbitEnv) -> float:
    # config wins; fallback to env default
    return float((cfg.get("actuation", {}) or {}).get("thrust_scale", getattr(env, "thrust_scale", 1.0)))


def build_controller(ControllerCls, env, obs):
    """Build controller using env.target_radius if available."""
    tr = getattr(env, "target_radius", None)
    if tr is None:
        x = np.asarray(obs, dtype=float)
        n = x.size // 2
        pos0 = x[:n]
        tr = float(np.linalg.norm(pos0))
    return ControllerCls(target_radius=float(tr))


def make_env_for_scenario(scenario_name=None, cfg=None, physical_override=None):
    """Emulate scenarios via:
    - reset(start_mode=...)
    - set_physical_params(...)

    Adds explicit scenario support metadata so we never "claim" effects that don't exist.
    """
    env = OrbitEnv()
    if cfg is not None:
        env.thrust_scale = get_thrust_scale(cfg, env)

    requested = (scenario_name or "default")
    effective = requested
    supported = True

    start_mode = "default"
    physical_kwargs = {}

    if scenario_name is None or scenario_name == "default":
        start_mode = "default"
    elif scenario_name == "spiral":
        start_mode = "spiral"
    elif scenario_name == "weak_thrust_far":
        # Base scenario uses a physical override to be "weak thrust".
        pass
    elif scenario_name == "misaligned_entry":
        # Currently implemented as a distinct scenario label but same start_mode.
        # Keep it supported, but do not pretend it changes anything if OrbitEnv doesn't.
        start_mode = "default"
    elif scenario_name == "oscillation_noise":
        # OrbitEnv has no noise API -> explicitly mark unsupported and force default.
        print("[SELF-CHECK][WARN] 'oscillation_noise' not supported by OrbitEnv (no noise API). Using default.")
        supported = False
        effective = "default"
        start_mode = "default"
    else:
        print(f"[SELF-CHECK][WARN] Unknown scenario '{scenario_name}'. Using default.")
        supported = False
        effective = "default"
        start_mode = "default"

    # Optional explicit physical overrides (e.g., thrust sweep)
    if physical_override:
        for k, v in dict(physical_override).items():
            physical_kwargs[k] = v

    if physical_kwargs and hasattr(env, "set_physical_params"):
        env.set_physical_params(**physical_kwargs)

    # Attach metadata for downstream logging
    env._qc_scenario_requested = requested
    env._qc_scenario_effective = effective
    env._qc_scenario_supported = bool(supported)
    env._qc_start_mode = start_mode
    env._qc_physical_kwargs = physical_kwargs
    return env


def _obs_signature(obs, k=8):
    x = np.asarray(obs, dtype=float).ravel()
    return x[: min(k, x.size)]


def _unpack_step(step_out):
    """Gym (obs, reward, done, info) or Gymnasium (obs, reward, terminated, truncated, info)"""
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    else:
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), info


def _try_shadow_step(env, action, atol_obs=1e-9, atol_reward=1e-9):
    """Compare step(action) vs step(0) from the exact same state using get_state/set_state.

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

        print(
            f"[SELF-CHECK] shadow_step | ||obs(a)-obs(0)||={dr:.6e} | |rew(a)-rew(0)|={d_reward:.6e} | done_diff={d_done}"
        )

        if dr < atol_obs and d_reward < atol_reward and (not d_done):
            print(
                "[SELF-CHECK][WARN] shadow_step diffs are ~0 (obs/reward/done). "
                "Strong evidence action is not affecting dynamics/reward at this state."
            )

        for k in ["thrust_vec", "acc_thrust", "acc_total"]:
            if isinstance(info_a, dict) and isinstance(info_z, dict) and (k in info_a or k in info_z):
                print(f"[SELF-CHECK] info[{k}] a={info_a.get(k, None)} | 0={info_z.get(k, None)}")

    finally:
        env.set_state(saved)


def _get_mu(env):
    """
    Try to obtain gravitational parameter mu (GM).
    Return None if not available.
    """
    for name in ["mu", "GM", "gm", "gravitational_parameter"]:
        if hasattr(env, name):
            try:
                return float(getattr(env, name))
            except Exception:
                pass

    # Fallback patterns
    # e.g., env may store G and central mass M
    if hasattr(env, "G") and hasattr(env, "M"):
        try:
            return float(env.G) * float(env.M)
        except Exception:
            return None
    if hasattr(env, "G") and hasattr(env, "central_mass"):
        try:
            return float(env.G) * float(env.central_mass)
        except Exception:
            return None

    return None


def _safe_unit(v, eps=1e-12):
    """Return (unit_vector_or_None, norm). If norm too small -> (None, norm)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if n <= eps:
        return None, n
    return (v / n), n


def run_episode(scenario, label, ControllerCls, cfg, max_steps=2000, physical_override=None, lambda_sat=0.10):
    env = make_env_for_scenario(scenario, cfg=cfg, physical_override=physical_override)

    start_mode = getattr(env, "_qc_start_mode", "default")

    reset_result = env.reset(start_mode=start_mode)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs, info = reset_result, {}

    controller = build_controller(ControllerCls, env, obs)

    # WHPL_03: initial radius r0 for CSV (no physics/control change)
    x0 = np.asarray(obs, dtype=float)
    n0 = x0.size // 2
    pos0 = x0[:n0]
    r0 = float(np.linalg.norm(pos0))

    print(f"[SELF-CHECK] {label} ControllerCls={ControllerCls} | file={inspect.getsourcefile(ControllerCls)}")

    target_radius = float(getattr(env, "target_radius", 0.0))
    if target_radius <= 0.0:
        x0 = np.asarray(obs, dtype=float)
        n = x0.size // 2
        pos0 = x0[:n]
        target_radius = float(np.linalg.norm(pos0))

    requested = getattr(env, "_qc_scenario_requested", scenario or "default")
    effective = getattr(env, "_qc_scenario_effective", requested)
    supported = bool(getattr(env, "_qc_scenario_supported", True))
    phys = getattr(env, "_qc_physical_kwargs", {})

    print(
        f"[SELF-CHECK] scenario_requested={requested} | scenario_effective={effective} | scenario_supported={supported} | "
        f"start_mode={start_mode} | phys={phys} | target_r={target_radius:.6e}"
    )

    ctrl_tr = getattr(controller, "target_radius", None)
    if ctrl_tr is not None:
        try:
            ctrl_tr = float(ctrl_tr)
            rel = abs(ctrl_tr - target_radius) / max(1e-12, abs(target_radius))
            if rel > 1e-6:
                print(
                    "[SELF-CHECK][WARN] controller.target_radius != env.target_radius "
                    f"({ctrl_tr:.6e} vs {target_radius:.6e}, rel_diff={rel:.3e})"
                )
        except Exception:
            pass

    total_reward = 0.0
    final_radius = None
    radius_errors = []
    jitter_terms = []
    prev_action_dir = None

    raw_norms = []   # ||thrust_intent|| (N)
    clip_norms = []  # ||action|| (dimensionless)
    sat_rates = []   # saturation_rate from action_interface (single source of truth)

    do_shadow_step = True
    # --- WHPL_08 BEGIN: state for dr/dt (logging only) ---
    prev_r_norm_scalar = None
    # --- WHPL_08 END ---
    steps = 0

    # Cache config for action interface
    cfg_ai = ActionInterfaceConfig(mode=get_ai_mode(cfg))

    # WHPL_02: statistics buffers
    thrust_intents = []
    actions = []
    a_effs = []

    for _ in range(max_steps):
        # 1) Controller outputs thrust intent in Newtons
        thrust_intent = controller.act(obs)

        # 2) Action interface: convert thrust intent -> normalized action
        # Use a fixed reference scale to decouple action normalization from engine strength
        REF_THRUST_SCALE = float((cfg.get("action_interface", {}) or {}).get("thrust_scale_ref", 3000.0))
        action, ainfo = thrust_to_action(thrust_intent, REF_THRUST_SCALE, cfg_ai)
        sat_rates.append(float(ainfo.get("saturation_rate", 0.0)))

        # 3) Stats
        ra = np.asarray(thrust_intent, dtype=float).ravel()
        aa_print = np.asarray(action, dtype=float).ravel()
        # WHPL_02: record sequences for interface-compression diagnosis
        thrust_intents.append(ra.astype(np.float64).copy())
        actions.append(aa_print.astype(np.float64).copy())

        raw_norms.append(float(np.linalg.norm(ra)))
        clip_norms.append(float(np.linalg.norm(aa_print)))

        if do_shadow_step:
            do_shadow_step = False
            _try_shadow_step(env, action)

        # Step environment
        obs, reward, done, info = _unpack_step(env.step(action))

        # Attach interface metrics to info (avoid key collisions)
        if isinstance(info, dict):
            info["action_if"] = dict(ainfo)
        if isinstance(info, dict) and ("a_eff" in info):
            a_effs.append(float(info["a_eff"]))

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

            x2 = np.asarray(obs, dtype=np.float64).ravel()
            n2 = x2.size // 2
            r_vec = x2[:n2]
            v_vec = x2[n2:]

            r_unit, r_norm = _safe_unit(r_vec)
            v_unit, v_norm = _safe_unit(v_vec)

            # Prefer env-provided thrust_vec; fallback to controller thrust_intent
            if isinstance(info, dict) and ("thrust_vec" in info) and (info["thrust_vec"] is not None):
                thrust_vec = np.asarray(info["thrust_vec"], dtype=np.float64).ravel()
            else:
                thrust_vec = np.asarray(thrust_intent, dtype=np.float64).ravel()

            t_unit, t_norm = _safe_unit(thrust_vec)

            # --- WHPL_08 BEGIN: radial/tangential decomposition (logging only) ---
            # NOTE: in your code, t_unit currently means "thrust unit". We'll keep it but add explicit names.
            th_unit = t_unit  # thrust direction unit

            # 2D tangential unit = rotate r_unit by +90 deg
            if (r_unit is not None) and (r_vec.size == 2):
                tan_unit = np.array([-r_unit[1], r_unit[0]], dtype=np.float64)
            else:
                tan_unit = None

            if (th_unit is not None) and (r_unit is not None) and (tan_unit is not None):
                cos_tr = float(np.clip(np.dot(th_unit, r_unit), -1.0, 1.0))  # cos(thrust, radial)
                cos_tt = float(np.clip(np.dot(th_unit, tan_unit), -1.0, 1.0))  # cos(thrust, tangential)
            else:
                cos_tr = float("nan")
                cos_tt = float("nan")

            # radial velocity v_r = dot(v, r_unit)
            if (v_vec is not None) and (r_unit is not None):
                v_r = float(np.dot(v_vec, r_unit))
            else:
                v_r = float("nan")

            # dr/dt using env dt if available
            dt_now = None
            if isinstance(info, dict) and ("dt" in info) and (info["dt"] is not None):
                try:
                    dt_now = float(info["dt"])
                except Exception:
                    dt_now = None

            if (prev_r_norm_scalar is not None) and (dt_now is not None) and (dt_now > 0.0) and np.isfinite(r_norm):
                dr_dt = float((r_norm - prev_r_norm_scalar) / dt_now)
            else:
                dr_dt = float("nan")
            # --- WHPL_08 END ---

            # (1) Thrust vs velocity alignment
            if (t_unit is not None) and (v_unit is not None):
                cos_tv = float(np.clip(np.dot(t_unit, v_unit), -1.0, 1.0))
            else:
                cos_tv = float("nan")

            # (2) Specific orbital energy: eps = v^2/2 - mu/r
            mu = _get_mu(env)
            if (mu is not None) and (r_norm > 1e-12) and (v_norm > 0.0):
                eps = 0.5 * (v_norm ** 2) - float(mu) / r_norm
            else:
                eps = float("nan")

            # (3) Angular momentum magnitude |h| = |r x v|
            if r_vec.size == 2 and v_vec.size == 2:
                h = float(abs(r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]))
            elif r_vec.size >= 3 and v_vec.size >= 3:
                h = float(np.linalg.norm(np.cross(r_vec[:3], v_vec[:3])))
            else:
                h = float("nan")

            print(
                f"[SELF-CHECK] step={steps} r={r_norm:.12e} r_err={r_err:.12e} rel_err={rel_err:.6e} | "
                f"v={v_norm:.3e} cos(thrust,vel)={cos_tv:+.6f} | "
                f"eps={eps:.6e} |h|={h:.6e} | thrust_norm={t_norm:.3e} | "
                f"[WHPL_10] cos(thrust,rad)={cos_tr:+.6f} cos(thrust,tan)={cos_tt:+.6f} v_r={v_r:+.6e} dr_dt={dr_dt:+.6e}"
            )

            # --- WHPL_08 BEGIN: update state ---
            prev_r_norm_scalar = float(r_norm) if np.isfinite(r_norm) else prev_r_norm_scalar
            # --- WHPL_08 END ---

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

    # WHPL_02: interface compression stats
    thrust_intents_arr = np.stack(thrust_intents, axis=0)  # (T, D)
    actions_arr = np.stack(actions, axis=0)                # (T, A)

    thrust_intent_norm_mean = norm_mean(thrust_intents_arr)
    thrust_intent_dir_var = dir_variation_unit(thrust_intents_arr)
    action_norm_mean = norm_mean(actions_arr)

    print(f"[STAT] thrust_intent_norm_mean={thrust_intent_norm_mean:.6e}")
    print(f"[STAT] thrust_intent_dir_var={thrust_intent_dir_var:.6e}")
    print(f"[STAT] action_norm_mean={action_norm_mean:.6e}")

    avg_radius_error = float(np.mean(radius_errors)) if radius_errors else 0.0
    avg_jitter = float(np.mean(jitter_terms)) if jitter_terms else 0.0

    if len(radius_errors) > 0:
        rmin = float(np.min(radius_errors))
        rmax = float(np.max(radius_errors))
        rmean = float(np.mean(radius_errors))
        print(f"[SELF-CHECK] radius_error stats ({label}/{requested}): min={rmin:.3e}, mean={rmean:.3e}, max={rmax:.3e}")

    sat_rate_mean = float(np.mean(sat_rates)) if sat_rates else 0.0

    # Saturation penalty metric (monotone, continuous, interpretable):
    # reward_minus_lambda_sat = total_reward - lambda_sat * sat_rate_mean * (|total_reward| + 1)
    # - sat_rate_mean up -> score strictly decreases
    # - does NOT collapse information to 0 even when sat_rate_mean -> 1
    penalty = float(lambda_sat) * float(np.clip(sat_rate_mean, 0.0, 1.0)) * (abs(float(total_reward)) + 1.0)
    reward_minus_lambda_sat = float(total_reward) - float(penalty)

    # Keep a legacy name, but now it means "saturation-penalized reward" (not scaling-to-zero).
    sat_adjusted_score = float(reward_minus_lambda_sat)

    print(
        f"[SELF-CHECK] saturation_rate_mean={sat_rate_mean:.3f} | "
        f"reward_minus_lambda_sat={reward_minus_lambda_sat:.3e} (lambda_sat={lambda_sat:.3f}) | "
        f"raw_norm_mean={np.mean(raw_norms):.3e} | "
        f"clip_norm_mean={np.mean(clip_norms):.3e}"
    )

    print(
        f"[{requested}] {label} | "
        f"steps={steps}, total_reward={total_reward:.3e}, "
        f"final_r={final_radius:.3e}, "
        f"avg_radius_error={avg_radius_error:.3e}, "
        f"avg_jitter={avg_jitter:.3e}"
    )

    return {
        "scenario_requested": requested,
        "scenario_effective": effective,
        "scenario_supported": bool(supported),
        "phys": dict(phys) if isinstance(phys, dict) else {},
        "label": label,
        "steps": int(steps),
        "total_reward": float(total_reward),
        "final_r": float(final_radius) if final_radius is not None else None,
        "avg_radius_error": float(avg_radius_error),
        "avg_jitter": float(avg_jitter),
        "saturation_rate_mean": float(sat_rate_mean),
        "lambda_sat": float(lambda_sat),
        "reward_minus_lambda_sat": float(reward_minus_lambda_sat),
        "sat_adjusted_score": float(sat_adjusted_score),
        "target_radius": float(target_radius),
        "r0_over_target": float(r0 / max(1e-12, target_radius)),

    }


def _git_short_sha() -> str:
    """Best-effort git commit id; empty if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _compute_dedup_key(row: dict) -> str:
    """
    Idempotent logging key. Same logical run => same key => block duplicate append.
    Only 'dedup_key' is persisted; other ingredients are not added as new columns.
    """
    code_ver = _git_short_sha()

    parts = [
        str(row.get("controller", "")),
        str(row.get("controller_variant", "legacy")),  # WHPL_11 NEW
        str(row.get("scenario", "")),
        str(row.get("thrust_newton", "")),
        f"{float(row.get('r0_over_target', 0.0)):.12g}",
        f"{float(row.get('target_r', 0.0)):.12g}",
        code_ver,
    ]

    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def append_row_csv(csv_path: str, fieldnames: list, row: dict, key_name: str = "dedup_key"):
    """
    Append one row into CSV with dedup based on `dedup_key`.
    Compatible with existing call: append_row_csv(CSV_PATH, FIELDNAMES, row)
    """

    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure key column exists in target schema
    if key_name not in fieldnames:
        fieldnames = list(fieldnames) + [key_name]

    # --- Create new file ---
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
        return "created"

    # --- Read existing file ---
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        existing_fieldnames = list(r.fieldnames) if r.fieldnames else []
        rows = list(r)

    # --- Upgrade header if missing dedup_key ---
    upgraded = False
    if key_name not in existing_fieldnames:
        existing_fieldnames.append(key_name)
        for rr in rows:
            rr[key_name] = rr.get(key_name, "")
        upgraded = True

    # Also ensure any new columns from fieldnames are included
    for col in fieldnames:
        if col not in existing_fieldnames:
            existing_fieldnames.append(col)
            for rr in rows:
                rr[col] = rr.get(col, "")
            upgraded = True

    # --- Dedup check ---
    new_key = row.get(key_name, "")
    if new_key:
        for rr in rows:
            if rr.get(key_name, "") == new_key:
                if upgraded:
                    with p.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=existing_fieldnames)
                        w.writeheader()
                        w.writerows([{k: x.get(k, "") for k in existing_fieldnames} for x in rows])
                    return "upgraded_skip"
                return "skip"

    # --- Append and rewrite ---
    rows.append({k: row.get(k, "") for k in existing_fieldnames})

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=existing_fieldnames)
        w.writeheader()
        w.writerows([{k: x.get(k, "") for k in existing_fieldnames} for x in rows])

    return "upgraded_append" if upgraded else "append"


def main():
    CFG_PATH = os.environ.get("CFG_PATH", os.path.join(PROJECT_ROOT, "config", "default.yaml"))
    cfg = load_cfg(CFG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None, help="run single scenario only")
    parser.add_argument("--thrust", type=float, default=None, help="use single thrust_newton")
    parser.add_argument("--no_sweep", action="store_true", help="disable default thrust sweep")
    args = parser.parse_args()

    # WHPL_03 guardrail: disallow CLI overrides that change today's single-run coordinate
    if args.scenario is not None or args.no_sweep or args.thrust is not None:
        parser.error("WHPL_03 is frozen: do not pass --scenario/--no_sweep/--thrust today.")

    # WHPL_03 fixed single-run coordinate (no sweep, no comparison)
    WHPL3_THRUST_NEWTON = 800.0
    WHPL3_DIFFICULTY_TAG = os.environ.get("DIFFICULTY_TAG", "Hard").strip() or "Hard"

    # CLI guardrails
    if args.no_sweep and (args.thrust is None):
        parser.error("--no_sweep requires --thrust (e.g., --no_sweep --thrust 200)")
    if (args.thrust is not None) and (not args.no_sweep):
        parser.error("--thrust only works with --no_sweep. Use: --no_sweep --thrust 200")
    if (args.scenario == "weak_thrust_far") and (not args.no_sweep) and (_parse_env_float("THRUST_NEWTON") is None):
        print(
            "[SELF-CHECK][WARN] --scenario weak_thrust_far without --no_sweep/THRUST_NEWTON -> will run default thrust sweep.")

    # One place to tune the saturation penalty.
    # Keep it small; we only want to punish saturation without deleting reward scale.
    LAMBDA_SAT = float(os.environ.get("LAMBDA_SAT", "0.10"))

    # Controlled contrast: same scenario, sweep thrust levels.
    env_ts = _parse_env_float("THRUST_NEWTON")

    # Priority: CLI --no_sweep/--thrust > env THRUST_NEWTON > default sweep
    if args.no_sweep:
        thrust_sweep = [float(args.thrust)]
        print(f"[SELF-CHECK] cli override: --no_sweep (single-run thrust={thrust_sweep[0]})")
    elif env_ts is not None:
        thrust_sweep = [env_ts]
        print(f"[SELF-CHECK] env override: THRUST_NEWTON={env_ts} (single-run)")
    else:
        thrust_sweep = [200.0, 800.0, 2000.0]
        print(f"[SELF-CHECK] env override: none (default sweep={thrust_sweep})")

    ts_for_scenario = float(env_ts) if env_ts is not None else float(WHPL3_THRUST_NEWTON)

    scenarios = [("weak_thrust_far", {"thrust_newton": ts_for_scenario})]

    # CLI: run single scenario only
    if args.scenario is not None:
        # keep physical overrides ONLY if that scenario already carries one
        scenarios = [(s, p) for (s, p) in scenarios if (s or "default") == args.scenario]
        if len(scenarios) == 0:
            # fallback: run requested scenario with no physical override
            scenarios = [(args.scenario, None)]

    results = []
    for s, phys_override in scenarios:
        results.append(
            run_episode(s, "ExpertImproved", ExpertV4, cfg, physical_override=phys_override, lambda_sat=LAMBDA_SAT)
        )

    assert len(results) == 1, f"WHPL_03 expects exactly 1 run, got {len(results)}"

    # WHPL_03: append exactly one CSV row per run
    CSV_PATH = Path(PROJECT_ROOT) / "analysis" / "results" / "ablation_thrust_x_difficulty.csv"
    FIELDNAMES = [
        "dedup_key",
        "controller_variant",  # WHPL_11 NEW
        "thrust_newton",
        "difficulty_tag",
        "r0_over_target",
        "avg_radius_error",
        "final_radius_error",
        "saturation_rate_mean",
        "total_reward",
    ]

    r = results[-1]
    target_r = float(r.get("target_radius", math.nan))
    r0_over_target = float(r.get("r0_over_target", math.nan))

    final_r = r.get("final_r", None)
    if final_r is None or (not np.isfinite(float(final_r))) or (not np.isfinite(target_r)):
        final_radius_error = math.nan
    else:
        final_radius_error = float(abs(float(final_r) - target_r))

    row = {
        "controller_variant": CONTROLLER_VARIANT,  # WHPL_11 NEW
        "thrust_newton": float(ts_for_scenario),
        "difficulty_tag": str(os.environ.get("DIFFICULTY_TAG", WHPL3_DIFFICULTY_TAG)),
        "r0_over_target": float(r0_over_target),
        "avg_radius_error": float(r.get("avg_radius_error", math.nan)),
        "final_radius_error": float(final_radius_error),
        "saturation_rate_mean": float(r.get("saturation_rate_mean", math.nan)),
        "total_reward": float(r.get("total_reward", math.nan)),
    }

    row["controller"] = "ExpertImproved"  # used only for key computation (not saved as a column)
    row["scenario"] = r.get("scenario_effective", r.get("scenario_requested", ""))
    row["target_r"] = target_r  # used only for key computation

    row["dedup_key"] = _compute_dedup_key(row)
    status = append_row_csv(CSV_PATH, FIELDNAMES, row)
    print(f"[WHPL_03] csv_status={status} -> {CSV_PATH}")
    print(f"[WHPL_03] row = {row}")

    # Sanity check: if everything is identical, something is wrong.
    key_fields = [
        "steps",
        "total_reward",
        "final_r",
        "avg_radius_error",
        "avg_jitter",
        "saturation_rate_mean",
        "reward_minus_lambda_sat",
    ]

    # for label in ["ExpertV3", "ExpertImproved"]:
    #     subset = [r for r in results if r["label"] == label]
    #     sigs = {
    #         tuple(
    #             round(r[k], 12) if isinstance(r.get(k, None), float) else r.get(k, None)
    #             for k in key_fields
    #         )
    #         for r in subset
    #     }
    #     if len(sigs) == 1 and len(subset) >= 2:
    #         print(
    #             f"[SELF-CHECK][WARN] All scenarios produced identical metrics for {label}. "
    #             "This suggests scenario switching is not applied in OrbitEnv (or scenarios are equivalent)."
    #         )

    out_path = os.path.join(PROJECT_ROOT, "analysis", "SESSION6_metrics_upgrade.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"cfg_path": CFG_PATH, "lambda_sat": LAMBDA_SAT}, "results": results}, f, indent=2)
    print(f"[SELF-CHECK] wrote: {out_path}")


if __name__ == "__main__":
    main()
