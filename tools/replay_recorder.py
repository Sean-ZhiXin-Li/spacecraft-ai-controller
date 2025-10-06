"""
Day 52 — Replay Recorder

Records a deterministic baseline trajectory as:
- replay.npz: obs (T+1, ...), actions (T, ...), rewards (T,), dones (T,)
- meta.json: metadata for reproducibility.

Highlights:
- Robust import via "module:object" (class or factory function).
- PowerShell-friendly args:
    * --scenario <name>                  # avoids JSON for common case
    * --extra-kv key=value (repeatable)  # add overrides without JSON
    * --extra '{...}'                    # still supported (JSON or Python literal)
- Controller call compatibility: .act(obs, info) / .act(obs) / __call__(obs, info=...) / __call__(obs)

Typical use:
  python tools/replay_recorder.py \
    --env-factory envs.multi_orbit_env:MultiOrbitEnv \
    --steps 400 --seed 51 \
    --policy controllers.expert:ExpertController \
    --out logs/day51/elliptic_strong/e_strong_e0.60_s51 \
    --scenario elliptic_strong
"""
from __future__ import annotations

import argparse
import importlib
import json
import ast
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ------------------------- utils -------------------------

def import_from_path(path: str):
    """Import object from 'module.submodule:object_name' path."""
    if ":" not in path:
        raise ValueError(f"Expected 'module:object', got: {path}")
    mod_name, obj_name = path.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, obj_name)


def get_git_commit() -> str:
    """Return short git commit hash if inside a git repo, else 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Filter/compact info dict to JSON-serializable scalars only."""
    compact = {}
    for k, v in info.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            compact[k] = v
        elif hasattr(v, "item"):
            try:
                compact[k] = v.item()
            except Exception:
                pass
    return compact


def compute_action(controller: Any, obs: np.ndarray, info: Dict[str, Any] | None) -> np.ndarray:
    """
    Try multiple common controller signatures gracefully.
    Priority:
      1) controller.act(obs, info)
      2) controller.act(obs)
      3) controller(obs, info=info)      # __call__
      4) controller(obs)
    """
    if hasattr(controller, "act"):
        try:
            return np.asarray(controller.act(obs, info), dtype=np.float64)
        except TypeError:
            return np.asarray(controller.act(obs), dtype=np.float64)
    # __call__ path
    try:
        return np.asarray(controller(obs, info=info), dtype=np.float64)
    except TypeError:
        return np.asarray(controller(obs), dtype=np.float64)


def coerce_scalar(s: str):
    """Convert a string to bool/int/float if possible; else keep as str."""
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def build_factory_kwargs(args) -> dict:
    """
    Build kwargs for the env factory/class constructor with multiple input styles:
      - --preset / --scene (kept for compatibility)
      - --scenario (direct mapping for classes expecting scenario=...)
      - --extra-kv key=value (repeatable; placed into preset_overrides by default)
      - --extra '{...}' (JSON or Python literal dict)
    """
    factory_kwargs: dict = {}

    # Legacy/compatible flags (ok if ignored by env)
    if args.preset is not None:
        factory_kwargs["preset"] = args.preset
    if args.scene is not None:
        factory_kwargs["scene"] = args.scene

    # Direct scenario for classes like MultiOrbitEnv(scenario=...)
    if args.scenario is not None:
        factory_kwargs["scenario"] = args.scenario

    # key=value extras -> inject into preset_overrides by default
    if args.extra_kv:
        po = factory_kwargs.get("preset_overrides", {})
        for kv in args.extra_kv:
            if "=" not in kv:
                raise ValueError(f"--extra-kv expects key=value, got: {kv}")
            k, v = kv.split("=", 1)
            po[k.strip()] = coerce_scalar(v.strip())
        factory_kwargs["preset_overrides"] = po

    # Still support --extra as JSON; fallback to Python literal
    if args.extra:
        extra_str = args.extra
        parsed = None
        try:
            parsed = json.loads(extra_str)
        except Exception:
            try:
                parsed = ast.literal_eval(extra_str)
            except Exception:
                raise ValueError(f"Cannot parse --extra: {extra_str!r}")
        if not isinstance(parsed, dict):
            raise ValueError("--extra must be a dict-like object")
        factory_kwargs.update(parsed)

    return factory_kwargs


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-factory", type=str, required=True,
                    help="Import path to env factory or class (e.g., envs.multi_orbit_env:MultiOrbitEnv)")
    ap.add_argument("--policy", type=str, default=None,
                    help="Import path to controller/policy. If omitted, uses action_space.sample() seeded.")
    ap.add_argument("--preset", type=str, default=None, help="Kept for compatibility (env may ignore)")
    ap.add_argument("--scene", type=str, default=None, help="Kept for compatibility (env may ignore)")
    ap.add_argument("--steps", type=int, default=1000, help="Number of environment steps to record")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for env & numpy")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory where replay.npz and meta.json are written")

    # PowerShell-friendly (no JSON needed)
    ap.add_argument("--scenario", type=str, default=None,
                    help="Scenario name for envs expecting scenario=... (e.g., elliptic_strong)")
    ap.add_argument("--extra-kv", action="append", default=[],
                    help='Extra overrides as key=value (repeatable). Example: --extra-kv ecc=0.60 --extra-kv dt=0.5')

    # Optional JSON/literal extras (still supported)
    ap.add_argument("--extra", type=str, default=None,
                    help="JSON/Python-literal dict forwarded to env factory")

    ap.add_argument("--record-info-keys", type=str, default="",
                    help="Comma-separated info keys to preserve (optional)")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                    help="Array dtype to save")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Build env
    factory = import_from_path(args.env_factory)
    factory_kwargs = build_factory_kwargs(args)

    env = factory(**factory_kwargs)
    # Gymnasium-style reset
    try:
        obs, info = env.reset(seed=args.seed)
    except TypeError:
        obs = env.reset(seed=args.seed)
        info = {}

    # Build controller / policy
    controller = None
    if args.policy:
        Policy = import_from_path(args.policy)
        try:
            controller = Policy()  # class
        except Exception:
            controller = Policy  # function/callable

    # Allocate storage
    obs_dtype = np.float64 if args.dtype == "float64" else np.float32
    acts, rews, dns, infos_kept = [], [], [], []

    # Loop
    for t in range(args.steps):
        if controller is None:
            # Seeded random baseline for reproducibility
            action = np.asarray(env.action_space.sample(), dtype=obs_dtype)
        else:
            action = compute_action(controller, np.asarray(obs, dtype=obs_dtype), info)

        nobs, rew, terminated, truncated, ninfo = env.step(action)

        # Keep selected info keys if requested
        if args.record_info_keys:
            selected = {}
            for k in args.record_info_keys.split(","):
                k = k.strip()
                if k and isinstance(ninfo, dict) and k in ninfo:
                    selected[k] = ninfo[k]
            if selected:
                infos_kept.append(serialize_info(selected))

        acts.append(np.asarray(action, dtype=obs_dtype))
        rews.append(float(rew))
        dns.append(bool(terminated or truncated))

        obs = nobs
        info = ninfo
        if terminated or truncated:
            break

    # Rebuild full obs sequence (T+1) by replaying actions once deterministically
    env2 = factory(**factory_kwargs)
    try:
        o2, _ = env2.reset(seed=args.seed)
    except TypeError:
        o2 = env2.reset(seed=args.seed)
    obs_seq = [np.asarray(o2, dtype=obs_dtype)]
    for a in acts:
        o2, _, ter2, tru2, _ = env2.step(a)
        obs_seq.append(np.asarray(o2, dtype=obs_dtype))
        if ter2 or tru2:
            break

    obs_arr = np.stack(obs_seq, axis=0)
    acts_arr = np.stack(acts, axis=0) if acts else np.zeros((0,), dtype=obs_dtype)
    rews_arr = np.asarray(rews, dtype=obs_dtype)
    dns_arr = np.asarray(dns, dtype=np.bool_)

    outdir = Path(args.out)
    ensure_parent(outdir / "replay.npz")

    # Save .npz
    np.savez_compressed(
        outdir / "replay.npz",
        obs=obs_arr, actions=acts_arr, rewards=rews_arr, dones=dns_arr
    )

    # Save meta.json
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": get_git_commit(),
        "seed": args.seed,
        "preset": args.preset,
        "scene": args.scene,
        "steps_recorded": int(acts_arr.shape[0]),
        "env_factory": args.env_factory,
        "policy": args.policy or "action_space.sample()",
        "dtype": args.dtype,
        "info_keys": [k.strip() for k in (args.record_info_keys or "").split(",") if k.strip()],
        "extra": factory_kwargs,
        "format_version": "day51.v1",
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[recorder] wrote {outdir/'replay.npz'} and {outdir/'meta.json'}")
    print(f"[recorder] steps={meta['steps_recorded']} seed={args.seed} scenario={factory_kwargs.get('scenario')}")
    print("[recorder] done.")


if __name__ == "__main__":
    main()
