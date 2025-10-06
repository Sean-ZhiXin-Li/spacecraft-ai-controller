"""
Day 52 — Replay Player

Replays recorded actions deterministically and computes drift metrics
against the recorded observation sequence.

Highlights:
- Robust import via "module:path" (class or factory function is fine).
- Metrics: l2_max, linf_max globally and on named slices (e.g., pos, vel).
- CI-friendly: --strict makes the process exit non-zero on failure.
- PowerShell-friendly args:
    * --scenario <name>                  # avoids JSON for common case
    * --extra-kv key=value (repeatable)  # add overrides without JSON
    * --extra '{...}'                    # still supported (JSON or Python literal)

Usage example (no JSON needed):
  python tools/replay_player.py \
    --env-factory envs.multi_orbit_env:MultiOrbitEnv \
    --seed 51 \
    --replay logs/day51/elliptic_strong/e_strong_e0.60_s51/replay.npz \
    --slices pos:0:2,vel:2:4 \
    --thr 5e-4 --strict \
    --scenario elliptic_strong

If you still want JSON for extra params:
  --extra '{"scenario":"elliptic_strong","preset_overrides":{"ecc":0.60}}'
"""
from __future__ import annotations

import argparse
import importlib
import json
import ast
from pathlib import Path
from typing import Dict
import sys

import numpy as np


def import_from_path(path: str):
    """Import object from 'module.submodule:object_name' path."""
    if ":" not in path:
        raise ValueError(f"Expected 'module:object', got: {path}")
    mod, obj = path.split(":", 1)
    return getattr(importlib.import_module(mod), obj)


def parse_slices(spec: str, dim: int) -> Dict[str, slice]:
    """
    Parse a slice spec like 'pos:0:2,vel:2:4' into a dict(name -> slice).
    """
    mapping: Dict[str, slice] = {}
    if not spec:
        return mapping
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name, a, b = token.split(":")
        i, j = int(a), int(b)
        if not (0 <= i <= j <= dim):
            raise ValueError(f"Invalid slice {name}: {i}:{j} for dim={dim}")
        mapping[name] = slice(i, j)
    return mapping


def l2(x): return float(np.linalg.norm(x, ord=2))
def linf(x): return float(np.linalg.norm(x, ord=np.inf))


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

    # Legacy/compatible flags (your env may ignore unknown keys)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-factory", required=True, type=str, help="Import path to env factory or class")
    ap.add_argument("--replay", required=True, type=str, help="Path to replay.npz")
    ap.add_argument("--seed", type=int, required=True, help="Seed used during recording")

    # Kept for compatibility; your env may ignore them
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--scene", type=str, default=None)

    # PowerShell-friendly additions (no JSON required)
    ap.add_argument("--scenario", type=str, default=None,
                    help="Scenario name for envs expecting scenario=... (e.g., elliptic_strong)")
    ap.add_argument("--extra-kv", action="append", default=[],
                    help='Extra overrides as key=value (repeatable). Example: --extra-kv ecc=0.60 --extra-kv dt=0.5')

    # Optional JSON/literal extras (still supported)
    ap.add_argument("--extra", type=str, default=None,
                    help="JSON/Python-literal dict forwarded to env factory")

    ap.add_argument("--slices", type=str, default="",
                    help="Named slices for metrics, e.g., pos:0:2,vel:2:4")
    ap.add_argument("--thr", type=float, default=5e-4,
                    help="Threshold for pass/fail on l2/linf (applied to all _max metrics)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any metric exceeds threshold")
    ap.add_argument("--json-out", type=str, default="",
                    help="Optional path to write metrics JSON")

    args = ap.parse_args()

    # Load replay
    data = np.load(args.replay)
    obs = data["obs"]        # (T+1, D)
    actions = data["actions"]  # (T, A)
    T = actions.shape[0]
    D = obs.shape[1]
    slices = parse_slices(args.slices, D)

    # Build env
    factory = import_from_path(args.env_factory)
    factory_kwargs = build_factory_kwargs(args)

    env = factory(**factory_kwargs)
    try:
        o, _ = env.reset(seed=args.seed)
    except TypeError:
        o = env.reset(seed=args.seed)

    # Step through and compare against recorded next-obs
    drifts = []
    for t in range(T):
        nobs, _, term, trunc, _ = env.step(actions[t])
        ref = obs[t + 1]
        diff = np.asarray(nobs, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        drifts.append(diff)
        if term or trunc:
            break

    drifts = np.stack(drifts, axis=0) if drifts else np.zeros((0, D), dtype=np.float64)

    # Aggregate metrics
    agg = {
        "steps_checked": int(drifts.shape[0]),
        "l2_max": float(np.max(np.linalg.norm(drifts, axis=1))) if drifts.size else 0.0,
        "linf_max": float(np.max(np.max(np.abs(drifts), axis=1))) if drifts.size else 0.0,
    }

    # Named slices
    for name, sl in slices.items():
        if drifts.size:
            l2m = float(np.max([l2(d[sl]) for d in drifts]))
            lim = float(np.max([linf(d[sl]) for d in drifts]))
        else:
            l2m = lim = 0.0
        agg[f"{name}_l2_max"] = l2m
        agg[f"{name}_linf_max"] = lim

    # Pass/Fail
    fail = any(v > args.thr for k, v in agg.items() if k.endswith("_max"))
    agg["threshold"] = args.thr
    agg["pass"] = (not fail)

    # Optional JSON output
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

    # Print metrics (stdout)
    print(json.dumps(agg, indent=2))

    if args.strict and fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
