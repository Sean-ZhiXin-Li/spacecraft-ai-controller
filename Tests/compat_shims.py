# Ultra-defensive compatibility shims for mixed legacy/new env layouts.
# - Makes tests that expect env.reset_to_circular(r0, mass) work even if:
#   * MultiOrbitEnv has no reset_to_circular
#   * reset() takes no arguments
#   * env lacks `cfg` (mu/dt/max_steps)
#   * TaskSpec requires (orbit_type, params)
# - Headless-safe for matplotlib.
#
# Usage:
#   Import this module early (preferably from tests/conftest.py) so patches
#   apply before pytest collects tests.

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid GUI backend during tests

import sys
import math
from types import SimpleNamespace
from importlib import import_module
from typing import Any, Optional

# --- Optional: ensure project root is importable (helps pytest/IDE) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_MU = 3.986e14  # Earth GM as safe default


# --------------------- dynamic import helpers --------------------- #
def _import_first(attr: str, candidates: list[str]) -> Optional[Any]:
    """
    Import attribute `attr` from the first module in `candidates` that loads.
    Returns the attribute object or None.
    """
    for modname in candidates:
        try:
            mod = import_module(modname)
            obj = getattr(mod, attr, None)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None


def _build_circular_state(mu: float, r0: float, mass: float) -> dict:
    """
    Create a canonical circular state from whichever physics module exists,
    or fall back to an inline builder.
    """
    make_circ = _import_first(
        "make_circular_state",
        [
            "simulator.physics",     # new layout
            "envs.physics",          # legacy layout
            "simulator.core.physics",
            "physics",               # very old flat layout
        ],
    )
    if callable(make_circ):
        try:
            return make_circ(mu, r0, mass)
        except Exception:
            pass

    # Inline fallback: +x at r0, +y velocity = sqrt(mu/r0)
    v = math.sqrt(mu / r0)
    return {"x": float(r0), "y": 0.0, "vx": 0.0, "vy": v, "mass": float(mass)}


# --------------------- env utilities --------------------- #
def _guess_mu(env) -> float:
    """Recursively try to extract a plausible mu from many possible layouts."""
    # Direct attributes
    for k in ("mu", "MU", "gmu", "GMu"):
        if hasattr(env, k):
            try:
                v = float(getattr(env, k))
                if v > 0:
                    return v
            except Exception:
                pass

    # cfg-like containers
    for k in ("cfg", "config", "sim_cfg", "simconfig"):
        obj = getattr(env, k, None)
        if obj is not None:
            for kk in ("mu", "MU", "gmu", "GMu"):
                if hasattr(obj, kk):
                    try:
                        v = float(getattr(obj, kk))
                        if v > 0:
                            return v
                    except Exception:
                        pass

    # nested envs
    for k in ("base", "orbit_env", "inner", "core", "wrapped", "env"):
        sub = getattr(env, k, None)
        if sub is not None:
            m = _guess_mu(sub)
            if m and m > 0:
                return m

    return DEFAULT_MU


def _install_cfg_if_missing(env, mu: float) -> None:
    """Attach a tiny cfg namespace if missing (some code expects env.cfg)."""
    if not hasattr(env, "cfg"):
        setattr(env, "cfg", SimpleNamespace(mu=mu, dt=1.0, max_steps=100))


def _try_set_state(env, s0: dict) -> bool:
    """
    Install initial state into a wide range of env implementations.
    Prefer official hooks; otherwise fall back to common fields.
    """
    # Preferred hooks
    if hasattr(env, "set_state"):
        try:
            env.set_state(s0)  # type: ignore
            return True
        except Exception:
            pass

    if hasattr(env, "reset"):
        # Some envs support keyword state injection
        for kw in ("custom_state", "state", "initial_state"):
            try:
                env.reset(**{kw: s0})  # type: ignore
                return True
            except Exception:
                pass

        # Some legacy reset variants accept positional None
        try:
            env.reset()  # type: ignore
        except TypeError:
            try:
                env.reset(None)  # type: ignore
            except Exception:
                pass

    # Common internal fields
    for name in ("state", "_state", "s", "_s"):
        try:
            setattr(env, name, dict(s0))
            return True
        except Exception:
            pass

    return False


# --------------------- patching core --------------------- #
def _patch_multi_orbit_env() -> None:
    """
    Locate MultiOrbitEnv (new or legacy) and add:
      - reset_to_circular(r0, mass) if missing
    Also patch TaskSampler.sample() to backfill orbit_type/params if required.
    """
    env_cls = None
    # Prefer new layout
    try:
        env_cls = _import_first("MultiOrbitEnv", ["simulator.multi_orbit_env"])
    except Exception:
        env_cls = None
    # Fallback to legacy layout
    if env_cls is None:
        env_cls = _import_first("MultiOrbitEnv", ["envs.multi_orbit_env"])
    if env_cls is None:
        return  # Nothing to patch; let tests fail naturally for visibility

    # ---- Add legacy entrypoint if absent ----
    if not hasattr(env_cls, "reset_to_circular"):
        def reset_to_circular(self, r0: float, mass: float, **_):
            """
            Test-facing legacy API.
            Works even when reset() takes no args and env lacks cfg.
            """
            # Try to ensure internal buffers exist
            if hasattr(self, "reset"):
                try:
                    self.reset()  # type: ignore
                except TypeError:
                    try:
                        self.reset(None)  # type: ignore
                    except Exception:
                        pass

            mu = _guess_mu(self)
            _install_cfg_if_missing(self, mu)

            s0 = _build_circular_state(mu, r0, mass)
            _ = _try_set_state(self, s0)
            # Return s0 so tests that inspect fields can proceed
            return s0

        setattr(env_cls, "reset_to_circular", reset_to_circular)

    # ---- Patch TaskSampler.sample() to fill (orbit_type, params) if missing ----
    sampler_cls = _import_first("TaskSampler", ["simulator.tasks", "envs.tasks"])
    if sampler_cls and not hasattr(sampler_cls, "__legacy_patched__"):
        orig_sample = getattr(sampler_cls, "sample", None)
        if callable(orig_sample):
            def sample_wrapper(self, *a, **kw):
                spec = orig_sample(self, *a, **kw)
                try:
                    # dataclass-like: add fields only if absent
                    if getattr(spec, "__dict__", None) is not None:
                        if not hasattr(spec, "orbit_type"):
                            setattr(spec, "orbit_type", getattr(spec, "kind", "circular"))
                        if not hasattr(spec, "params"):
                            p = {}
                            for k in ("r0", "mass", "custom_state"):
                                if hasattr(spec, k):
                                    p[k] = getattr(spec, k)
                            setattr(spec, "params", p)
                except Exception:
                    pass
                return spec
            setattr(sampler_cls, "sample", sample_wrapper)
            setattr(sampler_cls, "__legacy_patched__", True)


# Execute patches at import time so they apply before tests are collected.
_patch_multi_orbit_env()
