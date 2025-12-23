# utils/action_interface.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ActionInterfaceConfig:
    mode: str = "raw"  # "raw" | "prescale"
    eps: float = 1e-12


def thrust_to_action(
    thrust_intent_N: np.ndarray,
    thrust_scale_N: float,
    cfg: ActionInterfaceConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Map thrust intent in Newtons to env action in [-1, 1]^2.

    raw:
      action = clip(thrust/thrust_scale, -1, 1)

    prescale:
      vector-level prescale on normalized intent u = thrust/thrust_scale
      to preserve direction/ratio before clipping.
    """
    thrust = np.asarray(thrust_intent_N, dtype=np.float64)
    scale = max(cfg.eps, float(thrust_scale_N))
    u = thrust / scale  # normalized intent

    if cfg.mode == "raw":
        a = np.clip(u, -1.0, 1.0)

    elif cfg.mode == "prescale":
        max_abs = float(np.max(np.abs(u)))
        if max_abs > 1.0:
            u = u / (max_abs + cfg.eps)  # vector-level scaling
        a = np.clip(u, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown action interface mode: {cfg.mode}")

    sat = np.isclose(a, -1.0) | np.isclose(a, 1.0)
    metrics = {
        "action_interface_mode": cfg.mode,
        "saturation_rate": float(np.mean(sat)),
        "intent_normed_max_abs": float(np.max(np.abs(thrust / scale))),
    }
    return a.astype(np.float32), metrics
