from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Sequence


def _norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_vt(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = _norm2(pos[0], pos[1])
    if r < 1e-12:
        return 0.0
    r_hat_x = pos[0] / r
    r_hat_y = pos[1] / r
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    return vel[0] * t_hat_x + vel[1] * t_hat_y


@dataclass
class AntiShutdownConfig:
    low_action_norm_threshold: float = 0.02
    unresolved_r_threshold_ratio: float = 0.002
    unresolved_vt_threshold_ratio: float = 0.01
    quiet_vr_threshold_ratio: float = 0.02
    radial_rescue: float = 0.03
    tangential_rescue: float = 0.02

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class PPOAntiShutdownWrapper:
    """Minimal rescue layer that only intervenes inside the low-action shutdown band."""

    def __init__(self, base_policy, *, target_radius: float, mu: float, config: AntiShutdownConfig | None = None) -> None:
        self.base_policy = base_policy
        self.target_radius = float(target_radius)
        self.mu = float(mu)
        self.config = config or AntiShutdownConfig()
        self.v_circ = math.sqrt(self.mu / self.target_radius)
        self.last_info: Dict[str, float | bool | list[float]] = {}

    def _extract_errors(self, obs: Sequence[float]) -> Dict[str, float]:
        pos = [float(obs[0]), float(obs[1])]
        vel = [float(obs[2]), float(obs[3])]
        radius = _norm2(pos[0], pos[1])
        v_r = float(obs[4]) if len(obs) >= 5 else 0.0
        v_t = _compute_vt(pos, vel)
        return {
            "radius": radius,
            "r_error": radius - self.target_radius,
            "v_r": v_r,
            "v_t_error": v_t - self.v_circ,
        }

    def act(self, obs: Sequence[float]) -> list[float]:
        return self.act_with_info(obs)["final_action"]

    def act_with_info(self, obs: Sequence[float]) -> Dict[str, float | bool | list[float]]:
        raw_action = self.base_policy.act(obs)
        raw_action = [float(raw_action[0]), float(raw_action[1])]
        raw_norm = _norm2(raw_action[0], raw_action[1])

        errors = self._extract_errors(obs)
        r_error = float(errors["r_error"])
        v_r = float(errors["v_r"])
        v_t_error = float(errors["v_t_error"])

        unresolved_r = abs(r_error) > self.config.unresolved_r_threshold_ratio * self.target_radius
        unresolved_vt = abs(v_t_error) > self.config.unresolved_vt_threshold_ratio * self.v_circ
        quiet_vr = abs(v_r) < self.config.quiet_vr_threshold_ratio * self.v_circ
        low_action = raw_norm < self.config.low_action_norm_threshold
        intervene = bool(low_action and quiet_vr and (unresolved_r or unresolved_vt))

        rescue_r = 0.0
        rescue_t = 0.0
        if intervene:
            if unresolved_r:
                if abs(v_r) > 1e-6:
                    rescue_r = -self.config.radial_rescue * math.copysign(1.0, v_r)
                else:
                    rescue_r = -self.config.radial_rescue * math.copysign(1.0, r_error)
            if unresolved_vt:
                rescue_t = -self.config.tangential_rescue * math.copysign(1.0, v_t_error)

        pos = [float(obs[0]), float(obs[1])]
        radius = _norm2(pos[0], pos[1]) + 1e-12
        r_hat_x = pos[0] / radius
        r_hat_y = pos[1] / radius
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
        rescue_x = rescue_r * r_hat_x + rescue_t * t_hat_x
        rescue_y = rescue_r * r_hat_y + rescue_t * t_hat_y

        final_action = [
            _clamp(raw_action[0] + rescue_x, -1.0, 1.0),
            _clamp(raw_action[1] + rescue_y, -1.0, 1.0),
        ]
        final_norm = _norm2(final_action[0], final_action[1])

        self.last_info = {
            "intervene": intervene,
            "raw_action": raw_action,
            "final_action": final_action,
            "raw_action_norm": raw_norm,
            "final_action_norm": final_norm,
            "rescue_action": [rescue_x, rescue_y],
            "rescue_action_norm": _norm2(rescue_x, rescue_y),
            "r_error": r_error,
            "v_r": v_r,
            "v_t_error": v_t_error,
            "unresolved_r": unresolved_r,
            "unresolved_vt": unresolved_vt,
            "quiet_vr": quiet_vr,
            "low_action": low_action,
        }
        return self.last_info
