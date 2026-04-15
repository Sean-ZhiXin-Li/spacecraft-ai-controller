from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Sequence


def _norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_vr(pos: Sequence[float], vel: Sequence[float]) -> float:
    r = _norm2(pos[0], pos[1])
    if r < 1e-12:
        return 0.0
    return (pos[0] * vel[0] + pos[1] * vel[1]) / r


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
class OrbitLockConfig:
    descent_tangential_cmd: float = 1.0
    capture_radial_pos_gain: float = 3.0
    capture_radial_vel_gain: float = 4.0
    capture_tangential_gain: float = 1.2
    capture_radial_limit: float = 0.40
    capture_tangential_limit: float = 0.40
    lock_radial_pos_gain: float = 1.0
    lock_radial_vel_gain: float = 1.8
    lock_tangential_gain: float = 0.8
    lock_radial_limit: float = 0.10
    lock_tangential_limit: float = 0.10
    lock_r_threshold_ratio: float = 3.0e-4
    lock_vr_threshold_ratio: float = 3.0e-3
    lock_vt_threshold_ratio: float = 4.0e-3
    unlock_r_threshold_ratio: float = 1.0e-3
    unlock_vr_threshold_ratio: float = 1.0e-2

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class OrbitLockController:
    """Phase-based orbit insertion prototype with explicit descent, capture, and lock states."""

    STATE_DESCENT = "DESCENT"
    STATE_CAPTURE = "CAPTURE"
    STATE_LOCK = "LOCK"

    def __init__(self, *, target_radius: float, mu: float, config: OrbitLockConfig | None = None) -> None:
        self.target_radius = float(target_radius)
        self.mu = float(mu)
        self.v_circ = math.sqrt(self.mu / self.target_radius)
        self.config = config or OrbitLockConfig()
        self.phase = self.STATE_DESCENT
        self.prev_real_r_error: float | None = None
        self.phase_transitions: list[tuple[str, str, float, float, float]] = []
        self.last_info: Dict[str, float | bool | list[float] | str] = {}

    def _set_phase(self, new_phase: str, real_r_error: float, v_r: float, v_t_error: float) -> None:
        if new_phase == self.phase:
            return
        print(f"PHASE TRANSITION {self.phase} -> {new_phase} r_error={real_r_error} v_r={v_r} v_t_error={v_t_error}")
        self.phase_transitions.append((self.phase, new_phase, real_r_error, v_r, v_t_error))
        self.phase = new_phase

    def act(self, obs: Sequence[float]) -> list[float]:
        return self.act_with_info(obs)["final_action"]

    def act_with_info(self, obs: Sequence[float]) -> Dict[str, float | bool | list[float] | str]:
        pos = [float(obs[0]), float(obs[1])]
        vel = [float(obs[2]), float(obs[3])]
        radius = _norm2(pos[0], pos[1]) + 1e-12
        real_r_error = radius - self.target_radius
        v_r = float(obs[4]) if len(obs) >= 5 else _compute_vr(pos, vel)
        v_t_error = _compute_vt(pos, vel) - self.v_circ

        real_r_ratio = real_r_error / self.target_radius
        v_r_ratio = v_r / self.v_circ
        v_t_ratio = v_t_error / self.v_circ

        crossed_now = False
        if self.prev_real_r_error is not None:
            crossed_now = (self.prev_real_r_error > 0.0 and real_r_error <= 0.0) or (
                self.prev_real_r_error < 0.0 and real_r_error >= 0.0
            )
        self.prev_real_r_error = real_r_error

        if self.phase == self.STATE_DESCENT and crossed_now:
            self._set_phase(self.STATE_CAPTURE, real_r_error, v_r, v_t_error)
        elif self.phase == self.STATE_CAPTURE:
            if (
                abs(real_r_ratio) < self.config.lock_r_threshold_ratio
                and abs(v_r_ratio) < self.config.lock_vr_threshold_ratio
                and abs(v_t_ratio) < self.config.lock_vt_threshold_ratio
            ):
                self._set_phase(self.STATE_LOCK, real_r_error, v_r, v_t_error)
        elif self.phase == self.STATE_LOCK:
            if (
                abs(real_r_ratio) > self.config.unlock_r_threshold_ratio
                or abs(v_r_ratio) > self.config.unlock_vr_threshold_ratio
            ):
                self._set_phase(self.STATE_CAPTURE, real_r_error, v_r, v_t_error)

        radial_cmd = 0.0
        tangential_cmd = 0.0
        if self.phase == self.STATE_DESCENT:
            v_norm = _norm2(vel[0], vel[1])
            if v_norm > 1e-12:
                final_action = [
                    _clamp(-vel[0] / v_norm, -1.0, 1.0),
                    _clamp(-vel[1] / v_norm, -1.0, 1.0),
                ]
            else:
                final_action = [0.0, 0.0]
            self.last_info = {
                "final_action": final_action,
                "phase": self.phase,
                "crossed_now": crossed_now,
                "radial_cmd": 0.0,
                "tangential_cmd": -1.0,
                "action_norm": _norm2(final_action[0], final_action[1]),
                "real_r_error": real_r_error,
                "v_r": v_r,
                "v_t_error": v_t_error,
                "phase_transition_count": len(self.phase_transitions),
            }
            return self.last_info
        elif self.phase == self.STATE_CAPTURE:
            radial_cmd = -(
                self.config.capture_radial_pos_gain * real_r_ratio
                + self.config.capture_radial_vel_gain * v_r_ratio
            )
            tangential_cmd = -(self.config.capture_tangential_gain * v_t_ratio)
            radial_cmd = _clamp(radial_cmd, -self.config.capture_radial_limit, self.config.capture_radial_limit)
            tangential_cmd = _clamp(
                tangential_cmd,
                -self.config.capture_tangential_limit,
                self.config.capture_tangential_limit,
            )
        else:
            radial_cmd = -(
                self.config.lock_radial_pos_gain * real_r_ratio
                + self.config.lock_radial_vel_gain * v_r_ratio
            )
            tangential_cmd = -(self.config.lock_tangential_gain * v_t_ratio)
            radial_cmd = _clamp(radial_cmd, -self.config.lock_radial_limit, self.config.lock_radial_limit)
            tangential_cmd = _clamp(
                tangential_cmd,
                -self.config.lock_tangential_limit,
                self.config.lock_tangential_limit,
            )

        r_hat_x = pos[0] / radius
        r_hat_y = pos[1] / radius
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
        action_x = radial_cmd * r_hat_x + tangential_cmd * t_hat_x
        action_y = radial_cmd * r_hat_y + tangential_cmd * t_hat_y
        final_action = [
            _clamp(action_x, -1.0, 1.0),
            _clamp(action_y, -1.0, 1.0),
        ]

        self.last_info = {
            "final_action": final_action,
            "phase": self.phase,
            "crossed_now": crossed_now,
            "radial_cmd": radial_cmd,
            "tangential_cmd": tangential_cmd,
            "action_norm": _norm2(final_action[0], final_action[1]),
            "real_r_error": real_r_error,
            "v_r": v_r,
            "v_t_error": v_t_error,
            "phase_transition_count": len(self.phase_transitions),
        }
        return self.last_info
