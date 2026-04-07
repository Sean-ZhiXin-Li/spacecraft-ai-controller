import numpy as np


class StableOrbitController:
    """
    Lightweight physics-prior controller that outputs normalized action in [-1, 1]^2.

    Design goals:
    - Approach target radius with radial PD behavior.
    - Match tangential speed to local circular speed.
    - Reduce thrust near the target band to avoid limit-cycle jitter.
    """

    def __init__(
        self,
        target_radius: float,
        mu: float = 6.67430e-11 * 1.989e30,
        k_r: float = 2.0,
        k_vr: float = 1.5,
        k_t: float = 1.2,
        action_cap: float = 0.8,
    ) -> None:
        self.target_radius = float(target_radius)
        self.mu = float(mu)
        self.k_r = float(k_r)
        self.k_vr = float(k_vr)
        self.k_t = float(k_t)
        self.action_cap = float(action_cap)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        pos = obs[:2].astype(np.float64)
        vel = obs[2:4].astype(np.float64)
        return self._compute_action(pos, vel)

    def __call__(self, t, pos, vel):
        _ = t
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        return self._compute_action(pos, vel)

    def _compute_action(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(pos) + 1e-12
        u_r = pos / r
        u_t = np.array([-u_r[1], u_r[0]], dtype=np.float64)

        v_r = float(np.dot(vel, u_r))
        v_t = float(np.dot(vel, u_t))
        v_c = float(np.sqrt(self.mu / self.target_radius))

        e_r = float((r - self.target_radius) / (self.target_radius + 1e-12))
        e_t = float((v_c - v_t) / (v_c + 1e-12))
        e_vr = float(v_r / (v_c + 1e-12))

        a_r = -(self.k_r * e_r + self.k_vr * e_vr)
        a_t = self.k_t * e_t

        a_r = float(np.clip(a_r, -self.action_cap, self.action_cap))
        a_t = float(np.clip(a_t, -self.action_cap, self.action_cap))

        if abs(e_r) < 0.03 and abs(e_t) < 0.03 and abs(e_vr) < 0.03:
            a_r *= 0.2
            a_t *= 0.2

        action = a_r * u_r + a_t * u_t
        return np.clip(action, -1.0, 1.0).astype(np.float32)
