import numpy as np

class GreedyRTv2Controller:
    """
    Greedy baseline with PD radial correction + tangential speed shaping.
    Action space: normalized thrust in [-1, 1]^2.
    Design goals:
      - Reduce radius error with damping (PD on radial channel).
      - Drive tangential velocity toward +1 * v_circ.
      - Limit overall thrust magnitude (fuel-aware).
      - Auto-throttle (deadzone) when near the success band.
    """

    def __init__(self,
                 k_rp=0.35,   # radial P-gain on (r_norm - 1)
                 k_rd=0.60,   # radial D-gain on v_r (radial velocity)
                 k_t=1.00,    # tangential gain on dv_t = 1 - v_t
                 t_clip=0.8,  # clamp dv_t before scaling
                 a_max=0.60,  # global action magnitude cap (<= 1.0)
                 dead_r=0.02, # deadzone for |r_norm - 1| (matches relaxed success band)
                 dead_v=0.04  # deadzone for |dv_t|
                 ):
        self.k_rp = k_rp
        self.k_rd = k_rd
        self.k_t = k_t
        self.t_clip = t_clip
        self.a_max = a_max
        self.dead_r = dead_r
        self.dead_v = dead_v

    def act(self, obs):
        """
        obs layout (10 dims):
        [ x/rt, y/rt, vx/v_circ, vy/v_circ, e, thrust_limit,
          rt/1e12, e, mass/base, thrust_limit ]
        Returns:
            action (np.ndarray): normalized thrust vector in [-1, 1]^2
        """
        x_r, y_r = obs[0], obs[1]
        vx_n, vy_n = obs[2], obs[3]

        # radial unit in normalized coordinates
        r_vec = np.array([x_r, y_r], dtype=np.float64)
        r_norm = np.linalg.norm(r_vec) + 1e-9
        r_hat = r_vec / r_norm

        # radial velocity component (normalized by v_circ)
        v_vec_n = np.array([vx_n, vy_n], dtype=np.float64)
        v_r = float(np.dot(v_vec_n, r_hat))

        # sign of (r - rt) ~ (||[x/rt, y/rt]|| - 1)
        r_err_sign = np.sign(r_norm - 1.0)

        # PD radial correction
        a_radial = - self.k_rp * r_err_sign * r_hat - self.k_rd * v_r * r_hat

        # tangential unit and correction
        t_hat = np.array([-y_r, x_r], dtype=np.float64)
        t_hat /= (np.linalg.norm(t_hat) + 1e-9)

        # current tangential velocity (already normalized by v_circ)
        v_t = float(np.dot(v_vec_n, t_hat))

        # desire v_t -> +1 (aligned and magnitude ~ v_circ)
        dv_t = np.clip(1.0 - v_t, -self.t_clip, self.t_clip)
        a_tangential = self.k_t * dv_t * t_hat

        # deadzone gating (save fuel near target band)
        if abs(r_norm - 1.0) < self.dead_r and abs(1.0 - v_t) < self.dead_v:
            return np.array([0.0, 0.0], dtype=np.float64)

        # combine and soft-limit overall magnitude
        a = a_radial + a_tangential
        norm = np.linalg.norm(a)
        if norm > 1e-9:
            scale = min(1.0, self.a_max / norm)
            a = a * scale

        return np.clip(a, -1.0, 1.0)
