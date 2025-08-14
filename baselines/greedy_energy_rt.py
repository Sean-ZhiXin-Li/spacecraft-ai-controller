import numpy as np

class GreedyEnergyRTController:
    """
    Energy-shaping baseline (v1.3, stronger fuel saving):
      - v_t_des = 1 - k_e * (r_norm - 1) with mild saturation.
      - Light PD on radial channel (damping).
      - Proximity-aware throttle (strong nonlinearity) + hysteresis deadzone.
      - Direction-safety: coast if tangential correction would increase |r_err|.
      - Tighter tangential clamp and lower action caps to reduce fuel.
    """

    def __init__(self,
                 k_e=0.9,         # energy shaping gain
                 k_rp=0.10,       # small radial P gain
                 k_rd=0.40,       # radial D gain
                 t_clip=0.5,      # tighter tangential correction
                 a_max_lo=0.08,   # smaller cap near target
                 a_max_hi=0.50,   # lower far cap
                 # hysteresis deadzone: enter vs exit thresholds
                 dead_r_in=0.035, dead_r_out=0.028,
                 dead_v_in=0.070, dead_v_out=0.055,
                 v_des_min=0.80,  # mild saturation of desired tangential speed
                 v_des_max=1.20):
        self.k_e = k_e
        self.k_rp = k_rp
        self.k_rd = k_rd
        self.t_clip = t_clip
        self.a_max_lo = a_max_lo
        self.a_max_hi = a_max_hi
        self.dead_r_in = dead_r_in
        self.dead_r_out = dead_r_out
        self.dead_v_in = dead_v_in
        self.dead_v_out = dead_v_out
        self.v_des_min = v_des_min
        self.v_des_max = v_des_max
        self._in_deadzone = False  # hysteresis latch

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v) + 1e-9
        return v / n, n

    @staticmethod
    def _smoothstep2(x):
        """Squared smoothstep: (3x^2 - 2x^3)^2 on [0,1]."""
        x = np.clip(x, 0.0, 1.0)
        s = x*x*(3.0 - 2.0*x)
        return s*s

    def _proximity_cap(self, r_abs, v_t_abs):
        """
        Strong proximity shaping:
          - Map errors to [0,1] vs. far caps (heuristic 0.20 radius, 0.40 tangential).
          - Use squared smoothstep to aggressively throttle near the target.
        """
        pr = np.clip((r_abs   - self.dead_r_out) / max(1e-6, (0.20 - self.dead_r_out)), 0.0, 1.0)
        pv = np.clip((v_t_abs - self.dead_v_out) / max(1e-6, (0.40 - self.dead_v_out)), 0.0, 1.0)
        prox = max(pr, pv)
        s = self._smoothstep2(prox)
        return self.a_max_lo + (self.a_max_hi - self.a_max_lo) * s

    def act(self, obs):
        # Unpack normalized observation
        x_r, y_r = obs[0], obs[1]
        vx_n, vy_n = obs[2], obs[3]

        # Geometry
        r_vec = np.array([x_r, y_r], dtype=np.float64)
        r_hat, r_norm = self._unit(r_vec)
        t_hat, _      = self._unit(np.array([-y_r, x_r], dtype=np.float64))
        v_vec_n       = np.array([vx_n, vy_n], dtype=np.float64)

        # Components
        v_r = float(np.dot(v_vec_n, r_hat))     # radial velocity (normalized)
        v_t = float(np.dot(v_vec_n, t_hat))     # tangential velocity (normalized)
        r_err = r_norm - 1.0
        r_abs = abs(r_err)
        v_t_err = 1.0 - v_t
        v_t_abs = abs(v_t_err)

        # hysteresis deadzone (coast around success band)
        if self._in_deadzone:
            if (r_abs < self.dead_r_out) and (v_t_abs < self.dead_v_out):
                return np.array([0.0, 0.0], dtype=np.float64)
            else:
                self._in_deadzone = False
        else:
            if (r_abs < self.dead_r_in) and (v_t_abs < self.dead_v_in):
                self._in_deadzone = True
                return np.array([0.0, 0.0], dtype=np.float64)

        # energy shaping target for tangential speed (with mild saturation)
        v_t_des = 1.0 - self.k_e * r_err
        v_t_des = np.clip(v_t_des, self.v_des_min, self.v_des_max)
        dv_t = np.clip(v_t_des - v_t, -self.t_clip, self.t_clip)

        # Direction safety: if dv_t would increase |r_err|, coast
        if (r_err * dv_t) > 0.0:
            return np.array([0.0, 0.0], dtype=np.float64)

        a_tangential = dv_t * t_hat

        # light PD on radial channel (damping)
        a_radial = - self.k_rp * np.sign(r_err) * r_hat - self.k_rd * v_r * r_hat

        # Combine
        a = a_radial + a_tangential

        # Proximity-aware throttle
        a_cap = self._proximity_cap(r_abs, v_t_abs)
        norm = np.linalg.norm(a)
        if norm > 1e-9:
            a *= min(1.0, a_cap / norm)

        return np.clip(a, -1.0, 1.0)