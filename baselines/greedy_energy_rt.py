import numpy as np

class GreedyEnergyRTController:
    """
    Energy-shaping baseline (v1.4b-trim):
      - v1.4: far & hard tasks get mild cap relaxation; near-target stays fuel-conservative.
      - v1.4b: far & hard also mildly relax tangential clamp (t_clip).
      - trim edition: smaller relax: t_clip ×1.12 (was 1.15), caps ×(1.08, 1.16) (was 1.10, 1.20).
      - Outputs Cartesian action clipped to [-1, 1]^2.
    """

    def __init__(self,
                 k_e=0.9,
                 k_rp=0.10,
                 k_rd=0.40,
                 t_clip=0.5,
                 a_max_lo=0.08,
                 a_max_hi=0.50,
                 dead_r_in=0.035, dead_r_out=0.028,
                 dead_v_in=0.070, dead_v_out=0.055,
                 v_des_min=0.80, v_des_max=1.20):
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
        self._in_deadzone = False
        self._task = None
        self._relax_hits = 0  # telemetry

        # ---- trim multipliers (only applied when far & hard) ----
        self.relax_tclip = 1.12     # was 1.15
        self.relax_cap_lo = 1.08    # was 1.10
        self.relax_cap_hi = 1.16    # was 1.20

    def set_task(self, task_dict):
        self._task = dict(task_dict) if task_dict is not None else None

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v) + 1e-9
        return v / n, n

    @staticmethod
    def _smoothstep2(x):
        x = np.clip(x, 0.0, 1.0)
        s = x * x * (3.0 - 2.0 * x)
        return s * s

    def _proximity_cap(self, r_abs, v_t_abs):
        pr = np.clip((r_abs - self.dead_r_out) / max(1e-6, (0.20 - self.dead_r_out)), 0.0, 1.0)
        pv = np.clip((v_t_abs - self.dead_v_out) / max(1e-6, (0.40 - self.dead_v_out)), 0.0, 1.0)
        prox = max(pr, pv)
        s = self._smoothstep2(prox)
        return self.a_max_lo + (self.a_max_hi - self.a_max_lo) * s

    def _is_hard_task(self):
        if self._task is None:
            return False
        rt = float(self._task.get("target_radius", 0.0))
        e  = float(self._task.get("e", 0.0))
        return (rt > 4.0e12) or (e > 0.25)

    def _caps_with_v14(self, r_norm):
        a_lo, a_hi = self.a_max_lo, self.a_max_hi
        if (r_norm > 1.10) and self._is_hard_task():
            self._relax_hits += 1
            return a_lo * self.relax_cap_lo, a_hi * self.relax_cap_hi
        else:
            return a_lo, a_hi

    def act(self, obs):
        # [x_r, y_r, vx_n, vy_n, ...]
        x_r, y_r = obs[0], obs[1]
        vx_n, vy_n = obs[2], obs[3]

        r_vec = np.array([x_r, y_r], dtype=np.float64)
        r_hat, r_norm = self._unit(r_vec)
        t_hat, _ = self._unit(np.array([-y_r, x_r], dtype=np.float64))
        v_vec_n = np.array([vx_n, vy_n], dtype=np.float64)

        v_r = float(np.dot(v_vec_n, r_hat))
        v_t = float(np.dot(v_vec_n, t_hat))
        r_err = r_norm - 1.0
        r_abs = abs(r_err)
        v_t_err = 1.0 - v_t
        v_t_abs = abs(v_t_err)

        # Deadzone (hysteresis)
        if self._in_deadzone:
            if (r_abs < self.dead_r_out) and (v_t_abs < self.dead_v_out):
                return np.array([0.0, 0.0], dtype=np.float64)
            else:
                self._in_deadzone = False
        else:
            if (r_abs < self.dead_r_in) and (v_t_abs < self.dead_v_in):
                self._in_deadzone = True
                return np.array([0.0, 0.0], dtype=np.float64)

        # Energy-shaping target for tangential speed
        v_t_des = 1.0 - self.k_e * r_err
        v_t_des = np.clip(v_t_des, self.v_des_min, self.v_des_max)

        # v1.4b-trim: relax t_clip only when far & hard
        t_clip_eff = self.t_clip * (self.relax_tclip if ((r_norm > 1.10) and self._is_hard_task()) else 1.0)
        dv_t = np.clip(v_t_des - v_t, -t_clip_eff, t_clip_eff)

        # Direction safety near target or when not hard
        apply_dir_safety = (r_norm < 1.06) or (not self._is_hard_task())
        if apply_dir_safety and (r_err * dv_t) > 0.0:
            return np.array([0.0, 0.0], dtype=np.float64)

        a_tangential = dv_t * t_hat
        a_radial = - self.k_rp * np.sign(r_err) * r_hat - self.k_rd * v_r * r_hat
        a = a_radial + a_tangential

        base_cap = self._proximity_cap(r_abs, v_t_abs)
        a_lo_new, a_hi_new = self._caps_with_v14(r_norm)
        if (self.a_max_hi - self.a_max_lo) > 1e-9:
            tau = (base_cap - self.a_max_lo) / (self.a_max_hi - self.a_max_lo)
            a_cap = a_lo_new + tau * (a_hi_new - a_lo_new)
        else:
            a_cap = base_cap

        norm = np.linalg.norm(a)
        if norm > 1e-9:
            a *= min(1.0, a_cap / norm)

        return np.clip(a, -1.0, 1.0)
