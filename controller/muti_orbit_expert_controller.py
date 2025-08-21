import numpy as np

def _unit(v, eps=1e-12):
    """Return (unit_vector, norm) with numerical safety."""
    n = np.linalg.norm(v)
    return (v / max(n, eps), max(n, eps))

class ExpertController:
    """
    Heuristic multi-mode controller for orbital tasks.

    Supported modes:
      - "spiral_in": gentle inward spiral toward target radius
      - "bangband": bang-bang band controller on radius ratio
      - "transfer": baseline transfer with near/far switch
      - "elliptic_circ": baseline eccentricity damping toward circular orbit
      - "elliptic_strong": stronger adaptive eccentricity damping (Day 37++)
      - "elliptic_ecc": aggressive eccentricity killer w/ apsis boost (NEW)
      - "transfer_2phase": two-phase transfer with pre-shaping + ecc kill (Day 37++)
    """
    def __init__(self,
                 mode: str = "spiral_in",
                 band=(1.25, 1.80),
                 fire_frac: float = 0.35,
                 bang_bang: bool = False,
                 circ_tol: float = 0.03):
        """
        Args:
            mode: Controller strategy name.
            band: Allowed r/rt band for the bang-bang mode.
            fire_frac: Throttle fraction in (0,1] when not bang-bang.
            bang_bang: If True, always use full a_max.
            circ_tol: |r/rt - 1| threshold to consider "near target".
        """
        self.mode = mode
        self.band = band
        self.fire_frac = fire_frac
        self.bang_bang = bang_bang
        self.circ_tol = circ_tol

        # --- internal state for phase / event detection ---
        self.step = 0
        self.last_vr_sign = 0.0
        self.apsis_boost_timer = 0
        self.apsis_boost_dir = 0.0
        self.phase1_timer = 0  # for transfer_2phase

    # ---------------- helpers ----------------
    def _geom(self, state, r_target):
        """
        Compute geometry from state and target radius.

        Returns:
            r: |pos|
            r_hat: radial unit vector
            t_hat: tangential unit vector (CCW)
            v_hat: velocity unit vector
            vmag: |vel|
            r_ratio: r / r_target
            v_r: radial component of velocity
            v_t: tangential component of velocity
        """
        pos = np.array(state[:2], dtype=np.float64)
        vel = np.array(state[2:], dtype=np.float64)
        r = np.linalg.norm(pos) + 1e-12
        r_hat, _ = _unit(pos)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
        v_hat, vmag = _unit(vel)
        r_ratio = r / float(r_target)
        v_r = float(np.dot(vel, r_hat))
        v_t = float(np.dot(vel, t_hat))
        return r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t

    def _throttle(self, mass, thrust):
        """Return commanded acceleration magnitude based on throttle mode."""
        a_max = float(thrust) / float(mass)
        return (a_max if self.bang_bang else (self.fire_frac * a_max))

    # ---------------- base modes ----------------
    def _spiral_in(self, state, task):
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        u = 0.85 * t_hat - 0.15 * (r_ratio - 1.0) * r_hat
        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    def _bangband(self, state, task):
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        in_band = (self.band[0] <= r_ratio <= self.band[1])
        u = (t_hat if in_band else -np.sign(r_ratio - 1.0) * r_hat)
        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    def _transfer(self, state, task):
        """
        Baseline transfer:
          - Far from target: push +/- tangential to change energy/mean radius.
          - Near target: damp radial velocity and align to tangential.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        near = abs(r_ratio - 1.0) < self.circ_tol

        if not near:
            sign = +1.0 if (r_ratio < 1.0) else -1.0
            u = 0.90 * (sign * t_hat) - 0.10 * (r_ratio - 1.0) * r_hat
        else:
            u = -0.6 * np.sign(v_r) * r_hat + 0.8 * t_hat

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    def _elliptic_circ(self, state, task):
        """
        Baseline eccentricity damping toward a circular orbit at r_target.
        Combines tangential energy shaping with a fixed radial kill term.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        near = abs(r_ratio - 1.0) < self.circ_tol

        if not near:
            sign = +1.0 if (r_ratio < 1.0) else -1.0
            base = 0.75 * (sign * t_hat) - 0.25 * (r_ratio - 1.0) * r_hat
        else:
            base = 0.9 * t_hat

        damp = -np.sign(v_r) * r_hat
        u = base + 0.6 * damp
        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    def _elliptic_strong(self, state, task):
        """
        Stronger adaptive eccentricity damping.
        - Keep tangential energy shaping.
        - Add adaptive radial kill weight based on |r/rt-1| and |v_r|/|v|.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        near = abs(r_ratio - 1.0) < self.circ_tol

        if not near:
            base = 0.80 * ((+1.0 if (r_ratio < 1.0) else -1.0) * t_hat) \
                 - 0.20 * (r_ratio - 1.0) * r_hat
        else:
            base = 0.90 * t_hat

        err = abs(r_ratio - 1.0)
        speed = max(vmag, 1e-9)
        vr_norm = min(1.0, abs(v_r) / speed)
        w_rad = 1.2 + 0.8 * min(1.0, err) + 0.4 * vr_norm  # up to ~2.4

        damp = -np.sign(v_r) * r_hat
        u = base + w_rad * damp

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    # ---------------- NEW: aggressive eccentricity killer ----------------
    def _elliptic_ecc(self, state, task):
        """
        Aggressive eccentricity killer with apsis boosting.

        Key ideas:
          (1) Very strong adaptive radial kill to zero-out v_r quickly.
          (2) When close to an apsis (|v_r| small), lock a short tangential boost
              to reshape the opposite apsis, reducing eccentricity fast.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        err = abs(r_ratio - 1.0)
        speed = max(vmag, 1e-9)
        vr_norm = min(1.0, abs(v_r) / speed)

        # --- (1) very strong radial kill, scaled by error and |v_r| ---
        # Range about [3, 8]; larger when far from target and v_r is large.
        w_rad = 3.0 + 4.0 * min(1.0, err) + 1.0 * vr_norm

        # --- base tangential energy shaping to pull radius toward target ---
        base = 0.55 * ((+1.0 if (r_ratio < 1.0) else -1.0) * t_hat) \
             - 0.15 * (r_ratio - 1.0) * r_hat

        # --- apsis detection: near apsis when |v_r| is tiny ---
        vr_sign = float(np.sign(v_r))
        near_apsis = (abs(v_r) < 0.02 * speed)

        # Start/refresh a short tangential boost window when we detect apsis.
        if near_apsis and (self.apsis_boost_timer <= 0):
            # prograde if r < r_t (to increase speed), retrograde if r > r_t
            self.apsis_boost_dir = (+1.0 if (r_ratio < 1.0) else -1.0)
            # longer boost when error is large
            self.apsis_boost_timer = int(120 + 280 * min(1.0, err))

        # Count down the boost window.
        boost = 0.0
        if self.apsis_boost_timer > 0:
            boost = 0.35 * self.apsis_boost_dir  # moderate tangential push
            self.apsis_boost_timer -= 1

        damp = -np.sign(v_r) * r_hat
        u = base + w_rad * damp + boost * t_hat

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        self.last_vr_sign = vr_sign
        self.step += 1
        return a * u

    # ---------------- UPDATED: two-phase transfer ----------------
    def _transfer_2phase(self, state, task):
        """
        Two-phase transfer with memory:
          Phase 1: fixed-duration strong tangential push with radial pre-shaping.
          Phase 2: switch to aggressive eccentricity killer.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        err = abs(r_ratio - 1.0)
        speed = max(vmag, 1e-9)

        # Initialize phase-1 timer on the very first step or when timer is zero.
        if self.phase1_timer == 0 and self.step == 0:
            # Longer when far away; capped for safety.
            self.phase1_timer = int(200 + 800 * min(1.0, err))

        # Decide if we leave phase-1 early by hitting the near band.
        near_band = (err < self.circ_tol)

        if (self.phase1_timer > 0) and (not near_band):
            # --- Phase 1: energy change with radial pre-shaping ---
            sign = +1.0 if (r_ratio < 1.0) else -1.0
            u = ( 1.00 * sign * t_hat
                - 0.30 * (r_ratio - 1.0) * r_hat
                - 0.70 * np.sign(v_r) * r_hat )
            self.phase1_timer -= 1
        else:
            # --- Phase 2: aggressive eccentricity kill ---
            return self._elliptic_ecc(state, task)

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        self.step += 1
        return a * u

    # ---------------- public API ----------------
    def act(self, state, task):
        """
        Compute acceleration command.

        Returns:
            accel_cmd: np.array([ax, ay]) in m/s^2
            diag: dict with diagnostic values
        """
        if self.mode == "spiral_in":
            acc = self._spiral_in(state, task)
        elif self.mode == "bangband":
            acc = self._bangband(state, task)
        elif self.mode == "transfer":
            acc = self._transfer(state, task)
        elif self.mode == "elliptic_circ":
            acc = self._elliptic_circ(state, task)
        elif self.mode == "transfer_2phase":
            acc = self._transfer_2phase(state, task)
        elif self.mode == "elliptic_strong":
            acc = self._elliptic_strong(state, task)
        elif self.mode == "elliptic_ecc":
            acc = self._elliptic_ecc(state, task)
        else:
            acc = self._spiral_in(state, task)

        # diagnostics
        r, _, t_hat, _, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        return acc.astype(np.float64), {
            "mode": self.mode,
            "r_ratio": float(r_ratio),
            "v_r": float(v_r),
            "v_t": float(v_t),
            "step": int(self.step),
            "phase1_timer": int(self.phase1_timer),
            "apsis_boost_timer": int(self.apsis_boost_timer),
        }
