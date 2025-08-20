import numpy as np

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return (v / max(n, eps), max(n, eps))

class ExpertController:
    def __init__(self,
                 mode: str = "spiral_in",
                 band=(1.25, 1.80),
                 fire_frac: float = 0.35,
                 bang_bang: bool = False,
                 circ_tol: float = 0.03):
        """
        Args:
            mode: controller strategy.
            band: radius band for bang-bang logic (as r/r_target).
            fire_frac: throttle fraction in (0,1] if not bang_bang.
            bang_bang: if True, always full throttle magnitude (1.0 * thrust/mass).
            circ_tol: "near target" band to trigger circularization phase.
        """
        self.mode = mode
        self.band = band
        self.fire_frac = fire_frac
        self.bang_bang = bang_bang
        self.circ_tol = circ_tol

    # helpers
    def _geom(self, state, r_target):
        pos = np.array(state[:2], dtype=np.float64)
        vel = np.array(state[2:], dtype=np.float64)
        r = np.linalg.norm(pos) + 1e-12
        r_hat, _ = _unit(pos)
        # tangential unit vector (CCW)
        t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
        v_hat, vmag = _unit(vel)
        r_ratio = r / float(r_target)
        # radial velocity sign and magnitude
        v_r = float(np.dot(vel, r_hat))
        v_t = float(np.dot(vel, t_hat))
        return r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t

    def _throttle(self, mass, thrust):
        a_max = float(thrust) / float(mass)
        return (a_max if self.bang_bang else (self.fire_frac * a_max))

    # modes
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
        Coarse Hohmann-like:
          Phase A (energy shaping): push tangential prograde if r < r_target, retrograde if r > r_target.
          Phase B (circularize near target): damp radial velocity and align velocity with tangential.
        No mu needed; uses geometric heuristics.
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        near = abs(r_ratio - 1.0) < self.circ_tol

        if not near:
            # Energy shaping: push along +/- t_hat to raise/lower semi-major axis
            sign = +1.0 if (r_ratio < 1.0) else -1.0
            # Add a small radial term to bias toward target radius
            u = 0.90 * (sign * t_hat) - 0.10 * (r_ratio - 1.0) * r_hat
        else:
            # Circularization: kill radial velocity and align with tangential
            # PD-like mix: oppose radial vel + align velocity with t_hat
            u = -0.6 * np.sign(v_r) * r_hat + 0.8 * t_hat

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    def _elliptic_circ(self, state, task):
        """
        Eccentricity damping toward a circular orbit at r_target.
        Heuristic:
          - If outside target band: push +/- t_hat to move average radius toward target.
          - Always add a term to kill radial velocity (reduce e).
        """
        r, r_hat, t_hat, v_hat, vmag, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        near = abs(r_ratio - 1.0) < self.circ_tol

        if not near:
            sign = +1.0 if (r_ratio < 1.0) else -1.0
            base = 0.75 * (sign * t_hat) - 0.25 * (r_ratio - 1.0) * r_hat
        else:
            base = 0.9 * t_hat  # once near target, mostly tangential

        # Dampen radial velocity component to reduce eccentricity
        damp = -np.sign(v_r) * r_hat
        u = base + 0.6 * damp

        u, _ = _unit(u)
        a = self._throttle(task.mass, task.thrust_newton)
        return a * u

    # public API
    def act(self, state, task):
        """
        Returns:
            accel_cmd: np.array([ax, ay]) in m/s^2
            diag: dict
        """
        if self.mode == "spiral_in":
            acc = self._spiral_in(state, task)
        elif self.mode == "bangband":
            acc = self._bangband(state, task)
        elif self.mode == "transfer":
            acc = self._transfer(state, task)
        elif self.mode == "elliptic_circ":
            acc = self._elliptic_circ(state, task)
        else:
            # fallback
            acc = self._spiral_in(state, task)

        r, _, _, _, _, r_ratio, v_r, v_t = self._geom(state, task.r_target)
        return acc.astype(np.float64), {
            "mode": self.mode,
            "r_ratio": float(r_ratio),
            "v_r": float(v_r),
            "v_t": float(v_t),
        }
