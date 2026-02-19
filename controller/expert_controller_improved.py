# Expert Controller v4 -- Week4 Robust Version
# Features:
# 1) Distance-based thrust scaling (fixes weak_thrust_far)
# 2) Low-pass filtering on thrust direction (fixes oscillation_noise)
# 3) Near-target angular momentum alignment (fixes misaligned_entry)

import numpy as np


class ExpertController:
    """
    Expert Controller – physically consistent orbit insertion controller.
    v4 adds robustness improvements based on Week3 failure catalog.
    """

    def __init__(self,
                 target_radius,
                 G=6.67430e-11,
                 M=1.989e30,
                 mass=721.9,
                 radial_gain=4.0,
                 tangential_gain=5.0,
                 damping_gain=6.0,
                 thrust_limit=20.0,
                 enable_damping=True,
                 enable_scheduler=False,
                 enable_alignment=False):
        """
        Initialize the controller.

        target_radius: desired orbit radius
        G, M: gravitational parameters
        mass: spacecraft mass
        radial_gain, tangential_gain: control gainsz
        damping_gain: radial oscillation damping
        thrust_limit: max thrust magnitude
        enable_damping: toggle for radial damping term
        enable_scheduler: toggle for distance-based thrust scaling
        enable_alignment: toggle for angular-momentum alignment term
        """
        self.target_radius = target_radius
        self.G = G
        self.M = M
        self.mass = mass
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.damping_gain = damping_gain
        self.thrust_limit = thrust_limit
        self.enable_damping = enable_damping

        # Week4 robustness features (v4.2 baseline: only smoothing is active)
        self.enable_scheduler = enable_scheduler
        self.enable_alignment = enable_alignment

        self.smoothed_dir = None
        self.prev_thrust = None

        # WHPL_09: explicit radial PD (always-on, bounded)
        self.enable_radial_pd = True
        self.radial_pd_gain_p = 0.60    # dimensionless, used inside tanh
        self.radial_pd_gain_d = 0.60    # dimensionless, used inside tanh
        self.radial_pd_cap_frac = 0.60  # IMPORTANT: was 0.20; make it visible

        self.whpl09_debug = True
        self.whpl10_disable_smoothing = False  # WHPL_10: 1-shot smoothing control
        self.whpl09_every = 200
        self._whpl09_ctr = 0
        # WHPL_13: toggle error-band gating for radial PD injection
        self.enable_pd_gating = True

    # v4 Part 1: distance-based scaling

    def _thrust_scheduler(self, r, r_target):
        """Scale thrust when the spacecraft is far from the target orbit.

        v4.1: use a milder schedule to avoid overshoot.
        """
        ratio = r / (r_target + 1e-12)
        if ratio > 1.4:
            return 1.25
        elif ratio > 1.1:
            return 1.10
        return 1.0

    # API wrapper used by env

    def act(self, obs, info=None):
        x = np.asarray(obs)
        n = x.size // 2
        pos, vel = x[:n], x[n:]
        return self.__call__(0.0, pos, vel)

    # Main controller logic

    def __call__(self, t, pos, vel):
        """
        Compute thrust vector based on position and velocity.
        """

        # convert to vectors
        r_vec = np.array(pos)
        v_vec = np.array(vel)
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        radial_dir = r_vec / (r + 1e-12)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

        # target tangential velocity
        v_circular = np.sqrt(self.G * self.M / self.target_radius)
        v_tangential = np.dot(v_vec, tangential_dir)
        delta_v = v_circular - v_tangential

        # radial error
        radial_error = r - self.target_radius

        # tangential correction
        thrust_t = self.tangential_gain * np.tanh(delta_v / (v_circular + 1e-12))

        # radial correction
        thrust_r = -self.radial_gain * np.tanh(
            radial_error / (0.05 * self.target_radius)
        )

        # =========================
        # WHPL_09: always-on radial PD injection (bounded)
        thrust_r_pd = 0.0  # default evidence value
        g_r = 0.0
        d_scale = 1.0

        if self.enable_radial_pd:
            radial_velocity = float(np.dot(v_vec, radial_dir))  # v_r


            # WHPL_10 Knife 1: Error-band gating (continuous)

            r_on = 0.12
            r_full = 0.30
            rel = abs(radial_error) / (self.target_radius + 1e-12)  # dimensionless
            g_r = (rel - r_on) / (r_full - r_on + 1e-12)
            g_r = float(np.clip(g_r, 0.0, 1.0))

            # PD in the SAME normalized coordinates as your existing design
            p_term = -self.radial_pd_gain_p * np.tanh(
                radial_error / (0.05 * self.target_radius + 1e-12)
            )
            d_term = -self.radial_pd_gain_d * np.tanh(radial_velocity / 1e4)

            # WHPL_10 Knife 2: D-term sign gating (reduce D when already moving inward)
            # Condition uses YOUR conventions:
            # radial_error > 0  => r above target
            # radial_velocity < 0 => moving inward

            d_inward_scale = 0.3
            d_scale = 1.0
            if (radial_error > 0.0) and (radial_velocity < 0.0):
                d_scale = d_inward_scale
            d_term = d_scale * d_term

            thrust_r_pd = p_term + d_term

            # Bound PD injection by a fraction of thrust_limit (same units as thrust_r)
            cap = float(self.radial_pd_cap_frac) * float(self.thrust_limit)
            thrust_r_pd = float(np.clip(thrust_r_pd, -1.0, 1.0)) * cap

            # WHPL_13: optional error-band gating (structure toggle)
            if self.enable_pd_gating:
                thrust_r_pd = g_r * thrust_r_pd

            thrust_r += thrust_r_pd

        # =========================

        # damping term (near target)
        if self.enable_damping:
            radial_velocity = np.dot(v_vec, radial_dir)
            proximity = 1.0 - np.clip(
                abs(radial_error) / self.target_radius, 0.0, 1.0
            )
            thrust_r += -self.damping_gain * proximity * np.tanh(radial_velocity / 1e4)

        # stop thrust when nearly stable
        if (
            abs(radial_error) < 0.001 * self.target_radius
            and abs(delta_v) < 0.005 * v_circular
        ):
            return np.zeros(2)

        # v4 Part 1: thrust scaling

        if self.enable_scheduler:
            scale = self._thrust_scheduler(r, self.target_radius)
            thrust_r *= scale
            thrust_t *= scale

        # raw thrust vector
        thrust_vec = thrust_r * radial_dir + thrust_t * tangential_dir

        base_norm = np.linalg.norm(thrust_vec)
        if base_norm < 1e-12:
            return np.zeros(2)

        # v4 Part 2: low-pass filtering
        raw_dir = thrust_vec / base_norm

        if self.whpl10_disable_smoothing:
            thrust_dir = raw_dir
        else:
            if self.smoothed_dir is None:
                self.smoothed_dir = raw_dir
            else:
                alpha = 0.05
                self.smoothed_dir = alpha * raw_dir + (1.0 - alpha) * self.smoothed_dir
                self.smoothed_dir /= np.linalg.norm(self.smoothed_dir) + 1e-12
            thrust_dir = self.smoothed_dir

        thrust_mag = base_norm

        # v4 Part 3: angular momentum alignment near target
        # v4.1: only apply angular-momentum alignment when both
        # radius and h_err are in a reasonable band; use a mild gain.
        if self.enable_alignment:
            r_err_rel = abs(r - self.target_radius) / self.target_radius
            if r_err_rel < 0.12:
                h_mag = r * v_tangential
                h_target = self.target_radius * v_circular
                h_err = (h_mag - h_target) / (h_target + 1e-12)

                if abs(h_err) > 0.02:  # ignore tiny noise
                    k_align = 0.20
                    align_dir = -k_align * np.sign(h_err) * tangential_dir

                    combined = thrust_dir + align_dir
                    norm_c = np.linalg.norm(combined)
                    if norm_c > 1e-8:
                        thrust_dir = combined / norm_c

        thrust_vec = thrust_dir * thrust_mag

        # safety: thrust limit
        nrm = np.linalg.norm(thrust_vec)
        if nrm > self.thrust_limit:
            thrust_vec = thrust_vec / nrm * self.thrust_limit

        self.prev_thrust = thrust_vec

        if self.whpl09_debug:
            ctr = self._whpl09_ctr  # snapshot current counter

            if ctr in (0, 200, 400, 600, 800):
                print(
                    f"[WHPL_10] step={ctr} "
                    f"r={r:.12e} "
                    f"r_err={radial_error:.12e} "
                    f"v_r={np.dot(v_vec, radial_dir):+.6e} "
                    f"rel={abs(radial_error) / (self.target_radius + 1e-12):.4f} "
                    f"g_r={g_r:.3f} "
                    f"d_scale={d_scale:.2f} "
                    f"thrust_r={thrust_r:+.6e} "
                    f"thrust_r_pd={thrust_r_pd:+.6e} "
                    f"thrust_t={thrust_t:+.6e} "
                    f"thrust_norm={np.linalg.norm(thrust_vec):.6e}"
                )

            self._whpl09_ctr = ctr + 1

        return thrust_vec


# Unified policy adapter (simple & clean version)

_expert_singleton = None

def _build_controller_from_info(info):
    """Extract target_radius from env info dict."""
    if isinstance(info, dict):
        if "target_radius" in info and info["target_radius"] is not None:
            return ExpertController(target_radius=info["target_radius"])
        if "r0" in info and info["r0"] is not None:
            return ExpertController(target_radius=info["r0"])
        if "params" in info and isinstance(info["params"], dict):
            p = info["params"]
            if "target_radius" in p:
                return ExpertController(target_radius=p["target_radius"])
            if "r0" in p:
                return ExpertController(target_radius=p["r0"])

    # final fallback (safe)
    raise ValueError("Cannot initialize ExpertController: target_radius not found.")


def policy(obs, info=None):
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = _build_controller_from_info(info or {})

    # try __call__
    if hasattr(_expert_singleton, "__call__"):
        try:
            return _expert_singleton(obs, info=info)
        except TypeError:
            return _expert_singleton(obs)

    # fallback to act()
    if hasattr(_expert_singleton, "act"):
        try:
            return _expert_singleton.act(obs, info=info)
        except TypeError:
            return _expert_singleton.act(obs)

    raise TypeError("ExpertController does not implement __call__ or act().")
