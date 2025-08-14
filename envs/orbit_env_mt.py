import math
import numpy as np

class OrbitEnvMT:
    """
    Multi-task orbit insertion environment (Day 30 MLV).
    - Randomizes target orbit params per episode (radius, eccentricity, mass, thrust limit).
    - Normalized reward across tasks (comparable metrics).
    - Success criterion unified across tasks.
    - Hooks for noise/robustness (attitude/thrust noise).
    API: reset()->obs, step(action)->(obs, reward, done, info)
    """
    def __init__(
        self,
        G: float = 6.67430e-11,
        M: float = 1.989e30,           # central mass (Sun by default)
        base_mass: float = 720.0,      # ~Voyager-1 mass scale
        dt: float = 1.0,               # [s] integrator step
        max_steps: int = 20000,
        # ---- task distributions ----
        r_log_min: float = 1e11,
        r_log_max: float = 1e13,
        e_min: float = 0.0,
        e_max: float = 0.4,
        mass_choices=(600.0, 720.0, 1000.0),
        thrust_scale_range=(0.3, 1.2),  # relative to 1.0
        # ---- success criteria ----
        rerr_thr: float = 0.01,        # |r - rt|/rt < 1%
        verr_thr: float = 0.02,        # |v - v_circ|/v_circ < 2%
        align_thr: float = 0.98,       # cos(angle(v, tangential)) > 0.98
        stable_steps: int = 200,       # consecutive steps to count success
        # ---- reward weights ----
        w_rerr: float = 1.0,
        w_verr: float = 0.5,
        w_align: float = 0.1,
        w_fuel: float = 1e-3,
        w_violation: float = 0.05,
        # ---- noise toggles (set via set_noise) ----
        attitude_noise_deg: float = 0.0,
        thrust_amp_noise: float = 0.0,
        # ---- safety limits ----
        min_radius_factor: float = 0.2,  # terminate if r < 0.2*rt
        max_radius_factor: float = 5.0,  # terminate if r > 5*rt
        seed: int = 42
    ):
        self.G = G
        self.M = M
        self.base_mass = base_mass
        self.dt = dt
        self.max_steps = max_steps

        self.r_log_min = r_log_min
        self.r_log_max = r_log_max
        self.e_min = e_min
        self.e_max = e_max
        self.mass_choices = tuple(mass_choices)
        self.thrust_scale_range = thrust_scale_range

        self.rerr_thr = rerr_thr
        self.verr_thr = verr_thr
        self.align_thr = align_thr
        self.stable_steps_req = stable_steps

        self.w_rerr = w_rerr
        self.w_verr = w_verr
        self.w_align = w_align
        self.w_fuel = w_fuel
        self.w_violation = w_violation

        self.attitude_noise_deg = attitude_noise_deg
        self.thrust_amp_noise = thrust_amp_noise

        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor

        self.rng = np.random.default_rng(seed)

        # state
        self.state = None
        self.t = 0
        self.steps = 0
        self.stable_counter = 0

        # task
        self.task = None
        self.rt = None            # target radius
        self.e = None
        self.mass = None
        self.thrust_limit = None  # absolute thrust limit [N] relative scaled
        self.v_circ = None

        # stats
        self.fuel_used = 0.0
        self.violations = 0

    # ---------- public control ----------
    def set_noise(self, attitude_deg: float = None, thrust_amp: float = None):
        if attitude_deg is not None:
            self.attitude_noise_deg = attitude_deg
        if thrust_amp is not None:
            self.thrust_amp_noise = thrust_amp

    # ---------- core helpers ----------
    def _sample_task(self):
        # Log-uniform radius
        log_r = self.rng.uniform(np.log(self.r_log_min), np.log(self.r_log_max))
        rt = float(np.exp(log_r))
        e = float(self.rng.uniform(self.e_min, self.e_max))
        mass = float(self.rng.choice(self.mass_choices))
        thrust_scale = float(self.rng.uniform(*self.thrust_scale_range))
        thrust_limit = thrust_scale * 1.0  # 1.0 is a project-specific base; adapt if needed

        return dict(target_radius=rt, e=e, mass=mass, thrust_limit=thrust_limit)

    def _task_to_refs(self):
        self.rt = self.task["target_radius"]
        self.e = self.task["e"]
        self.mass = self.task["mass"]
        self.thrust_limit = self.task["thrust_limit"]
        self.v_circ = math.sqrt(self.G * self.M / self.rt)

    def _random_initial_state(self):
        """
        Start near the target radius with a mild angular offset and non-perfect tangential velocity.
        """
        theta = self.rng.uniform(0, 2*np.pi)
        r0 = self.rt * self.rng.uniform(0.8, 1.2)
        x = r0 * np.cos(theta)
        y = r0 * np.sin(theta)

        # start with approx circular speed, with perturbation influenced by e
        v_mag = self.v_circ * self.rng.uniform(1.0 - 0.4*self.e - 0.05, 1.0 + 0.4*self.e + 0.05)
        # direction ~ tangential
        tangential_dir = np.array([-np.sin(theta), np.cos(theta)])
        # add small radial leakage
        mixture = self.rng.uniform(0.9, 1.0)
        v = mixture * v_mag * tangential_dir + (1-mixture) * 0.1*v_mag * np.array([np.cos(theta), np.sin(theta)])

        return np.array([x, y, v[0], v[1]], dtype=np.float64)

    def _grav_acc(self, x, y):
        r2 = x*x + y*y + 1e-12
        r = math.sqrt(r2)
        a_mag = - self.G * self.M / r2
        ax = a_mag * (x / r)
        ay = a_mag * (y / r)
        return ax, ay, r

    def _norm_obs(self, s):
        # normalize by task refs
        x, y, vx, vy = s
        obs = np.array([
            x / self.rt,
            y / self.rt,
            vx / self.v_circ,
            vy / self.v_circ,
            self.e,
            self.thrust_limit
        ], dtype=np.float64)
        return obs

    def _task_embed(self):
        # Minimal task embedding (can be replaced by MLP later)
        return np.array([
            self.rt / 1e12,     # scale hint
            self.e,
            self.mass / self.base_mass,
            self.thrust_limit
        ], dtype=np.float64)

    # ---------- gym-like API ----------
    def reset(self, task: dict | None = None, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = task if task is not None else self._sample_task()
        self._task_to_refs()

        self.state = self._random_initial_state()
        self.t = 0.0
        self.steps = 0
        self.stable_counter = 0
        self.fuel_used = 0.0
        self.violations = 0

        obs = self._norm_obs(self.state)
        return np.concatenate([obs, self._task_embed()], axis=0)

    def step(self, action: np.ndarray):
        """
        action: thrust vector in normalized coordinates relative to thrust_limit
                expected shape (2,), each in [-1, 1]
        """
        # clip and denormalize thrust
        a = np.clip(action, -1.0, 1.0)
        thrust = a * self.thrust_limit  # [N], project uses dimensionless scaling; adapt if needed

        # apply noise
        if self.attitude_noise_deg > 0.0:
            ang = np.deg2rad(self.attitude_noise_deg) * self.rng.normal()
            c, s = math.cos(ang), math.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            thrust = R @ thrust

        if self.thrust_amp_noise > 0.0:
            thrust = thrust * (1.0 + self.thrust_amp_noise * self.rng.normal())

        # unpack state
        x, y, vx, vy = self.state

        # physics
        ax_g, ay_g, r = self._grav_acc(x, y)
        ax_t = thrust[0] / max(self.mass, 1e-9)
        ay_t = thrust[1] / max(self.mass, 1e-9)
        ax = ax_g + ax_t
        ay = ay_g + ay_t

        # integrate (semi-implicit Euler)
        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        x = x + vx * self.dt
        y = y + vy * self.dt

        self.state = np.array([x, y, vx, vy], dtype=np.float64)
        self.t += self.dt
        self.steps += 1

        # ----- metrics -----
        v_mag = math.sqrt(vx * vx + vy * vy)
        r_err = abs(math.sqrt(x * x + y * y) - self.rt) / self.rt
        v_err = abs(v_mag - self.v_circ) / self.v_circ

        # tangential unit
        theta = math.atan2(y, x)
        tdir = np.array([-math.sin(theta), math.cos(theta)])
        if v_mag < 1e-9:
            align = 0.0
        else:
            align = float(np.dot(np.array([vx, vy]) / v_mag, tdir))

        fuel_step = float(np.linalg.norm(thrust)) * self.dt

        # violations
        violation = 0
        if r < self.min_radius_factor * self.rt or r > self.max_radius_factor * self.rt:
            violation += 1

        # reward (normalized, task-invariant scale)
        fuel_pen = self.w_fuel * fuel_step  # scale fuel penalty with dt
        rew = - (self.w_rerr * abs(r_err) + self.w_verr * abs(v_err)) \
              + self.w_align * align - fuel_pen - self.w_violation * violation
        # success tracking
        if (r_err < self.rerr_thr) and (v_err < self.verr_thr) and (align > self.align_thr):
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        done = False
        success = False
        if violation > 0:
            done = True
        elif self.stable_counter >= self.stable_steps_req:
            done = True
            success = True
        elif self.steps >= self.max_steps:
            done = True

        # book-keeping
        self.fuel_used += fuel_step  # accumulate time-integrated fuel
        self.violations += violation

        obs = self._norm_obs(self.state)
        obs = np.concatenate([obs, self._task_embed()], axis=0)

        info = {
            "r_err": r_err,
            "v_err": v_err,
            "align": align,
            "fuel_used": self.fuel_used,
            "violations": self.violations,
            "success": success
        }
        return obs, float(rew), bool(done), info
