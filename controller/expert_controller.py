import numpy as np


class ExpertController:
    """
    Expert Controller  – Physically realistic orbit insertion controller.
    Features:
    - Radial + tangential control
    - Optional damping
    - Capture detection to stop thrust when orbit is reached
    - Clean single-pass entry without spiral loops
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
                 enable_damping=True):
        """
        Initialize the expert controller.

        Args:
            target_radius (float): Desired orbit radius in meters.
            G (float): Gravitational constant.
            M (float): Central mass (e.g. Sun).
            mass (float): Spacecraft mass.
            radial_gain (float): Radial correction gain.
            tangential_gain (float): Tangential speed correction gain.
            damping_gain (float): Damping gain to suppress radial velocity oscillation.
            thrust_limit (float): Max magnitude of thrust vector.
            enable_damping (bool): Toggle damping force.
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

    def act(self, obs, info=None):
        x = np.asarray(obs)
        n = x.size // 2
        pos, vel = x[:n], x[n:]
        return self.__call__(0.0, pos, vel)

    def __call__(self, t, pos, vel):
        """
        Compute thrust vector based on current position and velocity.

        Args:
            t (float): Time (unused here).
            pos (np.ndarray): Position vector [x, y].
            vel (np.ndarray): Velocity vector [vx, vy].

        Returns:
            np.ndarray: Thrust vector [tx, ty].
        """
        r_vec = np.array(pos)
        v_vec = np.array(vel)

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # Compute unit vectors
        radial_dir = r_vec / (r + 1e-12)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

        # Desired circular orbit speed
        v_circular = np.sqrt(self.G * self.M / self.target_radius)

        # Component along tangential only (for more realistic speed control)
        v_tangential = np.dot(v_vec, tangential_dir)
        delta_v = v_circular - v_tangential

        # Radial error
        radial_error = r - self.target_radius

        # Tangential control: accelerate/decelerate into circular speed
        thrust_t = self.tangential_gain * np.tanh(delta_v / v_circular)

        # Radial control: bring r to target radius
        thrust_r = -self.radial_gain * np.tanh(radial_error / (0.05 * self.target_radius))

        # Damping: suppress radial oscillation
        if self.enable_damping:
            radial_velocity = np.dot(v_vec, radial_dir)
            radial_error = r - self.target_radius
            proximity = 1.0 - np.clip(abs(radial_error) / self.target_radius, 0.0, 1.0)  # 越接近，越强
            thrust_r += -self.damping_gain * proximity * np.tanh(radial_velocity / 1e4)

        # Stop thrust when orbit is stable
        if abs(radial_error) < 0.001 * self.target_radius and abs(delta_v) < 0.005 * v_circular:
            return np.zeros(2)

        # Final thrust vector
        thrust_vec = thrust_r * radial_dir + thrust_t * tangential_dir

        # Clip to max thrust limit
        norm = np.linalg.norm(thrust_vec)
        if norm > self.thrust_limit:
            thrust_vec = thrust_vec / norm * self.thrust_limit

        return thrust_vec
# --- adapter for replay_recorder: expects a callable policy(obs, info=None) -> action(np.ndarray-like)
_expert_singleton = None

def policy(obs, info=None):
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = ExpertController()
    # 若类是 __call__ 可调用，就用它；否则优先用 act(...)
    if hasattr(_expert_singleton, "__call__"):
        return _expert_singleton(obs, info=info) if "info" in _expert_singleton.__call__.__code__.co_varnames else _expert_singleton(obs)
    if hasattr(_expert_singleton, "act"):
        # 兼容 act(obs) / act(obs, info=...)
        return _expert_singleton.act(obs, info=info) if "info" in _expert_singleton.act.__code__.co_varnames else _expert_singleton.act(obs)
    raise TypeError("ExpertController has neither __call__ nor act method")
# ==== adapter v2: build controller from info ====
_expert_singleton = None

def _build_controller_from_info(info):
    # Collect kwargs for ExpertController
    kw = {}
    if isinstance(info, dict):
        # common keys the env may expose
        for k in ("target_radius", "mu", "dt", "mass", "sc_mass"):
            if k in info and info[k] is not None:
                # sc_mass -> mass
                if k == "sc_mass":
                    kw["mass"] = info[k]
                else:
                    kw[k] = info[k]
        # map r0 -> target_radius if needed
        if "target_radius" not in kw and "r0" in info and info["r0"] is not None:
            kw["target_radius"] = info["r0"]
        # sometimes env may put params under "params"
        if "target_radius" not in kw and "params" in info and isinstance(info["params"], dict):
            p = info["params"]
            if "target_radius" in p and p["target_radius"] is not None:
                kw["target_radius"] = p["target_radius"]
            elif "r0" in p and p["r0"] is not None:
                kw["target_radius"] = p["r0"]

    if "target_radius" not in kw:
        raise ValueError(f"ExpertController needs target_radius; not found in info. got keys={sorted(list(info.keys())) if isinstance(info, dict) else type(info)}")
    return ExpertController(**kw)

def policy(obs, info=None):
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = _build_controller_from_info(info or {})
    # prefer __call__(obs, info?) then act(...)
    if hasattr(_expert_singleton, "__call__"):
        try:
            return _expert_singleton(obs, info=info)
        except TypeError:
            return _expert_singleton(obs)
    if hasattr(_expert_singleton, "act"):
        try:
            return _expert_singleton.act(obs, info=info)
        except TypeError:
            return _expert_singleton.act(obs)
    raise TypeError("ExpertController has neither __call__ nor act method")
# ==== adapter v3: fallback to PRESET_MAP when info is empty ====
def _build_controller_from_info(info):
    # try from info first
    kw = {}
    if isinstance(info, dict):
        for k in ("target_radius", "mu", "dt", "mass", "sc_mass"):
            if k in info and info[k] is not None:
                kw["mass" if k=="sc_mass" else k] = info[k]
        if "target_radius" not in kw and "r0" in info and info["r0"] is not None:
            kw["target_radius"] = info["r0"]
        if "target_radius" not in kw and "params" in info and isinstance(info["params"], dict):
            p = info["params"]
            if "target_radius" in p and p["target_radius"] is not None:
                kw["target_radius"] = p["target_radius"]
            elif "r0" in p and p["r0"] is not None:
                kw["target_radius"] = p["r0"]

    # fallback to preset if still missing
    if "target_radius" not in kw:
        try:
            from envs.orbit_presets import PRESET_MAP
            base = PRESET_MAP.get("transfer") or PRESET_MAP.get("circular")
            # tolerate both dataclass or simple object
            try:
                from dataclasses import asdict as _asdict
                bd = _asdict(base)
            except Exception:
                bd = {k: getattr(base, k) for k in dir(base) if not k.startswith("_")}
            # map r0 -> target_radius if present
            if "target_radius" in bd and bd["target_radius"] is not None:
                kw["target_radius"] = bd["target_radius"]
            elif "r0" in bd and bd["r0"] is not None:
                kw["target_radius"] = bd["r0"]
            # optional extras
            if "mu" in bd and bd["mu"] is not None: kw.setdefault("mu", bd["mu"])
            if "dt" in bd and bd["dt"] is not None: kw.setdefault("dt", bd["dt"])
            if "mass" in bd and bd["mass"] is not None: kw.setdefault("mass", bd["mass"])
            if "sc_mass" in bd and bd["sc_mass"] is not None: kw.setdefault("mass", bd["sc_mass"])
        except Exception as e:
            raise ValueError(f"Cannot build ExpertController: no target_radius in info and preset fallback failed: {e!r}")

    return ExpertController(**kw)

# redefine policy to use the updated builder
_expert_singleton = None
def policy(obs, info=None):
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = _build_controller_from_info(info or {})
    if hasattr(_expert_singleton, "__call__"):
        try:
            return _expert_singleton(obs, info=info)
        except TypeError:
            return _expert_singleton(obs)
    if hasattr(_expert_singleton, "act"):
        try:
            return _expert_singleton.act(obs, info=info)
        except TypeError:
            return _expert_singleton.act(obs)
    raise TypeError("ExpertController has neither __call__ nor act method")
# ==== adapter v4: filter kwargs by ExpertController.__init__ signature ====
def _build_controller_from_info(info):
    # 1) 先从 info 抽取候选键
    kw = {}
    if isinstance(info, dict):
        for k in ("target_radius", "r0", "dt", "mass", "sc_mass"):
            if k in info and info[k] is not None:
                if k == "r0":
                    kw["target_radius"] = info[k]
                elif k == "sc_mass":
                    kw["mass"] = info[k]
                else:
                    kw[k] = info[k]
        # 支持 info["params"] 的兜底
        p = info.get("params")
        if isinstance(p, dict):
            if "target_radius" in p and p["target_radius"] is not None:
                kw.setdefault("target_radius", p["target_radius"])
            if "r0" in p and p["r0"] is not None:
                kw.setdefault("target_radius", p["r0"])
            if "dt" in p and p["dt"] is not None:
                kw.setdefault("dt", p["dt"])
            if "mass" in p and p["mass"] is not None:
                kw.setdefault("mass", p["mass"])
            if "sc_mass" in p and p["sc_mass"] is not None:
                kw.setdefault("mass", p["sc_mass"])

    # 2) 若仍缺 target_radius，从 PRESET_MAP('transfer' -> 'circular') 回退
    if "target_radius" not in kw:
        try:
            from envs.orbit_presets import PRESET_MAP
            base = PRESET_MAP.get("transfer") or PRESET_MAP.get("circular")
            try:
                from dataclasses import asdict as _asdict
                bd = _asdict(base)
            except Exception:
                bd = {k: getattr(base, k) for k in dir(base) if not k.startswith("_")}
            if bd.get("target_radius") is not None:
                kw["target_radius"] = bd["target_radius"]
            elif bd.get("r0") is not None:
                kw["target_radius"] = bd["r0"]
            # 其它字段只作为候选，后面会过滤
            if bd.get("dt") is not None:
                kw.setdefault("dt", bd["dt"])
            if bd.get("mass") is not None:
                kw.setdefault("mass", bd["mass"])
            if bd.get("sc_mass") is not None:
                kw.setdefault("mass", bd["sc_mass"])
        except Exception as e:
            raise ValueError(f"Cannot build ExpertController: no target_radius in info and preset fallback failed: {e!r}")

    # 3) 严格按 ExpertController.__init__ 的形参过滤
    import inspect as _inspect
    sig = _inspect.signature(ExpertController.__init__)
    allowed = {name for name, p in sig.parameters.items() if name != "self"}
    filtered = {k: v for k, v in kw.items() if k in allowed}

    # 4) 构造实例
    return ExpertController(**filtered)

# 统一的策略函数，保持容错：__call__ / act
_expert_singleton = None
def policy(obs, info=None):
    global _expert_singleton
    if _expert_singleton is None:
        _expert_singleton = _build_controller_from_info(info or {})
    # 优先可调用实例
    if hasattr(_expert_singleton, "__call__"):
        try:
            return _expert_singleton(obs, info=info)
        except TypeError:
            return _expert_singleton(obs)
    # 其次 act 方法
    if hasattr(_expert_singleton, "act"):
        try:
            return _expert_singleton.act(obs, info=info)
        except TypeError:
            return _expert_singleton.act(obs)
    raise TypeError("ExpertController has neither __call__ nor act method")
# ==== adapter v5: auto-split obs into (pos, vel) if controller needs them ====
import numpy as _np
import inspect as _inspect

def _split_obs_pos_vel(obs):
    x = _np.asarray(obs)
    # 1D: [pos..., vel...]
    if x.ndim == 1 and x.size % 2 == 0:
        n = x.size // 2
        return x[:n], x[n:]
    # 2D batch: [:, pos..., vel...]
    if x.ndim == 2 and x.shape[1] % 2 == 0:
        n = x.shape[1] // 2
        return x[:, :n], x[:, n:]
    # unknown layout
    return None, None

def _call_like(sig_func, func, obs, info):
    """Call func according to signature of sig_func (without 'self')."""
    sig = _inspect.signature(sig_func)
    params = [name for name in sig.parameters if name != "self"]

    if len(params) >= 3 and params[0] in ("t", "time") and params[1] == "pos" and params[2] == "vel":
        pos, vel = _split_obs_pos_vel(obs)
        if pos is None:
            raise TypeError("Controller expects (t, pos, vel) but obs shape is incompatible.")
        t0 = 0.0
        if "info" in params:
            return func(t0, pos, vel, info=info)
        else:
            return func(t0, pos, vel)

    if len(params) >= 2 and params[0] == "pos" and params[1] == "vel":
        pos, vel = _split_obs_pos_vel(obs)
        if pos is None:
            raise TypeError("Controller expects (pos, vel) but obs shape is incompatible.")
        if "info" in params:
            return func(pos, vel, info=info)
        else:
            return func(pos, vel)

    if len(params) >= 1:
        if "info" in params:
            try:
                return func(obs, info=info)
            except TypeError:
                pass
        return func(obs)

    return func()


def policy(obs, info=None):
    """Universal policy wrapper that adapts to ExpertController API variants."""
    global _expert_singleton
    if _expert_singleton is None:
        # reuse previous builder (already appended earlier)
        try:
            builder = globals().get("_build_controller_from_info")
            _expert_singleton = builder(info or {}) if builder else ExpertController()
        except TypeError:
            _expert_singleton = ExpertController()

    # Prefer __call__
    if hasattr(_expert_singleton, "__call__"):
        try:
            return _call_like(_expert_singleton.__call__, _expert_singleton.__call__, obs, info)
        except TypeError:
            pass

    # Then act(...)
    if hasattr(_expert_singleton, "act"):
        try:
            return _call_like(_expert_singleton.act, _expert_singleton.act, obs, info)
        except TypeError:
            pass

    raise TypeError("ExpertController has neither a compatible __call__ nor act")
