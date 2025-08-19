import os
import csv
import numpy as np

from envs.orbit_env_mt import OrbitEnvMT
from baselines.zero_thrust import ZeroThrustController
from baselines.greedy_energy_rt import GreedyEnergyRTController  # Energy-shaping baseline
from tools.ab_tools import BatteryRecord, run_ab_compare
from controller.expert_controller import ExpertController  # Expert core


# paths
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "battery_day31.csv")


# fixed benchmark
def fixed_tasks():
    """5 deterministic tasks for reproducible comparisons."""
    return [
        dict(name="fixed_1", target_radius=5e11, e=0.0,  mass=720.0,  thrust_limit=1.0),
        dict(name="fixed_2", target_radius=7e11, e=0.2,  mass=720.0,  thrust_limit=1.0),
        dict(name="fixed_3", target_radius=5e11, e=0.05, mass=720.0,  thrust_limit=0.4),
        dict(name="fixed_4", target_radius=5e11, e=0.1,  mass=1000.0, thrust_limit=1.0),
        dict(name="fixed_5", target_radius=2e12, e=0.0,  mass=720.0,  thrust_limit=1.0),
    ]


# rollout
def rollout(env: OrbitEnvMT, controller, max_episodes=1, seed=123, pre_reset=False, initial_obs=None):
    """
    Run episodes and return success rate + per-episode totals.

    NOTE: We also call controller.bind_env(env) if available, so adapters that
    need env metadata (e.g., mu, target_radius, thresholds) can access it.
    """
    sr = 0
    totals = []
    for ep in range(max_episodes):
        if pre_reset:
            assert initial_obs is not None, "initial_obs must be provided when pre_reset=True"
            obs = initial_obs
        else:
            obs = env.reset(seed=seed + ep)
            if hasattr(controller, "set_task"):
                controller.set_task(env.task)

        if hasattr(controller, "bind_env"):
            try:
                controller.bind_env(env)
            except Exception:
                pass

        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            a = controller.act(obs)
            obs, r, done, info = env.step(a)
            ep_reward += r
            steps += 1
        totals.append((ep_reward, steps, info))
        if info.get("success", False):
            sr += 1
    return sr / max_episodes, totals


# evaluate
def eval_controller(name, controller, env: OrbitEnvMT, tasks, randN=20, seed=999):
    """
    Evaluate a controller on both the fixed task battery and N random tasks.
    Returns a list of rows to be written to CSV.
    """
    rows = []

    # Fixed tasks
    for i, t in enumerate(tasks):
        obs0 = env.reset(task=t, seed=seed + i)
        if hasattr(controller, "set_task"):
            controller.set_task(env.task)
        sr, totals = rollout(env, controller, max_episodes=1, seed=seed + i, pre_reset=True, initial_obs=obs0)
        rew, steps, info = totals[0]
        ended_by_max = int((steps >= env.max_steps) and (not info["success"]) and (info["violations"] == 0))
        ended_by_violation = int(info["violations"] > 0)
        rows.append([
            name, f"fixed_{i+1}", sr, rew, steps,
            info["r_err"], info["v_err"], info["align"],
            info["fuel_used"], info["violations"], info["success"],
            ended_by_max, ended_by_violation,
            env.task["target_radius"], env.task["e"], env.task["mass"], env.task["thrust_limit"]
        ])

    # Random tasks
    rng = np.random.default_rng(seed + 777)
    for j in range(randN):
        obs0 = env.reset(task=None, seed=int(rng.integers(0, 10_000_000)))
        if hasattr(controller, "set_task"):
            controller.set_task(env.task)
        sr, totals = rollout(env, controller, max_episodes=1, seed=seed + 1000 + j, pre_reset=True, initial_obs=obs0)
        rew, steps, info = totals[0]
        ended_by_max = int((steps >= env.max_steps) and (not info["success"]) and (info["violations"] == 0))
        ended_by_violation = int(info["violations"] > 0)
        rows.append([
            name, f"random_{j+1}", sr, rew, steps,
            info["r_err"], info["v_err"], info["align"],
            info["fuel_used"], info["violations"], info["success"],
            ended_by_max, ended_by_violation,
            env.task["target_radius"], env.task["e"], env.task["mass"], env.task["thrust_limit"]
        ])
    return rows


# ExpertAdapter
class ExpertAdapter:
    """
    Glue layer that feeds *physical* (pos, vel) to ExpertController.
    Adds:
      - stop-in-band with hysteresis for fuel saving
      - hard clamp on thrust magnitude
      - firing gate to zero-out tiny thrust
      - optional bang-bang (snap to cap when firing)
      - pacing: minimum coasting/burst steps to avoid chatter

    Priority to obtain (pos, vel):
      1) env.get_raw_rv() / get_state_vectors() / raw_rv()
      2) env.pos/env.vel (or r/v, position/velocity, etc.)
      3) Denormalize obs by rt and sqrt(mu/rt)
      4) Fallback: raw obs indices (last resort)
    """
    def __init__(self, expert_ctrl, pos_idx=(0, 1), vel_idx=(2, 3), quiet=True,
                 stop_in_band=True, band_in_scale=1.35, band_out_scale=2.00,
                 a_cap=None, fire_frac=0.65, bang_bang=False,
                 min_coast_steps=2, min_burst_steps=2):
        self.ctrl = expert_ctrl
        self.env = None
        self.pos_idx = tuple(pos_idx)
        self.vel_idx = tuple(vel_idx)
        self.quiet = quiet
        self._warned = False
        # Hysteresis band
        self.stop_in_band = bool(stop_in_band)
        self.band_in_scale = float(band_in_scale)
        self.band_out_scale = float(band_out_scale)
        self._captured = False
        # Thrust moderation
        self.a_cap = None if a_cap is None else float(a_cap)
        self.fire_frac = float(fire_frac)
        self.bang_bang = bool(bang_bang)
        # Pacing (anti-chatter)
        self.min_coast_steps = int(min_coast_steps)
        self.min_burst_steps = int(min_burst_steps)
        self._coast_ctr = 0
        self._burst_ctr = 0

    def bind_env(self, env):
        self.env = env

    def get_action(self, state):
        """Return action from expert; optionally zero when inside the success band."""
        r, v = self._extract_rv(state)

        # Hysteresis-based coasting (fuel saving)
        if self.stop_in_band and (self.env is not None):
            mu = self._get_mu()
            rt = self._get_target_radius()
            if (mu is not None) and (rt is not None):
                v_circ = np.sqrt(mu / (rt + 1e-12))
                # normalized errors wrt env thresholds
                rerr = abs(np.linalg.norm(r) - rt) / (rt + 1e-12)
                verr = abs(v_circ - self._vt(v, r)) / (v_circ + 1e-12)

                rin = getattr(self.env, "rerr_thr", 0.015) * self.band_in_scale
                vin = getattr(self.env, "verr_thr", 0.030) * self.band_in_scale
                rout = getattr(self.env, "rerr_thr", 0.015) * self.band_out_scale
                vout = getattr(self.env, "verr_thr", 0.030) * self.band_out_scale

                if self._captured:
                    # leave coasting only if we drift out of the outer band
                    if (rerr >= rout) or (verr >= vout):
                        self._captured = False
                else:
                    # enter coasting once inside the inner band
                    if (rerr <= rin) and (verr <= vin):
                        self._captured = True

                if self._captured:
                    # pacing: once coasting, enforce minimum coasting duration
                    self._coast_ctr = max(self._coast_ctr, self.min_coast_steps)
                    self._burst_ctr = 0
                    self._coast_ctr -= 1
                    return np.zeros(2, dtype=float)

        # If not force-returning zeros, still honor pacing if in coasting window
        if self._coast_ctr > 0:
            self._coast_ctr -= 1
            return np.zeros(2, dtype=float)

        # Raw expert output (in accel space)
        a = self.ctrl(0.0, r, v)
        a = np.asarray(a, dtype=float).reshape(-1)

        # Firing gate + hard clamp + optional bang-bang
        firing = False
        if self.a_cap is not None:
            n = np.linalg.norm(a) + 1e-12
            if n < self.fire_frac * self.a_cap:
                a[:] = 0.0
                firing = False
            else:
                if n > self.a_cap:  # hard clamp
                    a *= (self.a_cap / n)
                if self.bang_bang:  # snap to cap if enabled
                    a *= (self.a_cap / (np.linalg.norm(a) + 1e-12))
                firing = True
        else:
            firing = (np.linalg.norm(a) > 0.0)

        # Pacing: ensure minimum burst/coast durations to avoid chatter
        if firing:
            if self._burst_ctr <= 0:
                self._burst_ctr = self.min_burst_steps
            else:
                self._burst_ctr -= 1
            self._coast_ctr = self.min_coast_steps
        else:
            if self._coast_ctr <= 0:
                self._coast_ctr = self.min_coast_steps
            a[:] = 0.0

        return a[:2]

    # --- helpers ---
    def _vt(self, v_vec, r_vec):
        """Tangential component of velocity wrt the current radius direction."""
        r = np.linalg.norm(r_vec) + 1e-12
        er = r_vec / r
        et = np.array([-er[1], er[0]])
        return float(np.dot(v_vec, et))

    def _extract_rv(self, state):
        import numpy as _np
        if self.env is not None:
            for meth in ("get_raw_rv", "get_state_vectors", "raw_rv"):
                if hasattr(self.env, meth):
                    try:
                        r, v = getattr(self.env, meth)()
                        return _np.asarray(r).reshape(-1)[:2], _np.asarray(v).reshape(-1)[:2]
                    except Exception:
                        pass
            pairs = [("pos", "vel"), ("r", "v"), ("position", "velocity"),
                     ("x", "v"), ("x", "xdot"), ("r_vec", "v_vec")]
            for pa, va in pairs:
                if hasattr(self.env, pa) and hasattr(self.env, va):
                    r = _np.asarray(getattr(self.env, pa)).reshape(-1)[:2]
                    v = _np.asarray(getattr(self.env, va)).reshape(-1)[:2]
                    return r, v
            if hasattr(self.env, "state"):
                st = getattr(self.env, "state")
                if isinstance(st, dict):
                    for pa, va in [("pos","vel"),("r","v"),("position","velocity"),("x","v"),("x","xdot"),("r_vec","v_vec")]:
                        if pa in st and va in st:
                            r = _np.asarray(st[pa]).reshape(-1)[:2]
                            v = _np.asarray(st[va]).reshape(-1)[:2]
                            return r, v
            # Denormalize from obs using (mu, rt)
            mu = self._get_mu()
            rt = self._get_target_radius()
            if (mu is not None) and (rt is not None):
                obs = _np.asarray(state).reshape(-1)
                r = obs[list(self.pos_idx)] * float(rt)
                v_ref = (float(mu) / float(rt)) ** 0.5
                v = obs[list(self.vel_idx)] * v_ref
                if not self.quiet and not self._warned:
                    print(f"[AB] denorm via mu,rt; v_ref={v_ref:.3e}")
                    self._warned = True
                return r.astype(float), v.astype(float)

        # Fallback: slice raw obs
        obs = _np.asarray(state).reshape(-1)
        try:
            r = obs[list(self.pos_idx)]
            v = obs[list(self.vel_idx)]
        except Exception:
            r = obs[:2]
            v = obs[2:4] if obs.shape[0] >= 4 else obs[2:]
        if not self.quiet and not self._warned:
            print("[AB][warn] using obs indices without denorm")
            self._warned = True
        return r, v

    def _get_mu(self):
        if self.env is None:
            return None
        if hasattr(self.env, "mu"):
            try:
                return float(self.env.mu)
            except Exception:
                pass
        if hasattr(self.env, "G") and hasattr(self.env, "M"):
            try:
                return float(self.env.G) * float(self.env.M)
            except Exception:
                pass
        return None

    def _get_target_radius(self):
        if self.env is None:
            return None
        t = getattr(self.env, "task", None)
        if isinstance(t, dict):
            if "target_radius" in t:
                return float(t["target_radius"])
            if "rt" in t:
                return float(t["rt"])
        return None


# thrust mapping (glue only)
# Base accel cap mapping; per-controller scaling can be applied on top.
EXPERT_THRUST_MAP = "linear"   # set to "identity" to pass-through env thrust_limit
_T_LO, _T_HI = 100.0, 150.0
_A_LO, _A_HI = 0.03, 0.09     # eco uses this as-is; fast will scale slightly

def _map_env_thrust_to_expert(thrust):
    """
    Map env thrust_limit (N) to Expert's accel cap.
    Switch EXPERT_THRUST_MAP="identity" to keep your original feel.
    """
    t = float(thrust)
    if EXPERT_THRUST_MAP == "identity":
        return t
    u = (t - _T_LO) / (_T_HI - _T_LO)
    u = max(0.0, min(1.0, u))
    return _A_LO + u * (_A_HI - _A_LO)


# Expert baseline wrapper
class ExpertBaseline:
    """
    Rebuilds ExpertController per task and delegates actions to ExpertAdapter.

    - set_task(task): build ExpertController based on current task config
    - bind_env(env): pass environment to Adapter (needed for denormalization)
    - act(obs): call Adapter.get_action(obs)
    """
    def __init__(self,
                 radial_gain=3.4,
                 tangential_gain=3.6,
                 damping_gain=9.2,
                 band_in=1.35, band_out=2.00,
                 stop_in_band=True,
                 fire_frac=0.65,
                 bang_bang=False,
                 min_coast_steps=2,
                 min_burst_steps=2,
                 a_cap_scale=1.00,
                 label="expert"):
        self.radial_gain = radial_gain
        self.tangential_gain = tangential_gain
        self.damping_gain = damping_gain
        self.band_in = band_in
        self.band_out = band_out
        self.stop_in_band = stop_in_band
        self.fire_frac = fire_frac
        self.bang_bang = bang_bang
        self.min_coast_steps = int(min_coast_steps)
        self.min_burst_steps = int(min_burst_steps)
        self.a_cap_scale = float(a_cap_scale)
        self.label = str(label)
        self._adapter = None
        self._env = None

    def set_task(self, task: dict):
        rt = float(task.get("target_radius", task.get("rt")))
        mass = float(task.get("mass", 721.9))
        thrust = float(task.get("thrust_limit", task.get("thrust", 125.0)))

        a_cap = _map_env_thrust_to_expert(thrust) * self.a_cap_scale

        expert_core = ExpertController(
            target_radius=rt,
            mass=mass,
            thrust_limit=a_cap,
            radial_gain=self.radial_gain,
            tangential_gain=self.tangential_gain,
            damping_gain=self.damping_gain,
            enable_damping=True,
        )
        self._adapter = ExpertAdapter(
            expert_core,
            pos_idx=(0, 1),
            vel_idx=(2, 3),
            quiet=True,
            stop_in_band=self.stop_in_band,
            band_in_scale=self.band_in,
            band_out_scale=self.band_out,
            a_cap=a_cap,
            fire_frac=self.fire_frac,
            bang_bang=self.bang_bang,
            min_coast_steps=self.min_coast_steps,
            min_burst_steps=self.min_burst_steps,
        )
        if self._env is not None and hasattr(self._adapter, "bind_env"):
            self._adapter.bind_env(self._env)

    def bind_env(self, env):
        self._env = env
        if self._adapter is not None and hasattr(self._adapter, "bind_env"):
            self._adapter.bind_env(env)

    def act(self, obs):
        if self._adapter is None:
            import numpy as _np
            return _np.zeros(2, dtype=float)
        return self._adapter.get_action(obs)


# main
def main():
    env = OrbitEnvMT(
        rerr_thr=0.015, verr_thr=0.030, align_thr=0.97, stable_steps=160,
        w_fuel=2e-4, w_align=0.2,
        dt=6000.0, max_steps=30000,
        thrust_scale_range=(100.0, 150.0),
    )

    tasks = fixed_tasks()

    controllers = [
        ("zero", ZeroThrustController()),
        ("greedy_energy_rt", GreedyEnergyRTController(
            k_e=0.9, k_rp=0.10, k_rd=0.60,
            t_clip=0.41, a_max_lo=0.048, a_max_hi=0.43,
            dead_r_in=0.020, dead_r_out=0.017,
            dead_v_in=0.040, dead_v_out=0.033,
            v_des_min=0.82, v_des_max=1.19
        )),
        # ECO upper bound (fuel-aware)
        ("expert_eco", ExpertBaseline(
            radial_gain=3.4, tangential_gain=3.6, damping_gain=9.2,
            band_in=1.35, band_out=2.00,
            stop_in_band=True, fire_frac=0.65, bang_bang=False,
            min_coast_steps=2, min_burst_steps=2,
            a_cap_scale=1.00, label="expert_eco"
        )),
        # FAST upper bound (tiny fuel trim to flip return positive while keeping SR)
        ("expert_fast", ExpertBaseline(
            radial_gain=3.7, tangential_gain=3.9, damping_gain=9.2,
            band_in=1.38, band_out=2.08,           # widened band to reduce chatter
            stop_in_band=True, fire_frac=0.76,     # higher firing gate to save fuel
            bang_bang=False,
            min_coast_steps=2, min_burst_steps=1,  # short bursts retained
            a_cap_scale=1.14,                      # cap scaled to preserve SR
            label="expert_fast"
        )),
    ]

    headers = [
        "controller", "task_id", "SR_ep", "return", "steps",
        "r_err", "v_err", "align", "fuel_used",
        "violations", "success",
        "ended_by_max", "ended_by_violation",
        "target_radius", "e", "mass", "thrust_limit"
    ]
    all_rows = []

    # Evaluate all controllers
    for name, ctrl in controllers:
        rows = eval_controller(name, ctrl, env, tasks, randN=20, seed=999)
        all_rows.extend(rows)

    # Save CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    # Aggregate + console summary
    by_name = {}
    for r in all_rows:
        by_name.setdefault(r[0], []).append(r)

    print("== Battery Summary ==")
    for name, rows in by_name.items():
        rows = np.array(rows, dtype=object)
        success_mask = rows[:, 10].astype(bool)

        sr_all   = np.mean(success_mask.astype(float))
        mean_ret = np.mean(rows[:, 3].astype(float))
        mean_rerr = np.mean(rows[:, 5].astype(float))
        mean_verr = np.mean(rows[:, 6].astype(float))
        mean_align = np.mean(rows[:, 7].astype(float))
        mean_fuel  = np.mean(rows[:, 8].astype(float))
        pct_max  = 100.0 * np.mean(rows[:, 11].astype(int))
        pct_violate = 100.0 * np.mean(rows[:, 12].astype(int))

        if success_mask.any():
            succ_fuels = rows[success_mask, 8].astype(float)
            fuel_succ_mean = np.mean(succ_fuels)
            fuel_succ_median = np.median(succ_fuels)
            fuel_line = f"fuel_succ(mean/median)={fuel_succ_mean:.1f}/{fuel_succ_median:.1f}"
        else:
            fuel_line = "fuel_succ(mean/median)=n/a"

        print(f"{name:15s} | SR={sr_all:.3f} | ret={mean_ret:.1f} | r_err={mean_rerr:.3e} | "
              f"v_err={mean_verr:.3e} | align={mean_align:.3f} | fuel(all)={mean_fuel:.1f} | {fuel_line}")
        print(f"  ends: max_steps={pct_max:.1f}% | violation={pct_violate:.1f}%")

        def _sort_key(row):
            success = bool(row[10]); fuel = float(row[8]); rerr = float(row[5])
            return (0 if not success else 1, -fuel, rerr)
        worst = sorted(rows.tolist(), key=_sort_key)[:3]
        worst_list = ", ".join([
            (f"{w[1]}(r_err={float(w[5]):.3e}, align={float(w[7]):.2f}, "
             f"rt={float(w[13]):.2e}, e={float(w[14]):.2f}, mass={float(w[15]):.0f}, "
             f"thrust={float(w[16]):.2f})")
            for w in worst
        ])
        print(f"  worst tasks: {worst_list}")

    # AB compare (Agent vs Experts)
    target_agent_name = "greedy_energy_rt"
    target_rows = by_name.get(target_agent_name, [])
    if not target_rows:
        print(f"[AB] skip: no rows for agent '{target_agent_name}'")
        print(f"CSV saved -> {CSV_PATH}")
        return

    agent_records, task_list = [], []
    for row in target_rows:
        task_name = str(row[1])
        ret = float(row[3]); r_err = float(row[5]); v_err = float(row[6])
        align = float(row[7]); steps = int(row[4]); violation = float(row[9])
        rt = float(row[13]); e = float(row[14]); mass = float(row[15]); thrust = float(row[16])

        agent_records.append(BatteryRecord(
            name=task_name, ret=ret, r_err=r_err, v_err=v_err, align=align,
            steps=steps, violation=violation,
            rt=rt, e=e, mass=mass, thrust=thrust, seed=-1,
            task_cfg=dict(name=task_name, rt=rt, e=e, mass=mass, thrust=thrust)
        ))
        task_list.append(dict(
            name=task_name,
            rt=rt, e=e, mass=mass, thrust=thrust, seed=-1,     # Expert side
            target_radius=rt, thrust_limit=thrust              # Env side
        ))

    def make_env_fn(task_cfg):
        return OrbitEnvMT(
            rerr_thr=0.015, verr_thr=0.030, align_thr=0.97, stable_steps=160,
            w_fuel=2e-4, w_align=0.2,
            dt=6000.0, max_steps=30000,
            thrust_scale_range=(100.0, 150.0),
        )

    # ECO variant for AB curves
    def make_expert_fn_eco(task_cfg):
        env_T = task_cfg.get("thrust", task_cfg.get("thrust_limit", 125.0))
        a_cap = _map_env_thrust_to_expert(env_T) * 1.00
        expert_ctrl = ExpertController(
            target_radius=task_cfg["rt"],
            mass=task_cfg.get("mass", 721.9),
            thrust_limit=a_cap,
            radial_gain=3.4, tangential_gain=3.6, damping_gain=9.2,
            enable_damping=True
        )
        return ExpertAdapter(expert_ctrl, pos_idx=(0, 1), vel_idx=(2, 3),
                             quiet=True, stop_in_band=True,
                             band_in_scale=1.35, band_out_scale=2.00,
                             a_cap=a_cap, fire_frac=0.65, bang_bang=False,
                             min_coast_steps=2, min_burst_steps=2)

    # FAST variant for AB curves (synced with controllers)
    def make_expert_fn_fast(task_cfg):
        env_T = task_cfg.get("thrust", task_cfg.get("thrust_limit", 125.0))
        a_cap = _map_env_thrust_to_expert(env_T) * 1.14
        expert_ctrl = ExpertController(
            target_radius=task_cfg["rt"],
            mass=task_cfg.get("mass", 721.9),
            thrust_limit=a_cap,
            radial_gain=3.7, tangential_gain=3.9, damping_gain=9.2,
            enable_damping=True
        )
        return ExpertAdapter(expert_ctrl, pos_idx=(0, 1), vel_idx=(2, 3),
                             quiet=True, stop_in_band=True,
                             band_in_scale=1.38, band_out_scale=2.08,   # synced
                             a_cap=a_cap, fire_frac=0.76,               # synced
                             bang_bang=False,
                             min_coast_steps=2, min_burst_steps=1)

    # Run AB for ECO
    ab_out_eco = run_ab_compare(
        agent_records=agent_records,
        task_list=task_list,
        make_env_fn=make_env_fn,
        make_expert_fn=make_expert_fn_eco,
        export_k=3,
        worst_by="ret",
        out_root="ab/day32_eco",
        metrics_for_curve=("ret", "r_err"),
    )
    print("[AB][eco] Saved plots:", ab_out_eco["plots"])
    print("[AB][eco] Worst-3 (by ret):", [w.name for w in ab_out_eco["worst"]])

    # Run AB for FAST
    ab_out_fast = run_ab_compare(
        agent_records=agent_records,
        task_list=task_list,
        make_env_fn=make_env_fn,
        make_expert_fn=make_expert_fn_fast,
        export_k=3,
        worst_by="ret",
        out_root="ab/day32_fast",
        metrics_for_curve=("ret", "r_err"),
    )
    print("[AB][fast] Saved plots:", ab_out_fast["plots"])
    print("[AB][fast] Worst-3 (by ret):", [w.name for w in ab_out_fast["worst"]])

    print(f"CSV saved -> {CSV_PATH}")


if __name__ == "__main__":
    main()
