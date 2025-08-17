import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from envs.orbit_env_mt import OrbitEnvMT
from baselines.greedy_energy_rt import GreedyEnergyRTController
from baselines.zero_thrust import ZeroThrustController
from controller.expert_controller import ExpertController

# ---- Environment config (aligned with eval_battery.py) ----
ENV_KW = dict(
    rerr_thr=0.015,
    verr_thr=0.030,
    align_thr=0.97,
    stable_steps=160,
    w_fuel=2e-4,
    w_align=0.2,
    dt=6000.0,
    max_steps=30000,
    thrust_scale_range=(100.0, 150.0),
)

# ------------------------ ExpertAdapter --------------------------
class ExpertAdapter:
    """
    Adapter that:
      - Feeds PHYSICAL (position, velocity) to ExpertController.
      - Uses stop-in-band with hysteresis to coast when near target.
      - Applies HARD magnitude clamp on action (||a|| <= a_cap).
      - Applies a firing GATE: if ||a|| < fire_frac * a_cap, output zeros.
      - Optional bang-bang mode: when firing, snap ||a|| to a_cap.
      - Auto de-normalizes from observations if needed using (mu, rt).
    """
    def __init__(
        self,
        expert_ctrl,
        a_cap,
        fire_frac=0.45,
        pos_idx=(0, 1),
        vel_idx=(2, 3),
        stop_in_band=True,
        band_in_scale=1.3,
        band_out_scale=1.6,
        verbose=True,
        bang_bang=False,
    ):
        self.ctrl = expert_ctrl
        self.env = None
        self.a_cap = float(a_cap)
        self.fire_frac = float(fire_frac)
        self.pos_idx = tuple(pos_idx)
        self.vel_idx = tuple(vel_idx)
        self.stop_in_band = bool(stop_in_band)
        self.band_in_scale = float(band_in_scale)
        self.band_out_scale = float(band_out_scale)
        self.verbose = verbose
        self.bang_bang = bool(bang_bang)

        self._captured = False
        self._warned = False
        # Stats
        self.on_steps = 0
        self.all_steps = 0

    def bind_env(self, env):
        self.env = env
        if self.verbose:
            print(
                f"[AB] bind_env; a_cap={self.a_cap:.6g}, fire_frac={self.fire_frac:.2f}, "
                f"hyst(in,out)=({self.band_in_scale:.2f},{self.band_out_scale:.2f}), "
                f"bang_bang={self.bang_bang}"
            )

    def get_action(self, state):
        r, v = self._extract_rv(state)

        # Coast with hysteresis when inside the band
        if self.stop_in_band and (self.env is not None):
            mu = self._get_mu()
            rt = self._get_target_radius()
            if (mu is not None) and (rt is not None):
                v_circ = (mu / (rt + 1e-12)) ** 0.5
                rerr = abs(np.linalg.norm(r) - rt) / (rt + 1e-12)
                verr = abs(v_circ - self._vt(v, r)) / (v_circ + 1e-12)

                rin = getattr(self.env, "rerr_thr", 0.015) * self.band_in_scale
                vin = getattr(self.env, "verr_thr", 0.030) * self.band_in_scale
                rout = getattr(self.env, "rerr_thr", 0.015) * self.band_out_scale
                vout = getattr(self.env, "verr_thr", 0.030) * self.band_out_scale

                if self._captured:
                    # Leave band if error drifts out
                    if (rerr >= rout) or (verr >= vout):
                        self._captured = False
                        if self.verbose:
                            print(f"[AB] exit band: rerr={rerr:.4f}, verr={verr:.4f}")
                else:
                    # Enter band if both errors are small enough
                    if (rerr <= rin) and (verr <= vin):
                        self._captured = True
                        if self.verbose:
                            print(f"[AB] enter band: rerr={rerr:.4f}, verr={verr:.4f}")

                # Inside band → coast
                if self._captured:
                    self.all_steps += 1
                    return np.zeros(2, dtype=float)

        # Raw expert output
        a = np.asarray(self.ctrl(0.0, r, v), dtype=float).reshape(-1)

        # HARD clamp by norm: ||a|| <= a_cap
        n = np.linalg.norm(a) + 1e-12
        if n > self.a_cap:
            a = a * (self.a_cap / n)
            n = self.a_cap

        # Firing gate: suppress tiny thrusts that waste fuel in "on/off" models
        if n < self.fire_frac * self.a_cap:
            a[:] = 0.0
            n = 0.0

        # Optional bang-bang: when firing, snap magnitude to a_cap
        if n > 0.0 and self.bang_bang:
            a = a * (self.a_cap / (np.linalg.norm(a) + 1e-12))

        # Stats
        self.on_steps += int(np.linalg.norm(a) > 0.0)
        self.all_steps += 1
        return a[:2]

    # --- helpers ---
    def _vt(self, v_vec, r_vec):
        """Tangential component of velocity."""
        r = np.linalg.norm(r_vec) + 1e-12
        er = r_vec / r
        et = np.array([-er[1], er[0]])
        return float(np.dot(v_vec, et))

    def _extract_rv(self, state):
        """Best-effort extraction of physical (r, v)."""
        if self.env is not None:
            # 1) Explicit environment methods
            for meth in ("get_raw_rv", "get_state_vectors", "raw_rv"):
                if hasattr(self.env, meth):
                    try:
                        r, v = getattr(self.env, meth)()
                        if self.verbose:
                            print("[AB] using env method:", meth)
                        return np.asarray(r).reshape(-1)[:2], np.asarray(v).reshape(-1)[:2]
                    except Exception:
                        pass

            # 2) Common attribute names
            pairs = [
                ("pos", "vel"),
                ("r", "v"),
                ("position", "velocity"),
                ("x", "v"),
                ("x", "xdot"),
                ("r_vec", "v_vec"),
            ]
            for pa, va in pairs:
                if hasattr(self.env, pa) and hasattr(self.env, va):
                    if self.verbose:
                        print(f"[AB] using env attrs: ({pa}, {va})")
                    r = np.asarray(getattr(self.env, pa)).reshape(-1)[:2]
                    v = np.asarray(getattr(self.env, va)).reshape(-1)[:2]
                    return r, v

            # 3) Inside env.state dict
            if hasattr(self.env, "state") and isinstance(self.env.state, dict):
                st = self.env.state
                for pa, va in pairs:
                    if pa in st and va in st:
                        if self.verbose:
                            print(f"[AB] using env.state keys: ({pa}, {va})")
                        r = np.asarray(st[pa]).reshape(-1)[:2]
                        v = np.asarray(st[va]).reshape(-1)[:2]
                        return r, v

            # 4) De-normalize from observation using (mu, rt)
            mu = self._get_mu()
            rt = self._get_target_radius()
            if (mu is not None) and (rt is not None):
                obs = np.asarray(state).reshape(-1)
                r = obs[list(self.pos_idx)] * float(rt)
                v_ref = (float(mu) / float(rt)) ** 0.5
                v = obs[list(self.vel_idx)] * v_ref
                if not self._warned:
                    print(f"[AB] denorm via mu,rt; v_ref={v_ref:.3e}")
                    self._warned = True
                return r.astype(float), v.astype(float)

        # 5) Fallback: slice raw obs by indices (not recommended)
        obs = np.asarray(state).reshape(-1)
        try:
            r = obs[list(self.pos_idx)]
            v = obs[list(self.vel_idx)]
        except Exception:
            r = obs[:2]
            v = obs[2:4] if obs.shape[0] >= 4 else obs[2:]
        if not self._warned:
            print("[AB][warn] using obs indices without denorm")
            self._warned = True
        return r, v

    def _get_mu(self):
        """Try to read gravitational parameter mu = GM."""
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
        """Try to read target radius from env.task."""
        if self.env is None:
            return None
        t = getattr(self.env, "task", None)
        if isinstance(t, dict):
            if "target_radius" in t:
                return float(t["target_radius"])
            if "rt" in t:
                return float(t["rt"])
        return None


# --------------------------- rollout -----------------------------
def rollout(env: OrbitEnvMT, controller, seed=123, initial_obs=None):
    """
    Run a single episode and return (return, steps, last_info).
    """
    if initial_obs is None:
        obs = env.reset(seed=seed)
        if hasattr(controller, "set_task"):
            controller.set_task(env.task)
    else:
        obs = initial_obs
        if hasattr(controller, "set_task"):
            controller.set_task(env.task)

    done = False
    ep_reward = 0.0
    steps = 0
    last_info = {}
    while not done:
        if hasattr(controller, "get_action"):
            a = controller.get_action(obs)
        elif hasattr(controller, "act"):
            a = controller.act(obs)
        else:
            a = controller(obs)
        obs, r, done, info = env.step(a)
        ep_reward += r
        steps += 1
        last_info = info

    # If controller is ExpertAdapter, print firing ratio for diagnostics
    if isinstance(controller, ExpertAdapter) and controller.all_steps > 0:
        frac = controller.on_steps / controller.all_steps
        print(f"[AB] Expert thrust_on_frac={frac:.3f} ({controller.on_steps}/{controller.all_steps})")

    return ep_reward, steps, last_info


# --------------------------- agent zoo ---------------------------
def build_agent(name: str):
    if name == "greedy_energy_rt":
        return GreedyEnergyRTController(
            k_e=0.9,
            k_rp=0.10,
            k_rd=0.60,
            t_clip=0.41,
            a_max_lo=0.048,
            a_max_hi=0.43,
            dead_r_in=0.020,
            dead_r_out=0.017,
            dead_v_in=0.040,
            dead_v_out=0.033,
            v_des_min=0.82,
            v_des_max=1.19,
        )
    if name == "zero":
        return ZeroThrustController()
    raise ValueError(f"Unknown agent '{name}'")


# ---------------------------- plotting ---------------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_bars(names, values, ylabel, out_path, zero_line=True):
    _ensure_dir(os.path.dirname(out_path))
    fig = plt.figure(figsize=(10, 4.5))
    x = np.arange(len(names))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(ylabel + " (Expert − Agent)")
    if zero_line:
        plt.axhline(0.0, linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ------------------------------ I/O ------------------------------
def load_task_specs(in_dir):
    """
    Load task JSON specs; normalize key names across sources.
    Expected keys per task: rt/target_radius, thrust/thrust_limit, mass, e, name, seed.
    """
    tasks = []
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(".json"):
            continue
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            spec = json.load(f)
        rt = spec.get("rt", spec.get("target_radius"))
        thrust = spec.get("thrust", spec.get("thrust_limit", 125.0))
        mass = spec.get("mass", 721.9)
        e = spec.get("e", 0.0)
        name = spec.get("name", os.path.splitext(fn)[0])
        tasks.append(
            dict(
                name=name,
                rt=float(rt),
                e=float(e),
                mass=float(mass),
                thrust=float(thrust),
                target_radius=float(rt),
                thrust_limit=float(thrust),
                seed=int(spec.get("seed", -1)),
            )
        )
    return tasks


# ------------------------------ main -----------------------------
def main():
    ap = argparse.ArgumentParser("Replay worst tasks w/ gating, hard clamp, and optional bang-bang")
    ap.add_argument("--in_dir", default="ab/day32/worst/task_specs", type=str)
    ap.add_argument("--out_root", default="ab/day33", type=str)
    ap.add_argument("--agent", default="greedy_energy_rt", type=str)

    # Acceleration cap range used AFTER converting thrust(N)/mass(kg).
    # IMPORTANT: a_lo default is 0.0 so tiny-thrust tasks are NOT floored up.
    ap.add_argument("--a_lo", type=float, default=0.0)
    ap.add_argument("--a_hi", type=float, default=0.45)

    # Firing gate: if ||a|| < fire_frac * a_cap → set action to zero
    ap.add_argument("--fire_frac", type=float, default=0.45)

    # Hysteresis band (coast window)
    ap.add_argument("--band_in", type=float, default=1.3, help="hysteresis inner scale for (rerr, verr)")
    ap.add_argument("--band_out", type=float, default=1.6, help="hysteresis outer scale for (rerr, verr)")

    # Expert gains
    ap.add_argument("--radial_gain", type=float, default=4.0)
    ap.add_argument("--tangential_gain", type=float, default=5.0)
    ap.add_argument("--damping_gain", type=float, default=6.0)

    # Bang-bang (full magnitude when firing)
    ap.add_argument("--bang_bang", action="store_true")

    ap.add_argument("--seed", default=999, type=int)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks = load_task_specs(args.in_dir)
    if not tasks:
        raise FileNotFoundError(f"No task specs found in {args.in_dir}")

    out_csv = os.path.join(args.out_root, "replay", "replay_compare.csv")
    out_figdir = os.path.join(args.out_root, "replay", "figs")
    _ensure_dir(os.path.dirname(out_csv))
    _ensure_dir(out_figdir)

    env = OrbitEnvMT(**ENV_KW)
    agent = build_agent(args.agent)

    rows, names = [], []
    d_rets, d_rerrs, d_fuels = [], [], []

    print(
        f"[CFG] HARD-CLAMP + GATING; a in [{args.a_lo},{args.a_hi}] m/s^2, "
        f"fire_frac={args.fire_frac}, band=({args.band_in},{args.band_out}), "
        f"bang_bang={args.bang_bang}"
    )

    for i, t in enumerate(tasks):
        # Convert thrust(N) to raw accel and then clamp to [a_lo, a_hi]
        a_raw = t["thrust"] / max(t["mass"], 1e-9)
        a_cap = max(args.a_lo, min(args.a_hi, a_raw))
        floor_hit = a_raw < args.a_lo
        ceil_hit = a_raw > args.a_hi
        print(
            f"[MAP] {t['name']}: thrust={t['thrust']:.3g} N, mass={t['mass']:.3g} kg "
            f"-> a_raw={a_raw:.6g} m/s^2, a_cap={a_cap:.6g} "
            f"(floor_hit={floor_hit}, ceil_hit={ceil_hit})"
        )

        # Agent rollout
        obs0 = env.reset(
            task=dict(
                name=t["name"],
                target_radius=t["rt"],
                e=t["e"],
                mass=t["mass"],
                thrust_limit=t["thrust"],
            ),
            seed=args.seed + i,
        )
        if hasattr(agent, "set_task"):
            agent.set_task(env.task)
        a_ret, _, a_info = rollout(env, agent, seed=args.seed + i, initial_obs=obs0)

        # Expert rollout (same task)
        obs0b = env.reset(
            task=dict(
                name=t["name"],
                target_radius=t["rt"],
                e=t["e"],
                mass=t["mass"],
                thrust_limit=t["thrust"],
            ),
            seed=args.seed + i,
        )
        expert_core = ExpertController(
            target_radius=t["rt"],
            mass=t["mass"],
            thrust_limit=a_cap,  # even if ignored internally, adapter clamps output
            radial_gain=args.radial_gain,
            tangential_gain=args.tangential_gain,
            damping_gain=args.damping_gain,
            enable_damping=True,
        )
        expert = ExpertAdapter(
            expert_core,
            a_cap=a_cap,
            fire_frac=args.fire_frac,
            pos_idx=(0, 1),
            vel_idx=(2, 3),
            stop_in_band=True,
            band_in_scale=args.band_in,
            band_out_scale=args.band_out,
            verbose=True,
            bang_bang=args.bang_bang,
        )
        expert.bind_env(env)
        e_ret, _, e_info = rollout(env, expert, seed=args.seed + i, initial_obs=obs0b)

        # Deltas (Expert − Agent)
        d_ret = float(e_ret - a_ret)
        d_rerr = float(e_info["r_err"] - a_info["r_err"])
        d_fuel = float(e_info["fuel_used"] - a_info["fuel_used"])

        rows.append(
            dict(
                name=t["name"],
                agent=args.agent,
                agent_success=bool(a_info.get("success", False)),
                expert_success=bool(e_info.get("success", False)),
                agent_ret=float(a_ret),
                expert_ret=float(e_ret),
                d_ret=d_ret,
                agent_r_err=float(a_info["r_err"]),
                expert_r_err=float(e_info["r_err"]),
                d_r_err=d_rerr,
                agent_fuel=float(a_info["fuel_used"]),
                expert_fuel=float(e_info["fuel_used"]),
                d_fuel=d_fuel,
                rt=float(t["rt"]),
                e=float(t["e"]),
                mass=float(t["mass"]),
                thrust=float(t["thrust"]),
                thrust_map="force_auto+hard_clamp+gating" + ("+bangbang" if args.bang_bang else ""),
            )
        )
        names.append(t["name"])
        d_rets.append(d_ret)
        d_rerrs.append(d_rerr)
        d_fuels.append(d_fuel)

    # Save CSV
    import csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "agent",
                "agent_success",
                "expert_success",
                "agent_ret",
                "expert_ret",
                "d_ret",
                "agent_r_err",
                "expert_r_err",
                "d_r_err",
                "agent_fuel",
                "expert_fuel",
                "d_fuel",
                "rt",
                "e",
                "mass",
                "thrust",
                "thrust_map",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["name"],
                    r["agent"],
                    int(r["agent_success"]),
                    int(r["expert_success"]),
                    r["agent_ret"],
                    r["expert_ret"],
                    r["d_ret"],
                    r["agent_r_err"],
                    r["expert_r_err"],
                    r["d_r_err"],
                    r["agent_fuel"],
                    r["expert_fuel"],
                    r["d_fuel"],
                    r["rt"],
                    r["e"],
                    r["mass"],
                    r["thrust"],
                    r["thrust_map"],
                ]
            )
    print(f"[replay] CSV -> {out_csv}")

    # Save figures
    plot_bars(names, d_rets, "Δret (Expert − Agent)", os.path.join(out_figdir, "ab_ret.png"))
    plot_bars(names, d_rerrs, "Δr_err (Expert − Agent)", os.path.join(out_figdir, "ab_r_err.png"))
    plot_bars(names, d_fuels, "Δfuel (Expert − Agent)", os.path.join(out_figdir, "ab_fuel.png"))
    print(
        f"[replay] Figures -> "
        f"{[os.path.join(out_figdir, 'ab_ret.png'), os.path.join(out_figdir, 'ab_r_err.png'), os.path.join(out_figdir, 'ab_fuel.png')]}"
    )

    # Console summary table
    def _fmt(x):
        return f"{x:.3g}" if isinstance(x, float) else str(x)

    print("\nname, agent_success, expert_success, Δret, Δr_err, Δfuel")
    for r in rows:
        print(
            f"{r['name']:>12s}  {int(r['agent_success'])}  {int(r['expert_success'])}  "
            f"{_fmt(r['d_ret']):>8s}  {_fmt(r['d_r_err']):>8s}  {_fmt(r['d_fuel']):>10s}"
        )


if __name__ == "__main__":
    main()
