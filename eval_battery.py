import os
import csv
import numpy as np
from envs.orbit_env_mt import OrbitEnvMT
from baselines.zero_thrust import ZeroThrustController
from baselines.greedy_energy_rt import GreedyEnergyRTController  # Energy-shaping baseline


# Paths and output CSV location
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "battery_day31.csv")  # was day30


def fixed_tasks():
    """
    Define 5 fixed benchmark tasks for SR@T evaluation.
    This ensures reproducible performance comparisons across runs.
    Each task specifies target orbit radius, eccentricity, spacecraft mass, and thrust limit.
    """
    return [
        dict(target_radius=5e11, e=0.0,  mass=720.0,  thrust_limit=1.0),  # Near-circular, mid radius
        dict(target_radius=7e11, e=0.2,  mass=720.0,  thrust_limit=1.0),  # Elliptic, e=0.2
        dict(target_radius=5e11, e=0.05, mass=720.0,  thrust_limit=0.4),  # Low thrust limit
        dict(target_radius=5e11, e=0.1,  mass=1000.0, thrust_limit=1.0),  # Heavy mass
        dict(target_radius=2e12, e=0.0,  mass=720.0,  thrust_limit=1.0),  # Far target radius
    ]


def rollout(env: OrbitEnvMT, controller, max_episodes=1, seed=123, pre_reset=False, initial_obs=None):
    """
    Run the environment for one or more episodes with the given controller.
    If pre_reset=True, the caller must supply 'initial_obs' from env.reset(...),
    and should have already called controller.set_task(...) if needed.
    Returns:
        sr: success rate over episodes (0.0 - 1.0)
        totals: list of (episode_reward, steps, info) tuples
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


def eval_controller(name, controller, env: OrbitEnvMT, tasks, randN=20, seed=999):
    """
    Evaluate a given controller on:
        - Fixed battery: the 5 predefined tasks (deterministic)
        - Random battery: N randomly generated tasks
    Returns:
        rows: list of result rows, one per task
    """
    rows = []

    # Evaluate on fixed tasks
    for i, t in enumerate(tasks):
        obs0 = env.reset(task=t, seed=seed + i)
        if hasattr(controller, "set_task"):
            controller.set_task(t)
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

    # Evaluate on random tasks
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


def main():
    # Use a large dt so one episode spans ~5.7 years (close to ~6.1-year period at r~5e11 m).
    # dt = 6000 s and max_steps = 30000 -> 1.8e8 s ≈ 5.7 years.
    env = OrbitEnvMT(
        # Stage B: tightened success band
        rerr_thr=0.015, verr_thr=0.030, align_thr=0.97, stable_steps=160,

        # Rewards
        w_fuel=2e-4,
        w_align=0.2,

        # Time scale
        dt=6000.0,
        max_steps=30000,

        # Controller authority
        thrust_scale_range=(100.0, 150.0),
    )

    tasks = fixed_tasks()

    controllers = [
        ("zero", ZeroThrustController()),
        ("greedy_energy_rt", GreedyEnergyRTController(
            k_e=0.9,
            k_rp=0.10,
            k_rd=0.60,
            t_clip=0.41,
            a_max_lo=0.048,
            a_max_hi=0.43,
            # tighten deadzone near Stage B thresholds
            dead_r_in=0.020, dead_r_out=0.017,
            dead_v_in=0.040, dead_v_out=0.033,
            v_des_min=0.82, v_des_max=1.19
        ))
    ]

    headers = [
        "controller", "task_id", "SR_ep", "return", "steps",
        "r_err", "v_err", "align", "fuel_used",
        "violations", "success",
        "ended_by_max", "ended_by_violation",
        "target_radius", "e", "mass", "thrust_limit"
    ]
    all_rows = []

    for name, ctrl in controllers:
        rows = eval_controller(name, ctrl, env, tasks, randN=20, seed=999)
        all_rows.extend(rows)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    by_name = {}
    for r in all_rows:
        by_name.setdefault(r[0], []).append(r)

    print("== Battery Summary ==")
    for name, rows in by_name.items():
        rows = np.array(rows, dtype=object)
        success_mask = rows[:, 10].astype(bool)

        sr_all      = np.mean(success_mask.astype(float))
        mean_ret    = np.mean(rows[:, 3].astype(float))
        mean_rerr   = np.mean(rows[:, 5].astype(float))
        mean_verr   = np.mean(rows[:, 6].astype(float))
        mean_align  = np.mean(rows[:, 7].astype(float))
        mean_fuel   = np.mean(rows[:, 8].astype(float))
        pct_max     = 100.0 * np.mean(rows[:, 11].astype(int))
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

        # Failures first → higher fuel → larger r_err
        def _sort_key(row):
            success = bool(row[10])
            fuel    = float(row[8])
            rerr    = float(row[5])
            return (0 if not success else 1, -fuel, rerr)

        worst = sorted(rows.tolist(), key=_sort_key)[:3]
        worst_list = ", ".join([
            (f"{w[1]}(r_err={float(w[5]):.3e}, align={float(w[7]):.2f}, "
             f"rt={float(w[13]):.2e}, e={float(w[14]):.2f}, mass={float(w[15]):.0f}, "
             f"thrust={float(w[16]):.2f})")
            for w in worst
        ])
        print(f"  worst tasks: {worst_list}")

    print(f"CSV saved -> {CSV_PATH}")


if __name__ == "__main__":
    main()
