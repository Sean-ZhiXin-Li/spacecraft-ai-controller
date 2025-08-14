import os
import csv
import numpy as np
from envs.orbit_env_mt import OrbitEnvMT
from baselines.zero_thrust import ZeroThrustController
from baselines.greedy_energy_rt import GreedyEnergyRTController  # Energy-shaping baseline

# ===============================
# Paths and output CSV location
# ===============================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "battery_day30.csv")


def fixed_tasks():
    """
    Define 5 fixed benchmark tasks for SR@T evaluation.
    This ensures reproducible performance comparisons across runs.
    Each task specifies target orbit radius, eccentricity, spacecraft mass, and thrust limit.
    """
    return [
        dict(target_radius=5e11, e=0.0, mass=720.0, thrust_limit=1.0),   # Near-circular, mid radius
        dict(target_radius=7e11, e=0.2, mass=720.0, thrust_limit=1.0),   # Elliptic, e=0.2
        dict(target_radius=5e11, e=0.05, mass=720.0, thrust_limit=0.4),  # Low thrust limit
        dict(target_radius=5e11, e=0.1, mass=1000.0, thrust_limit=1.0),  # Heavy mass
        dict(target_radius=2e12, e=0.0, mass=720.0, thrust_limit=1.0),   # Far target radius
    ]


def rollout(env: OrbitEnvMT, controller, max_episodes=1, seed=123):
    """
    Run the environment for one or more episodes with the given controller.
    Collects total reward, steps taken, and final episode info.
    Returns:
        sr: success rate over episodes (0.0 - 1.0)
        totals: list of (episode_reward, steps, info) tuples
    """
    sr = 0
    totals = []
    for ep in range(max_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            a = controller.act(obs)  # Controller outputs action
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
        env.reset(task=t, seed=seed + i)
        sr, totals = rollout(env, controller, max_episodes=1, seed=seed + i)
        rew, steps, info = totals[0]

        # Diagnostics: how the episode ended
        ended_by_max = int((steps >= env.max_steps) and (not info["success"]) and (info["violations"] == 0))
        ended_by_violation = int(info["violations"] > 0)

        # Append with task parameters so CSV has full context
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
        env.reset(task=None, seed=int(rng.integers(0, 10_000_000)))
        sr, totals = rollout(env, controller, max_episodes=1, seed=seed + 1000 + j)
        rew, steps, info = totals[0]

        ended_by_max = int((steps >= env.max_steps) and (not info["success"]) and (info["violations"] == 0))
        ended_by_violation = int(info["violations"] > 0)

        # IMPORTANT: also include task parameters for random tasks
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
        # Stage A: slightly tighter success band
        rerr_thr=0.018, verr_thr=0.035, align_thr=0.96, stable_steps=120,

        # Rewards
        w_fuel=2e-4,      # fuel penalty scales with dt inside env.step()
        w_align=0.2,

        # Time scale
        dt=6000.0,        # 6000 seconds per step (~1.67 hours/step)
        max_steps=30000,  # up to ~5.7 years per episode

        # Controller authority
        thrust_scale_range=(100.0, 150.0),
        # Optional safety window adjustment:
        # min_radius_factor=0.15, max_radius_factor=6.0,
    )

    tasks = fixed_tasks()

    controllers = [
        ("zero", ZeroThrustController()),
        ("greedy_energy_rt", GreedyEnergyRTController(
            # Keep energy shaping; add stronger damping and tighter clamps/caps
            k_e=0.9,  # energy shaping gain (unchanged)
            k_rp=0.08,  # ↓ smaller radial P: let energy do more work
            k_rd=0.55,  # ↑ stronger radial damping to avoid oscillations
            t_clip=0.40,  # ↓ tighter tangential correction to save fuel
            a_max_lo=0.045,  # ↓ smaller near-target cap (more coasting)
            a_max_hi=0.45,  # keep modest far cap (won't starve far cases too much)
            # Hysteresis deadzone: enter/exit slightly wider than success band
            dead_r_in=0.040, dead_r_out=0.030,
            dead_v_in=0.080, dead_v_out=0.060,
            # Mild saturation of desired tangential speed
            v_des_min=0.82, v_des_max=1.18
        ))
    ]

    # CSV headers (must match the appended row length)
    headers = [
        "controller", "task_id", "SR_ep", "return", "steps",
        "r_err", "v_err", "align", "fuel_used",
        "violations", "success",
        "ended_by_max", "ended_by_violation",
        "target_radius", "e", "mass", "thrust_limit"
    ]
    all_rows = []

    # Evaluate each controller
    for name, ctrl in controllers:
        rows = eval_controller(name, ctrl, env, tasks, randN=20)
        all_rows.extend(rows)

    # Save to CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    # Console summary and diagnostics
    by_name = {}
    for r in all_rows:
        name = r[0]
        by_name.setdefault(name, []).append(r)

    print("== Battery Summary ==")
    for name, rows in by_name.items():
        rows = np.array(rows, dtype=object)
        success_mask = rows[:, 10].astype(bool)

        sr_all     = np.mean(success_mask.astype(float))
        mean_ret   = np.mean(rows[:, 3].astype(float))
        mean_rerr  = np.mean(rows[:, 5].astype(float))
        mean_verr  = np.mean(rows[:, 6].astype(float))
        mean_align = np.mean(rows[:, 7].astype(float))
        mean_fuel  = np.mean(rows[:, 8].astype(float))
        pct_max    = 100.0 * np.mean(rows[:, 11].astype(int))  # % ended due to max_steps
        pct_violate= 100.0 * np.mean(rows[:, 12].astype(int))  # % ended due to violation

        # Fuel stats for successful episodes only
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

        # Show worst 3 tasks by final radius error
        idx_sorted = np.argsort(rows[:, 5].astype(float))[::-1]
        worst = rows[idx_sorted[:3]]
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
