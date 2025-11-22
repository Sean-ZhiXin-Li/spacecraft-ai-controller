import os
import numpy as np

from tools.metrics.energy_view import EnergyConfig, analyze_replay
from envs.orbit_env import OrbitEnv
from controller.expert_controller import ExpertController


# Stress scenarios: only our own keys, NOT passed into OrbitEnv.__init__
STRESS_SCENARIOS = [
    {
        "name": "weak_thrust_far",
        "description": "Start far from target orbit with reduced thrust scale.",
        "config": {
            "start_mode": "default",      # use your default reset pattern
            "initial_radius_factor": 1.5, # r0 = 1.5 * target_radius
            "vel_angle_deg": 30.0,        # keep similar to default
            "vel_scale": 1.0,
            "thrust_scale_factor": 0.3,   # weaker thrust than default
        },
    },
    {
        "name": "oscillation_noise",
        "description": "Add action noise so thrust direction oscillates.",
        "config": {
            "start_mode": "default",
            "action_noise_std": 0.4,      # noise applied to controller action
        },
    },
    {
        "name": "misaligned_entry",
        "description": "Start at target radius but with misaligned velocity.",
        "config": {
            "start_mode": "default",
            "initial_radius_factor": 1.0, # at target radius
            "vel_angle_deg": 75.0,        # strongly misaligned w.r.t. tangential
            "vel_scale": 0.7,             # wrong speed
        },
    },
]


# Helper: configure env for a given scenario
def setup_env_for_scenario(env: OrbitEnv, scenario_cfg: dict):
    """
    Use OrbitEnv.set_physical_params and set_initial_state to realize
    the stress scenario. We do NOT pass these keys into __init__.
    """
    cfg = scenario_cfg or {}

    # 1) Base reset with given start_mode
    start_mode = cfg.get("start_mode", "default")
    obs, info = env.reset(start_mode=start_mode)

    # 2) Weak thrust: scale thrust_scale via set_physical_params
    if "thrust_scale_factor" in cfg:
        factor = float(cfg["thrust_scale_factor"])
        base_thrust = env.thrust_scale
        new_thrust = base_thrust * factor
        env.set_physical_params(thrust_newton=new_thrust)
        # reset obs after changing physical params? state不变，但 thrust_scale 变没关系
        obs = env._get_obs()

    # 3) Custom initial radius / velocity using set_initial_state
    init_state = {}

    if "initial_radius_factor" in cfg:
        r0 = float(cfg["initial_radius_factor"]) * env.target_radius
        init_state["pos"] = [0.0, r0]

    if "vel_angle_deg" in cfg:
        init_state["vel_angle_deg"] = float(cfg["vel_angle_deg"])

    if "vel_scale" in cfg:
        init_state["vel_scale"] = float(cfg["vel_scale"])

    if init_state:
        env.set_initial_state(init_state)
        obs = env._get_obs()

    return obs


# Run one episode and export replay + energy metrics
def run_single_episode(env: OrbitEnv,
                       controller: ExpertController,
                       scenario_cfg: dict,
                       out_dir: str,
                       energy_cfg: EnergyConfig,
                       run_idx: int):
    os.makedirs(out_dir, exist_ok=True)

    # Prepare env according to scenario: position, velocity, thrust, etc.
    obs = setup_env_for_scenario(env, scenario_cfg)
    done = False

    traj_obs = []
    traj_actions = []

    # Scenario-specific action noise
    action_noise_std = float(scenario_cfg.get("action_noise_std", 0.0))

    while not done:
        # 1) base action from controller
        action = controller.act(obs)

        # 2) optional noise injection (oscillation_noise scenario)
        if action_noise_std > 0.0:
            noise = np.random.normal(loc=0.0,
                                     scale=action_noise_std,
                                     size=np.shape(action))
            action = action + noise

        # keep in reasonable range; env.step 里还有 clip
        action = np.asarray(action, dtype=np.float32)

        traj_obs.append(obs)
        traj_actions.append(action)

        # NOTE: your OrbitEnv.step uses classic Gym API: obs, reward, done, info
        obs, reward, done, info = env.step(action)

    # Save replay
    replay_path = os.path.join(out_dir, "replay.npz")
    np.savez_compressed(
        replay_path,
        obs=np.asarray(traj_obs, dtype=np.float32),
        actions=np.asarray(traj_actions, dtype=np.float32),
    )

    # Save energy metrics
    metrics_path = os.path.join(out_dir, "metrics_energy.json")
    analyze_replay(
        replay_path,
        energy_cfg,
        save_json=metrics_path,
    )


# Main
def main():
    # Keep dt consistent with OrbitEnv for correct energy integration
    energy_cfg = EnergyConfig(
        G=6.67430e-11,
        M=1.989e30,
        m=722.0,
        dt=10.0,
        target_radius=7.5e12,
        energy_eps=0.05,
        angmom_eps=0.05,
    )

    controller = ExpertController(
        target_radius=energy_cfg.target_radius,
        G=energy_cfg.G,
        M=energy_cfg.M,
        mass=energy_cfg.m,
        radial_gain=4.0,
        tangential_gain=5.0,
        damping_gain=6.0,
        thrust_limit=20.0,
        enable_damping=True,
    )

    MAX_STEPS_STRESS = 20000

    for scenario in STRESS_SCENARIOS:
        name = scenario["name"]
        cfg = scenario.get("config", {})

        for run_idx in range(1, 4):
            env = OrbitEnv(
                G=energy_cfg.G,
                M=energy_cfg.M,
                mass=energy_cfg.m,
                dt=energy_cfg.dt,
                max_steps=MAX_STEPS_STRESS,
                target_radius=energy_cfg.target_radius,
                thrust_scale=3000.0,
            )

            out_dir = f"logs/new_week_3/{name}/run_{run_idx:02d}"
            print(f"[Running] {name} | run {run_idx:02d} -> {out_dir}")
            run_single_episode(env, controller, cfg, out_dir, energy_cfg, run_idx)

    print("Stress test finished.")



if __name__ == "__main__":
    main()
