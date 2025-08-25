from envs.multi_orbit_env import MultiOrbitEnv
from simulator.types import SimConfig, State, Goal

def test_env_step_smoke():
    """Smoke test to ensure env.step() executes without crashing."""
    cfg = SimConfig(mu=3.986e14, dt=1.0, max_steps=100)
    env = MultiOrbitEnv(cfg)

    # Adjust to your real API if different:
    s = env.reset_to_circular(r0=1.0e7, mass=720.0)

    s2, r, done, info = env.step((0.0, 0.0))
    assert s2 is not None
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
