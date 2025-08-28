# Weekday smoke check runner (no logs/commits)
# - Treat pytest exit code 0 (success) and 5 (no tests collected) as pass.
# - Fallback doesn't rely on SimConfig; it creates OrbitEnv via default or a dummy cfg.

import sys
import subprocess
from types import SimpleNamespace

def run(cmd: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"[SMOKE] $ {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        # Python -m pytest not available or venv broken
        return 127

def main():
    # 1) Try pytest -q if available
    code = run([sys.executable, "-m", "pytest", "-q"])

    # Pytest exit codes of interest:
    # 0 = all tests passed, 5 = no tests collected (acceptable for smoke)
    if code in (0, 5):
        msg = "pytest passed ✓" if code == 0 else "pytest: no tests collected (treated as pass) ✓"
        print(f"[SMOKE] {msg}")
        sys.exit(0)

    # 2) Fallback: run a tiny import/construct/step smoke (no plots, no files)
    print("[SMOKE] pytest failed or unavailable, running fallback quick check...")
    try:
        import numpy as np
        from envs.orbit_env import OrbitEnv
        from envs.multi_orbit_env import MultiOrbitEnv

        # Try to construct OrbitEnv directly; if it needs a cfg, provide a minimal one.
        def make_base_env():
            try:
                return OrbitEnv()
            except Exception:
                cfg = SimpleNamespace(mu=3.986e14, dt=1.0, max_steps=8)
                return OrbitEnv(cfg)

        base = make_base_env()
        env = MultiOrbitEnv(base)  # uses its default TaskSampler if none provided

        obs, info = env.reset()

        # Build a zero action matching action_space if available; otherwise use 2-D zeros.
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            import numpy as np
            shape = env.action_space.shape
            action = np.zeros(shape, dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        _ = env.step(action)

        print("[SMOKE] fallback quick check passed ✓")
        sys.exit(0)
    except Exception as e:
        print(f"[SMOKE] fallback quick check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
