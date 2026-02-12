# scripts/run_grid_min.py
# Minimal 2x2 grid runner for WHPL_07 (3 new points).
# This script does NOT modify physics/reward/controller.
# It only orchestrates existing run pipeline via subprocess.

import os
import sys
import subprocess
from datetime import datetime

# EDIT THIS ONLY
# Put the exact command you used to generate the existing (800, Hard) row.
# Example patterns:
#   ["python", "src/quick_compare_v3_v4.py"]
#   ["python", "scripts/quickrun.py"]
#   ["python", "scripts/run_baseline_complex.py"]
RUN_CMD = ["python", "src/quick_compare_v3_v4.py"]  # <-- change to your real entrypoint

def run_one(thrust: float, difficulty: str) -> None:
    env = os.environ.copy()
    # Common env knobs you already used in WHPL_06
    env["THRUST_NEWTON"] = str(thrust)
    env["DIFFICULTY_TAG"] = difficulty  # if your code reads this
    env["DIFFICULTY"] = difficulty      # fallback key, harmless if ignored

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== RUN point thrust={thrust} difficulty={difficulty} @ {stamp} ===")
    print(f"[CMD] {' '.join(RUN_CMD)}")

    r = subprocess.run(RUN_CMD, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] run failed for ({thrust}, {difficulty}), code={r.returncode}")

def main() -> None:
    # 3 new points for WHPL_07
    points = [
        (2000.0, "Hard"),
        (800.0,  "Easy"),
        (2000.0, "Easy"),
    ]

    for thrust, diff in points:
        run_one(thrust, diff)

    print("\n[OK] all WHPL_07 points completed")

if __name__ == "__main__":
    main()
