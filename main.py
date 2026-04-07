from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run best-trajectory demo exporter.")
    parser.add_argument("--controller-variant", type=str, default="gated")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    script_path = project_root / "scripts" / "find_best_trajectory_demo.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--runs-dir",
        str(project_root / "analysis" / "runs"),
        "--ppo-dir",
        str(project_root / "ppo_orbit"),
        "--out-dir",
        str(project_root / "analysis" / "demo_best"),
    ]

    if args.controller_variant is not None:
        cmd.extend(["--controller-variant", args.controller_variant])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
