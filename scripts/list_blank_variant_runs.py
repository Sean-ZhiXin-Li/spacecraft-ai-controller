from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    runs_dir = Path("analysis/runs")

    print("Runs with blank controller_variant:\n")

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue

        variant = metrics.get("controller_variant", None)

        if variant == "":
            print(f"run: {run_dir.name}")
            print(f"  total_reward: {metrics.get('total_reward')}")
            print(f"  thrust_newton: {metrics.get('thrust_newton')}")
            print(f"  r0_over_target: {metrics.get('r0_over_target')}")
            print(f"  final_radius_error: {metrics.get('final_radius_error')}")
            print(f"  avg_radius_error: {metrics.get('avg_radius_error')}")
            print()


if __name__ == "__main__":
    main()