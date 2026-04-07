from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    runs_dir = Path("analysis/runs")
    counter = Counter()

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue

        variant = metrics.get("controller_variant", "<missing>")
        counter[str(variant)] += 1

    print("Available controller_variant values:\n")
    for name, count in counter.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()