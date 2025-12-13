from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class SimConfig:
    max_steps: int
    dt: float


@dataclass(frozen=True)
class Config:
    target_radius: float
    spacecraft_mass: float
    thrust_limit: float
    simulation: SimConfig


def load_config(path: str | Path = "config/default.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    sim = raw.get("simulation", {})
    return Config(
        target_radius=float(raw["target_radius"]),
        spacecraft_mass=float(raw["spacecraft_mass"]),
        thrust_limit=float(raw["thrust_limit"]),
        simulation=SimConfig(
            max_steps=int(sim.get("max_steps", 2000)),
            dt=float(sim.get("dt", 1.0)),
        ),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
