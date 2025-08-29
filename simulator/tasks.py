from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Iterable

from simulator.physics import make_circular_state

@dataclass
class TaskSpec:
    """Defines a single task to initialize the environment."""
    kind: str                          # 'circular' | 'custom'
    r0: Optional[float] = None         # For circular
    mass: Optional[float] = None       # Vehicle mass (kg)
    custom_state: Optional[Dict[str, float]] = None  # Full dict for 'custom'

class TaskSampler:
    """
    Minimal task sampler: emits TaskSpec.
    If not provided to env, a default circular sampler is used.
    """
    def __init__(self, mu: float, r_choices: Iterable[float] | None = None, mass: float = 720.0, seed: Optional[int] = None):
        self.mu = mu
        self.mass = mass
        self.r_choices = list(r_choices) if r_choices is not None else [1.0e7, 2.0e7, 3.0e7]
        self.rng = random.Random(seed)

    def sample(self) -> TaskSpec:
        r0 = self.rng.choice(self.r_choices)
        return TaskSpec(kind="circular", r0=r0, mass=self.mass)

    def realize(self, spec: TaskSpec) -> Dict[str, float]:
        if spec.kind == "circular":
            assert spec.r0 is not None and spec.mass is not None
            return make_circular_state(self.mu, spec.r0, spec.mass)
        elif spec.kind == "custom":
            assert spec.custom_state is not None
            return dict(spec.custom_state)
        else:
            raise ValueError(f"Unknown TaskSpec.kind: {spec.kind}")
