from dataclasses import dataclass
from typing import Optional

@dataclass
class SimConfig:
    """Global simulation configuration."""
    mu: float = 3.986e14          # Gravitational parameter (m^3/s^2), default ~Earth
    dt: float = 1.0               # Integration time step (s)
    max_steps: int = 1000         # Max episode length
    render_debug: bool = False    # Whether to render debug plots
    r_tol: float = 1e-3           # Radius tolerance for 'near-circular' checks
    v_tol: float = 1e-6           # Velocity tolerance
    seed: Optional[int] = None    # RNG seed (if any)
