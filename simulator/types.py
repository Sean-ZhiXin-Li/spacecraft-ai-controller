from dataclasses import dataclass

@dataclass
class SimConfig:
    """Configuration for the orbital environment."""
    mu: float          # gravitational parameter, m^3/s^2
    dt: float          # timestep, seconds
    max_steps: int     # maximum number of steps per episode

@dataclass
class State:
    """Orbital state variables."""
    x: float
    y: float
    vx: float
    vy: float
    mass: float

@dataclass
class Goal:
    """Target orbit parameters."""
    r_target: float
    tol: float