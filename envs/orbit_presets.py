from dataclasses import dataclass
from typing import Optional, Dict
import math

@dataclass
class OrbitParams:
    # Central body & spacecraft
    mu: float                 # GM [m^3/s^2]
    body_radius: float        # central body radius [m] (doc purpose here)
    sc_mass: float            # spacecraft mass [kg]

    # Initial/reference
    r0: float                 # reference orbital radius [m]
    v0: Optional[float] = None  # if None, env can compute circular speed
    ecc: float = 0.0          # documentation for scenario
    inc: float = 0.0          # kept for future 3D; not used in 2D

    # Metadata
    name: str = "circular"
    notes: str = ""

# Solar-like defaults (align with your previous scale)
MU_SUN = 1.32712440018e20       # m^3/s^2
R_SUN  = 6.9634e8               # m
DEFAULT_SC_MASS = 721.9         # kg (Voyager-scale)

def v_circ(mu: float, r: float) -> float:
    """Circular speed at radius r."""
    return math.sqrt(mu / r)

# --- Presets ---
CIRCULAR_FAR = OrbitParams(
    mu=MU_SUN,
    body_radius=R_SUN,
    sc_mass=DEFAULT_SC_MASS,
    r0=7.50e12,
    v0=None,
    ecc=0.0,
    name="circular",
    notes="Near-circular reference orbit",
)

ELLIPTIC_HIGH = OrbitParams(
    mu=MU_SUN,
    body_radius=R_SUN,
    sc_mass=DEFAULT_SC_MASS,
    r0=5.00e12,   # start near periapsis
    v0=None,
    ecc=0.5,
    name="elliptic",
    notes="High-eccentricity ellipse",
)

# Hohmann transfer between r1 and r2
r1 = 5.00e12
r2 = 8.00e12
TRANSFER_HOHMANN = OrbitParams(
    mu=MU_SUN,
    body_radius=R_SUN,
    sc_mass=DEFAULT_SC_MASS,
    r0=r1,
    v0=None,  # wrapper will inject Δv1
    ecc=0.0,
    name="transfer",
    notes=f"Hohmann transfer r1={r1:.2e}, r2={r2:.2e}",
)

PRESET_MAP: Dict[str, OrbitParams] = {
    "circular": CIRCULAR_FAR,
    "elliptic": ELLIPTIC_HIGH,
    "transfer": TRANSFER_HOHMANN,
}
