import numpy as np

def compute_reward(pos, vel, thrust, target_radius, fuel_used, G, M , step_count=None, done=False):
    """
    Computes a shaped reward for orbit stabilization and fuel-efficient control.

    Args:
        pos (np.ndarray): Current position vector (2,)
        vel (np.ndarray): Current velocity vector (2,)
        thrust (np.ndarray): Applied thrust vector (2,)
        target_radius (float): Desired orbital radius
        fuel_used (float): Scalar amount of fuel consumed
        G (float): Gravitational constant
        M (float): Central body mass

    Returns:
        reward (float): Total reward signal
        shaping (float): Error-based shaping reward
        bonus (float): Extra reward if orbit is near perfect
        penalty (float): Fuel usage penalty
        r_error (float): Relative radius error
        v_error (float): Relative velocity error
    """

    # Magnitudes of position and velocity
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    # Ideal circular orbit speed at target radius
    v_target = np.sqrt(G * M / target_radius)

    # Relative errors
    r_error = abs(r - target_radius) / target_radius
    v_error = abs(v - v_target) / v_target

    # Angle between position and velocity: ideal circular orbit has orthogonal vectors
    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = np.dot(unit_r, unit_v)  # should be ~0 for circular

    # Reward shaping: penalize deviation from desired orbit
    shaping = (
        - 10.0 * r_error           # radial deviation penalty
        - 10.0 * v_error           # speed deviation penalty
        - 5.0 * abs(angle_cos)     # angular misalignment penalty
    )

    # Soft penalty on fuel usage
    penalty = -0.001 * fuel_used

    # Bonus for near-perfect orbit stabilization
    bonus = 0.0
    if r_error < 0.01 and v_error < 0.01 and abs(angle_cos) < 0.1:
        bonus = 5.0

    # Total reward
    reward = shaping + penalty + bonus

    return reward, shaping, bonus, penalty, r_error, v_error
