import numpy as np

def compute_reward(pos, vel, target_radius, thrust, success_counter, G, M, mass):
    """
    Compute the reward signal at each step for orbital control.
    The reward is composed of:
    - A shaping term (continuous feedback for getting closer to the target orbit)
    - A bonus (discrete rewards for entering or maintaining target orbit)
    - Penalties (for inefficient or dangerous behavior)

    Parameters:
        pos (np.ndarray): Current position vector [x, y] in meters
        vel (np.ndarray): Current velocity vector [vx, vy] in m/s
        target_radius (float): Desired orbital radius in meters
        thrust (np.ndarray): Applied thrust vector [Fx, Fy]
        success_counter (int): Number of consecutive steps near target orbit
        G (float): Gravitational constant
        M (float): Mass of central celestial body (e.g. Sun)
        mass (float): Mass of the spacecraft

    Returns:
        total_reward (float): Total reward signal
        shaping (float): Smooth guiding reward (to shape trajectory)
        bonus (float): Discrete reward for orbit success
        penalty (float): Punishment for fuel use or escaping
        radius_error (float): Distance error from target orbit
        speed_error (float): Velocity magnitude error from ideal orbital speed
    """

    # Basic calculations
    curr_radius = np.linalg.norm(pos)            # Distance to center
    v_actual = np.linalg.norm(vel)               # Current speed
    v_circular = np.sqrt(G * M / curr_radius)    # Ideal circular speed

    radius_error = abs(curr_radius - target_radius)
    speed_error = abs(v_actual - v_circular)

    shaping = 0.0
    bonus = 0.0
    penalty = 0.0



    # Shaping reward: Encourage minimizing radius and speed errors
    shaping += -5.0 * (radius_error / target_radius)
    shaping += -0.3 * (speed_error / v_circular)

    # Smooth bonus as the agent approaches the orbit
    shaping += 2.5 * np.exp(-radius_error / 1e11)
    shaping += 0.4 * np.exp(-speed_error / 300)

    shaping += 0.01 * (8000 - success_counter)

    # Bonus reward: Snap rewards for orbit success
    if radius_error < 0.02 * target_radius and speed_error < 200:
        bonus += 30.0

    if success_counter >= 50:
        bonus += 10.0

    # Penalty: For escape or unsafe speed
    escape_speed = np.sqrt(2 * G * M / target_radius)
    if curr_radius > 2.5 * target_radius or v_actual > escape_speed:
        penalty -= 10.0

    # Penalty: Fuel usage
    fuel_penalty = 0.05 * np.linalg.norm(thrust)
    penalty -= fuel_penalty

    # Bonus: Good alignment between velocity and position
    cos_theta = np.dot(pos, vel) / (np.linalg.norm(pos) * np.linalg.norm(vel) + 1e-8)
    shaping += 0.3 * cos_theta

    # Final reward
    total_reward = shaping + bonus + penalty

    return np.tanh(total_reward / 1e4), shaping, bonus, penalty, radius_error, speed_error

