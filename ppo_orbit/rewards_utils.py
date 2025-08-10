import numpy as np

def compute_reward(pos, vel, thrust, target_radius, fuel_used, G, M, step_count=None, done=False):
    """
    PPO-friendly shaped reward for orbit control.

    Returns:
        reward (float): total reward
        shaping (float): main shaping term (radius, speed, angle)
        bonus (float): smooth near-target bonus
        penalty (float): fuel penalty
        r_error (float): relative radial error
        v_error (float): relative velocity error
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(G * M / target_radius)

    r_error = abs(r - target_radius) / target_radius
    v_error = abs(v - v_target) / v_target

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = np.dot(unit_r, unit_v)  # want near 0 for circular

    # Soft shaping (bounded by tanh)
    r_term    = -5.0 * np.tanh(r_error * 5.0)
    v_term    = -5.0 * np.tanh(v_error * 5.0)
    angle_term= -2.5 * abs(angle_cos) ** 1.2

    shaping = r_term + v_term + angle_term

    # Fuel penalty (small)
    penalty = -0.001 * float(fuel_used)

    # Stronger smooth bonus near the target configuration
    bonus_r   = np.exp(-20.0 * r_error ** 2)
    bonus_v   = np.exp(-20.0 * v_error ** 2)
    bonus_ang = np.exp(-10.0 * angle_cos ** 2)
    bonus = float(12.0 * bonus_r * bonus_v * bonus_ang)  # << increased from 6 -> 10

    reward = shaping + penalty + bonus
    return reward, shaping, bonus, penalty, r_error, v_error
