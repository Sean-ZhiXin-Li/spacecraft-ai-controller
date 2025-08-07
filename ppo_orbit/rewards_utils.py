import numpy as np


def compute_reward(pos, vel, thrust, target_radius, fuel_used, G, M, step_count=None, done=False):
    """
    PPO-optimized reward function with smooth shaping and structured components.

    Returns:
        reward (float): Total reward.
        shaping (float): Orbital deviation shaping.
        bonus (float): Near-target smooth bonus.
        penalty (float): Fuel usage penalty.
        r_error (float): Relative radial error.
        v_error (float): Relative velocity error.
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(G * M / target_radius)

    r_error = abs(r - target_radius) / target_radius
    v_error = abs(v - v_target) / v_target

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = np.dot(unit_r, unit_v)

    r_term = -5.0 * np.tanh(r_error * 5)
    v_term = -5.0 * np.tanh(v_error * 5)
    angle_term = -3.0 * abs(angle_cos) ** 1.5

    shaping = r_term + v_term + angle_term
    penalty = -0.001 * fuel_used

    bonus_r = np.exp(-20 * r_error ** 2)
    bonus_v = np.exp(-20 * v_error ** 2)
    bonus_ang = np.exp(-10 * angle_cos ** 2)
    bonus = 3.0 * bonus_r * bonus_v * bonus_ang

    reward = shaping + penalty + bonus
    return reward, shaping, bonus, penalty, r_error, v_error
