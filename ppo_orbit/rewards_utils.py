import numpy as np

def compute_reward(
    pos,
    vel,
    thrust,
    target_radius,
    fuel_used,
    G,
    M,
    step_count=None,
    done=False,
    prev_pos=None,
    reward_mode="base",
    w_radius=0.0,
    w_progress=0.0,
    w_speed=0.0,
):
    """
    PPO-friendly shaped reward for orbit control.
    """

    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(G * M / target_radius)

    r_error = abs(r - target_radius) / target_radius
    v_error = abs(v - v_target) / v_target

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = np.dot(unit_r, unit_v)

    # Base shaping
    r_term = -5.0 * np.tanh(r_error * 5.0)
    v_term = -5.0 * np.tanh(v_error * 5.0)
    angle_term = -2.5 * abs(angle_cos) ** 1.2
    shaping = r_term + v_term + angle_term

    # Fuel penalty
    penalty = -0.001 * float(fuel_used)

    # Bonus
    bonus_r = np.exp(-20.0 * r_error ** 2)
    bonus_v = np.exp(-20.0 * v_error ** 2)
    bonus_ang = np.exp(-10.0 * angle_cos ** 2)
    bonus = 3.0 * bonus_r * bonus_v * bonus_ang

    progress = 0.0
    radius_term = 0.0
    progress_term = 0.0
    speed_term = 0.0

    if reward_mode in ["speed", "combined_speed", "full"]:
        speed_term = -w_speed * v_error

    if reward_mode in ["radius", "combined", "full"]:
        radius_term = -w_radius * r_error

    if reward_mode in ["progress", "combined", "full"]:
        if prev_pos is not None:
            prev_r = np.linalg.norm(prev_pos)
            prev_r_error = abs(prev_r - target_radius) / target_radius
            progress = prev_r_error - r_error
            progress = float(np.clip(progress, -0.02, 0.02))
            progress_term = w_progress * progress

    reward = shaping + penalty + bonus + radius_term + progress_term + speed_term

    return {
        "reward": float(reward),
        "shaping": float(shaping),
        "bonus": float(bonus),
        "penalty": float(penalty),
        "r_error": float(r_error),
        "v_error": float(v_error),
        "angle_cos": float(angle_cos),
        "r_term": float(r_term),
        "v_term": float(v_term),
        "angle_term": float(angle_term),
        "progress": float(progress),
        "radius_term": float(radius_term),
        "progress_term": float(progress_term),
        "speed_term": float(speed_term),
    }
