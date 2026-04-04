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
    v_r=0.0,
):
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(G * M / target_radius)

    r_error = abs(r - target_radius) / target_radius
    v_error = abs(v - v_target) / v_target

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = np.dot(unit_r, unit_v)

    unit_t = np.array([-unit_r[1], unit_r[0]], dtype=np.float64)

    # Signed alignments
    tangential_signed = float(np.dot(unit_v, unit_t))
    radial_signed = float(np.dot(unit_v, unit_r))

    # Magnitudes / gated tangential reward
    radial_alignment = abs(radial_signed)

    # Only reward positive tangential motion
    tangential_alignment = max(0.0, tangential_signed)

    # Sharpen reward: weak tangential motion gets very little credit
    tangential_alignment = tangential_alignment ** 2

    angular_momentum = pos[0] * vel[1] - pos[1] * vel[0]
    h_norm = angular_momentum / (r * v_target + 1e-8)

    progress = 0.0
    if prev_pos is not None:
        prev_r = np.linalg.norm(prev_pos)
        prev_r_error = abs(prev_r - target_radius) / target_radius
        progress = prev_r_error - r_error
        progress = np.clip(progress, -0.05, 0.05)

    vr_norm = abs(v_r) / (v_target + 1e-12)
    thrust_norm = np.linalg.norm(thrust) / (np.sqrt(2.0) * 3000.0 + 1e-12)
    speed_ratio = v / (v_target + 1e-12)

    if r_error > 0.2:
        stage = "approach"
        reward = 5.0 * progress - 1.0 * vr_norm
    else:
        stage = "stabilize"
        near_target_gate = np.exp(-20.0 * r_error * r_error)

        tangential_reward = 40.0 * tangential_alignment
        radial_penalty = 6.0 * radial_alignment

        if np.linalg.norm(thrust) > 1e-6:
            thrust_dir = thrust / (np.linalg.norm(thrust) + 1e-8)
            thrust_tangential = np.dot(thrust_dir, unit_t)
        else:
            thrust_tangential = 0.0

        reward = (
                - 40.0 * (v_r / v_target) ** 2
                + tangential_reward
                - radial_penalty
                - 0.05 * thrust_norm
                - 8.0 * max(0.0, speed_ratio - 1.0)
                - 8.0 * r_error
                + 80.0 * thrust_tangential
        )

        if tangential_alignment < 0.2:
            reward -= 10.0

    stop_bonus = 0.0
    vr_stop_threshold = 0.01 * v_target
    if r_error < 0.05 and abs(v_r) < vr_stop_threshold:
        stop_bonus = 50.0
        reward += stop_bonus

    overspeed_penalty = 0.0
    if v > 1.5 * v_target:
        overspeed_penalty = 20.0 * (v / (v_target + 1e-12) - 1.5)
        reward -= overspeed_penalty

    shaping = 0.0
    bonus = 0.0
    penalty = 0.0
    r_term = 0.0
    v_term = 0.0
    angle_term = 0.0
    radius_term = 0.0
    progress_term = 0.0
    speed_term = 0.0
    damping_term = 0.0

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
        "damping_term": float(damping_term),
        "stage": stage,
        "tangential_alignment": float(tangential_alignment),
        "radial_alignment": float(radial_alignment),
        "stop_bonus": float(stop_bonus),
        "overspeed_penalty": float(overspeed_penalty),
        "thrust_norm": float(thrust_norm),
        "v_r": float(v_r),
        "h_norm": float(h_norm),
        "angular_momentum": float(angular_momentum),
    }
