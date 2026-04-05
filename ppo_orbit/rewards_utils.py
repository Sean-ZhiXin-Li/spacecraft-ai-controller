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
    thrust_scale=3000.0,
):
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(G * M / target_radius)

    r_error = abs(r - target_radius) / (target_radius + 1e-12)
    v_error = abs(v - v_target) / (v_target + 1e-12)

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    angle_cos = float(np.dot(unit_r, unit_v))

    unit_t = np.array([-unit_r[1], unit_r[0]], dtype=np.float64)

    tangential_signed = float(np.dot(unit_v, unit_t))
    radial_signed = float(np.dot(unit_v, unit_r))

    tangential_alignment = max(0.0, tangential_signed)
    radial_alignment = abs(radial_signed)

    progress = 0.0
    if prev_pos is not None:
        prev_r = np.linalg.norm(prev_pos)
        prev_r_error = abs(prev_r - target_radius) / (target_radius + 1e-12)
        progress = prev_r_error - r_error
        progress = float(np.clip(progress, -0.05, 0.05))

    vr_norm = float(v_r / (v_target + 1e-12))
    thrust_norm = float(
        np.linalg.norm(thrust) / (np.sqrt(2.0) * thrust_scale + 1e-12)
    )
    speed_error = float((v - v_target) / (v_target + 1e-12))

    # soft transition: 1 = approach, 0 = stabilize
    w = float(np.clip((r_error - 0.02) / 0.08, 0.0, 1.0))

    # ---------- approach reward ----------
    reward_approach = 0.0
    reward_approach -= 6.0 * r_error
    reward_approach -= 8.0 * (vr_norm ** 2)
    reward_approach += 4.0 * progress
    reward_approach += 2.0 * tangential_alignment
    reward_approach -= 0.2 * thrust_norm

    # ---------- stabilize reward ----------
    h = abs(pos[0] * vel[1] - pos[1] * vel[0]) / (target_radius * v_target + 1e-12)

    reward_stable = 0.0
    reward_stable -= 14.0 * r_error
    reward_stable -= 18.0 * (vr_norm ** 2)
    reward_stable -= 10.0 * (speed_error ** 2)
    reward_stable += 18.0 * tangential_alignment
    reward_stable -= 4.0 * radial_alignment

    # Do not punish thrust too hard, otherwise policy collapses to near-zero action
    reward_stable -= 0.08 * thrust_norm
    reward_stable += 1.2 * np.tanh(3.0 * thrust_norm)

    # Encourage real orbital motion instead of flat drift
    reward_stable += 4.0 * h

    # blend instead of hard switch
    reward = w * reward_approach + (1.0 - w) * reward_stable

    # local stability bonus
    stop_bonus = 0.0
    if r_error < 0.03 and abs(v_r) < 0.03 * v_target and v_error < 0.03:
        stop_bonus = 30.0
        reward += stop_bonus

    overspeed_penalty = 0.0
    if v > 1.3 * v_target:
        overspeed_penalty = 10.0 * (v / (v_target + 1e-12) - 1.3)
        reward -= overspeed_penalty

    h = pos[0] * vel[1] - pos[1] * vel[0]
    h_target = target_radius * np.sqrt(G * M / target_radius)

    h_norm = h / (h_target + 1e-8)

    h_term = 2.0 * h_norm
    reward += h_term

    if v > 1.2 * v_target:
        reward -= 2.0 * (v / v_target - 1.2)

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

    stage = "approach" if w > 0.5 else "stabilize"

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
        "angular_momentum": float(pos[0] * vel[1] - pos[1] * vel[0]),
    }
