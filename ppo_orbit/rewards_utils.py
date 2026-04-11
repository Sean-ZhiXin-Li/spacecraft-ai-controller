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
    dt=2.0,
):
    """
    Reward design goals:
    1. Reach target orbital radius
    2. Match circular-orbit speed
    3. Suppress radial velocity
    4. Encourage tangential motion
    5. Penalize excessive thrust
    6. Penalize abrupt action changes
    7. Reward sustained stable orbit behavior
    """

    eps = 1e-12
    mu = G * M

    # ---------- Basic kinematics ----------
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    v_target = np.sqrt(mu / (target_radius + eps))

    unit_r = pos / (r + 1e-8)
    unit_v = vel / (v + 1e-8)
    unit_t = np.array([-unit_r[1], unit_r[0]], dtype=np.float64)

    angle_cos = float(np.dot(unit_r, unit_v))

    # Signed / projected velocities
    v_r_true = float(np.dot(vel, unit_r))
    v_t = float(np.dot(vel, unit_t))
    tangential_signed = float(np.dot(unit_v, unit_t))
    radial_signed = float(np.dot(unit_v, unit_r))

    tangential_alignment = max(0.0, tangential_signed)
    radial_alignment = abs(radial_signed)

    # ---------- Normalized errors ----------
    r_error = abs(r - target_radius) / (target_radius + eps)
    v_error = abs(v - v_target) / (v_target + eps)
    vr_norm = abs(v_r_true) / (v_target + eps)
    vt_error = abs(v_t - v_target) / (v_target + eps)
    v_t_ratio = v_t / (v_target + eps)
    speed_ratio = v / (v_target + eps)

    # ---------- Progress ----------
    progress = 0.0
    radial_growth_penalty = 0.0
    phase_eff = 0.0
    if prev_pos is not None:
        prev_r = np.linalg.norm(prev_pos)
        prev_r_error = abs(prev_r - target_radius) / (target_radius + eps)

        progress = float(np.clip(prev_r_error - r_error, -0.05, 0.05))

        dr = float(r - prev_r)
        if dr > 0.0 and r > target_radius:
            radial_growth_penalty = 4.0 * dr / (v_target * max(dt, 1e-9) + eps)

        theta_prev = float(np.arctan2(prev_pos[1], prev_pos[0]))
        theta_now = float(np.arctan2(pos[1], pos[0]))
        dtheta = float((theta_now - theta_prev + np.pi) % (2.0 * np.pi) - np.pi)
        omega_target = v_target / (target_radius + eps)
        phase_eff = abs(dtheta) / (abs(omega_target) * max(dt, 1e-9) + eps)

    # ---------- Action statistics ----------
    thrust_mag = np.linalg.norm(thrust)
    thrust_norm = float(thrust_mag / (np.sqrt(2.0) * thrust_scale + eps))

    if not hasattr(compute_reward, "prev_thrust"):
        compute_reward.prev_thrust = np.zeros_like(thrust, dtype=np.float64)

    if not hasattr(compute_reward, "stable_steps"):
        compute_reward.stable_steps = 0

    delta_u = thrust - compute_reward.prev_thrust
    delta_norm = float(np.linalg.norm(delta_u) / (thrust_scale + eps))

    # ---------- Orbit geometry ----------
    h = float(pos[0] * vel[1] - pos[1] * vel[0])
    h_target = float(target_radius * v_target)
    h_norm = h / (h_target + 1e-8)
    h_error = abs(h_norm - 1.0)

    specific_energy = 0.5 * (v ** 2) - mu / (r + eps)
    target_energy = -mu / (2.0 * target_radius + eps)
    energy_rel_error = float(
        abs(specific_energy - target_energy) / (abs(target_energy) + eps)
    )


    ecc_proxy = 0.0
    if mu > 0.0:
        ecc_sq = 1.0 + (2.0 * specific_energy * (h ** 2)) / (mu ** 2 + eps)
        ecc_proxy = float(np.sqrt(max(0.0, ecc_sq)))

    # ---------- Smooth gating: approach -> stabilize ----------
    # w = 1 means more approach-like; w = 0 means more stabilize-like
    w = float(np.clip((r_error - 0.02) / 0.08, 0.0, 1.0))

    # ==========================================================
    # Approach reward
    # ==========================================================
    reward_approach = 0.0
    reward_approach += 2.0 * progress
    reward_approach += 1.5 * (1.0 - np.tanh(5.0 * r_error))
    reward_approach += 0.8 * tangential_alignment
    reward_approach -= 1.2 * vr_norm
    reward_approach -= 0.6 * radial_alignment
    reward_approach -= 0.15 * thrust_norm

    # ==========================================================
    # Stabilize reward
    # ==========================================================
    reward_stable = 0.0
    reward_stable += 2.0 * (1.0 - np.tanh(6.0 * r_error))
    reward_stable += 1.8 * (1.0 - np.tanh(5.0 * v_error))
    reward_stable += 2.5 * (1.0 - np.tanh(6.0 * vr_norm))
    reward_stable += 1.2 * tangential_alignment
    reward_stable -= 1.4 * radial_alignment
    reward_stable -= 1.0 * vt_error
    reward_stable -= 0.8 * h_error
    reward_stable -= 0.8 * energy_rel_error
    reward_stable -= 0.4 * min(ecc_proxy, 2.0)
    reward_stable -= 0.08 * thrust_norm

    # Blend
    reward = w * reward_approach + (1.0 - w) * reward_stable

    # ---------- Global shaping ----------
    # Discourage purely radial motion
    reward -= 0.8 * abs(angle_cos)

    # Penalize outward drift when already outside target orbit
    outward_escape_penalty = 0.0
    if r > target_radius and v_r_true > 0.0:
        outward_escape_penalty = 2.0 * vr_norm * (1.0 + 3.0 * r_error)
        reward -= outward_escape_penalty

    # Penalize large action changes (critical for staircase trajectory issue)
    action_smooth_penalty = 1.2 * (delta_norm ** 2)
    reward -= action_smooth_penalty

    # Slightly penalize too much radial growth
    reward -= radial_growth_penalty

    # Near-target enforcement
    near_target_radial_penalty = 0.0
    tangential_speed_penalty = 0.0
    if r_error < 0.12:
        near_target_radial_penalty = 1.8 * vr_norm
        tangential_speed_penalty = 1.2 * abs(v_t_ratio - 1.0)
        reward -= near_target_radial_penalty
        reward -= tangential_speed_penalty
        reward += 1.5 * tangential_alignment
        reward -= 1.5 * radial_alignment

    # Overspeed penalty
    overspeed_penalty = 0.0
    if speed_ratio > 1.05:
        overspeed_penalty = 2.5 * ((speed_ratio - 1.05) ** 2)
        reward -= overspeed_penalty

    # ---------- Stability hold bonus ----------
    stable_r = r_error < 0.01
    stable_v = v_error < 0.02
    stable_vr = vr_norm < 0.02
    stable_vt = abs(v_t_ratio - 1.0) < 0.03

    is_stable = stable_r and stable_v and stable_vr and stable_vt

    if is_stable:
        compute_reward.stable_steps += 1
    else:
        compute_reward.stable_steps = 0

    stop_bonus = 0.0
    hold_bonus = 0.0

    if is_stable:
        stop_bonus = 2.0
        hold_bonus = min(compute_reward.stable_steps / 50.0, 1.0) * 3.0
        reward += stop_bonus + hold_bonus

    # ---------- Optional reward modes ----------
    if reward_mode == "base":
        pass


    elif reward_mode == "simple_orbit":

        reward = 0.0

        reward += 20.0 * (1.0 - np.tanh(6.0 * r_error))

        reward += 15.0 * (1.0 - np.tanh(5.0 * v_error))

        reward += 20.0 * (1.0 - np.tanh(6.0 * vr_norm))

        reward += 10.0 * tangential_alignment

        reward -= 10.0 * radial_alignment

        reward -= 1.5 * thrust_norm

        reward -= 3.0 * (delta_norm ** 2)

        reward += 10.0 * stop_bonus + 10.0 * hold_bonus

    elif reward_mode == "orbit_strict":
        reward += 1.5 * tangential_alignment
        reward -= 1.5 * radial_alignment
        reward -= 1.2 * abs(v_t_ratio - 1.0)
        reward -= 1.2 * h_error
        reward -= 1.0 * energy_rel_error
        reward -= 1.5 * (delta_norm ** 2)
        reward += 1.2 * stop_bonus + 1.2 * hold_bonus

        if r_error < 0.08 and tangential_alignment < 0.55:
            reward -= 2.0
        if r_error < 0.08 and vr_norm > 0.05:
            reward -= 2.0 * vr_norm




    elif reward_mode == "orbit_smooth_v2":

        reward = 0.0

        outward_escape_penalty = 0.0

        action_smooth_penalty = 0.0

        near_target_radial_penalty = 0.0

        tangential_speed_penalty = 0.0

        radial_growth_penalty = 0.0

        # ===== Core objectives =====

        reward += 3.0 * (1.0 - np.tanh(8.0 * r_error))

        reward += 1.5 * (1.0 - np.tanh(5.0 * v_error))

        # ===== Strong orbital shaping =====

        reward += 4.0 * (1.0 - np.tanh(8.0 * vr_norm))

        reward += 5.0 * tangential_alignment

        reward -= 4.0 * radial_alignment

        reward += 2.5 * np.exp(-3.0 * abs(v_t_ratio - 1.0))

        # ===== Physics =====

        reward -= 1.0 * h_error

        reward -= 1.5 * energy_rel_error

        if hasattr(compute_reward, "prev_energy"):
            dE = abs(specific_energy - compute_reward.prev_energy)

            reward -= 10.0 * dE / (abs(target_energy) + eps)

        compute_reward.prev_energy = specific_energy

        # ===== Control =====

        reward -= 0.03 * thrust_norm

        reward -= 6.0 * (delta_norm ** 2)

        # ===== HARD constraint =====

        reward -= 2.5 * abs(angle_cos)

        reward -= 2.5 * (radial_alignment ** 2)

        # ===== Progress =====

        reward += 0.8 * progress

        # ===== Hold =====

        reward += 1.2 * stop_bonus + 1.8 * hold_bonus

        # ===== Near target =====

        if r_error < 0.08:
            reward += 4.0 * tangential_alignment

            reward -= 5.0 * vr_norm

            reward -= 4.0 * abs(v_t_ratio - 1.0)

        # Speed explosion penalties

        if speed_ratio > 1.05:
            reward -= 10.0 * (speed_ratio - 1.05)

        orbit_lock = np.exp(-10.0 * r_error) * np.exp(-8.0 * v_error)

        reward += 8.0 * orbit_lock

        compute_reward.prev_energy = specific_energy

        # Prevent speed explosion

        reward -= 3.0 * abs(v - v_target) / (v_target + 1e-6)

        if v > 1.2 * v_target:
            reward -= 2.0 * thrust_norm

        # ===== ANGULAR MOMENTUM (CRITICAL) =====

        reward += 4.0 * np.exp(-4.0 * h_error)

        # ===== HARD NEAR-ORBIT ENFORCEMENT =====

        # When the spacecraft is already close to the target orbit,

        # force tangential motion and suppress radial behavior aggressively.

        if r_error < 0.12:
            reward += 8.0 * tangential_alignment

            reward -= 10.0 * radial_alignment

            reward -= 12.0 * vr_norm

            reward -= 8.0 * abs(v_t_ratio - 1.0)

        # Kill fake-stable solutions: close radius but still not rotating properly.

        if r_error < 0.10 and tangential_alignment < 0.55:
            reward -= 8.0

        # Extra reward for true near-circular orbital behavior.

        if r_error < 0.08 and vr_norm < 0.04 and abs(v_t_ratio - 1.0) < 0.05:
            reward += 10.0

        # Penalize outward drift harder once outside the target orbit.

        if r > target_radius and v_r_true > 0.0:
            reward -= 6.0 * vr_norm * (1.0 + 2.0 * r_error)

        # Stronger overspeed suppression.

        if speed_ratio > 1.08:
            reward -= 12.0 * (speed_ratio - 1.08)

    elif reward_mode == "orbit_circular_minimal":

        reward = 0.0

        reward -= 5.0 * r_error
        reward -= 2.0 * vt_error
        reward -= 2.0 * vr_norm
        reward += 2.0 * (1.0 - abs(v_r) / (abs(v_t) + 1e-6))

    if reward_mode != "orbit_circular_minimal":
        reward = 20.0 * np.tanh(reward / 20.0)

    # ---------- Save internal state ----------
    compute_reward.prev_thrust = thrust.copy()

    # ---------- Debug fields ----------
    shaping = 0.0
    bonus = stop_bonus + hold_bonus
    penalty = (
        action_smooth_penalty
        + overspeed_penalty
        + outward_escape_penalty
        + near_target_radial_penalty
        + tangential_speed_penalty
        + radial_growth_penalty
    )

    r_term = float(1.0 - np.tanh(6.0 * r_error))
    v_term = float(1.0 - np.tanh(5.0 * v_error))
    angle_term = float(-abs(angle_cos))
    radius_term = float(r_term)
    progress_term = float(progress)
    speed_term = float(1.0 - np.tanh(6.0 * vr_norm))
    damping_term = float(-(delta_norm ** 2))

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
        "hold_bonus": float(hold_bonus),
        "stable_steps": int(compute_reward.stable_steps),
        "overspeed_penalty": float(overspeed_penalty),
        "thrust_norm": float(thrust_norm),
        "delta_norm": float(delta_norm),
        "v_r": float(v_r_true),
        "v_t": float(v_t),
        "h_norm": float(h_norm),
        "angular_momentum": float(h),
        "radial_trap_penalty": float(near_target_radial_penalty),
        "lock_gate": float(np.exp(-((r_error / 0.06) ** 2 + (v_error / 0.08) ** 2))),
        "v_t_ratio": float(v_t_ratio),
        "thrust_tangential": float(0.0 if thrust_mag < 1e-9 else np.dot(thrust / (thrust_mag + eps), unit_t)),
        "sustained_tan_force_bonus": float(0.0),
        "tangential_speed_penalty": float(tangential_speed_penalty),
        "outward_escape_penalty": float(outward_escape_penalty),
        "near_target_radial_penalty": float(near_target_radial_penalty),
        "phase_eff": float(phase_eff),
        "closure_bonus": float(0.0),
        "closure_penalty": float(0.0),
        "radial_growth_penalty": float(radial_growth_penalty),
        "energy_rel_error": float(energy_rel_error),
        "specific_energy": float(specific_energy),
        "target_energy": float(target_energy),
        "ecc_proxy": float(ecc_proxy),
    }

