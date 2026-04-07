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
    v_t = float(np.dot(vel, unit_t))
    v_t_ratio = v_t / (v_target + 1e-12)

    tangential_alignment = max(0.0, tangential_signed)
    radial_alignment = abs(radial_signed)

    progress = 0.0
    phase_eff = 0.0
    closure_bonus = 0.0
    closure_penalty = 0.0
    radial_growth_penalty = 0.0
    if prev_pos is not None:
        prev_r = np.linalg.norm(prev_pos)
        prev_r_error = abs(prev_r - target_radius) / (target_radius + 1e-12)
        progress = prev_r_error - r_error
        progress = float(np.clip(progress, -0.05, 0.05))
        dr = float(r - prev_r)
        if dr > 0.0:
            radial_growth_penalty = 20.0 * dr / (v_target * max(1e-9, dt) + 1e-12)
        theta_prev = float(np.arctan2(prev_pos[1], prev_pos[0]))
        theta_now = float(np.arctan2(pos[1], pos[0]))
        dtheta = float((theta_now - theta_prev + np.pi) % (2.0 * np.pi) - np.pi)
        omega_target = v_target / (target_radius + 1e-12)
        phase_eff = abs(dtheta) / (abs(omega_target) * max(1e-9, dt) + 1e-12)

    vr_norm = float(v_r / (v_target + 1e-12))
    thrust_norm = float(
        np.linalg.norm(thrust) / (np.sqrt(2.0) * thrust_scale + 1e-12)
    )
    speed_error = float((v - v_target) / (v_target + 1e-12))
    mu = G * M
    specific_energy = 0.5 * (v ** 2) - mu / (r + 1e-12)
    target_energy = -mu / (2.0 * target_radius + 1e-12)
    energy_rel_error = (specific_energy - target_energy) / (abs(target_energy) + 1e-12)

    # soft transition: 1 = approach, 0 = stabilize
    w = float(np.clip((r_error - 0.02) / 0.08, 0.0, 1.0))

    # ---------- approach reward ----------
    reward_approach = 0.0
    reward_approach -= 6.0 * r_error
    reward_approach -= 8.0 * (vr_norm ** 2)
    reward_approach += 4.0 * progress
    reward_approach += 3.0 * tangential_alignment
    reward_approach -= 2.0 * radial_alignment
    reward_approach -= 0.2 * thrust_norm

    # ---------- stabilize reward ----------
    h = abs(pos[0] * vel[1] - pos[1] * vel[0]) / (target_radius * v_target + 1e-12)

    reward_stable = 0.0
    reward_stable -= 14.0 * r_error
    reward_stable -= 22.0 * (vr_norm ** 2)
    reward_stable -= 10.0 * (speed_error ** 2)
    reward_stable -= 14.0 * ((v_t_ratio - 1.0) ** 2)
    reward_stable += 26.0 * tangential_alignment
    reward_stable -= 16.0 * radial_alignment

    # Do not punish thrust too hard, otherwise policy collapses to near-zero action
    reward_stable -= 0.08 * thrust_norm
    reward_stable += 1.2 * np.tanh(3.0 * thrust_norm)

    # Encourage real orbital motion instead of flat drift
    reward_stable += 4.0 * h
    reward_stable -= 6.0 * (energy_rel_error ** 2)

    # blend instead of hard switch
    reward = w * reward_approach + (1.0 - w) * reward_stable
    reward -= 5.0 * abs(angle_cos)
    reward -= radial_growth_penalty

    # Enforce "must turn" near target: reward tangential motion and punish radial alignment.
    lock_gate = float(np.exp(-((r_error / 0.06) ** 2 + (v_error / 0.08) ** 2)))
    orbit_lock = lock_gate * (2.5 * tangential_alignment - 2.5 * radial_alignment)
    reward += orbit_lock

    thrust_tangential = 0.0
    if np.linalg.norm(thrust) > 1e-9:
        thrust_u = thrust / (np.linalg.norm(thrust) + 1e-12)
        thrust_tangential = float(np.dot(thrust_u, unit_t))
    sustained_tan_force_bonus = lock_gate * max(0.0, thrust_tangential)
    reward += 2.0 * sustained_tan_force_bonus
    closure_bonus = lock_gate * float(np.clip(phase_eff, 0.0, 2.0))
    reward += 3.0 * closure_bonus

    # local stability bonus
    stop_bonus = 0.0
    if r_error < 0.03 and abs(v_r) < 0.03 * v_target and v_error < 0.03:
        stop_bonus = 30.0
        reward += stop_bonus

    overspeed_penalty = 0.0
    speed_ratio = v / (v_target + 1e-12)
    if speed_ratio > 1.05:
        overspeed_penalty = 20.0 * ((speed_ratio - 1.05) ** 2)
        reward -= overspeed_penalty

    h = pos[0] * vel[1] - pos[1] * vel[0]
    h_target = target_radius * np.sqrt(G * M / target_radius)

    h_norm = h / (h_target + 1e-8)
    h_close = 1.0 - abs(h_norm - 1.0)
    h_term = 2.0 * float(np.clip(h_close, -1.0, 1.0))
    reward += h_term

    ecc_proxy = 0.0
    if mu > 0.0:
        ecc_sq = 1.0 + (2.0 * specific_energy * (h ** 2)) / (mu ** 2 + 1e-12)
        ecc_proxy = float(np.sqrt(max(0.0, ecc_sq)))
    reward -= 3.0 * min(ecc_proxy, 2.0)

    if speed_ratio > 1.12:
        reward -= 8.0 * (speed_ratio - 1.12)

        # Extra hard penalty when close to target radius but still flying radially.
    radial_trap_penalty = 0.0
    if r_error < 0.10:
        radial_trap_penalty = 4.0 * radial_alignment
        reward -= radial_trap_penalty

    tangential_speed_penalty = 0.0
    if r_error < 0.12 and v_t_ratio < 0.92:
        tangential_speed_penalty = 4.0 * (0.92 - v_t_ratio)
        reward -= tangential_speed_penalty

    outward_escape_penalty = 0.0
    if r > target_radius and vr_norm > 0.0:
        outward_escape_penalty = 6.0 * vr_norm * (1.0 + 3.0 * r_error)
        reward -= outward_escape_penalty

    near_target_radial_penalty = 0.0
    if r_error < 0.15:
        near_target_radial_penalty = 8.0 * abs(vr_norm)
        reward -= near_target_radial_penalty
        if phase_eff < 0.25:
            closure_penalty = 4.0 * (0.25 - phase_eff)
            reward -= closure_penalty

    if reward_mode == "simple_orbit":
        reward = 0.0

        # 1. radius tracking
        reward -= 20.0 * r_error

        # 2. suppress radial velocity
        reward -= 15.0 * abs(vr_norm)

        # 3. tangential speed matching
        reward -= 10.0 * (v_t_ratio - 1.0) ** 2

        speed_err = (v - v_target) / (v_target + 1e-12)
        reward -= 10.0 * (speed_err ** 2)
        reward -= 2.0 * abs(v - v_target) / (v_target + 1e-12)

        # 4. direction shaping
        reward += 10.0 * tangential_alignment
        reward -= 10.0 * radial_alignment

        # 5. control effort + smoothness
        reward -= 2.0 * thrust_norm
        reward -= 3.0 * (thrust_norm ** 2)

        # 6. centripetal consistency
        v_r = float(np.dot(vel, pos) / (r + 1e-8))
        reward -= 3.0 * abs(v_r) / (v_target + 1e-8)

        # 7. angular momentum stabilization
        h = pos[0] * vel[1] - pos[1] * vel[0]
        h_target = r * v_target
        h_err = abs(h - h_target) / (abs(h_target) + 1e-8)
        if r_error < 0.2:
            reward -= 6.0 * (h_err ** 2)

        # 8. near-orbit hard guidance
        if r_error < 0.2:
            reward += 6.0 * tangential_alignment
            reward -= 6.0 * radial_alignment

        # 9. angular velocity consistency
        omega = abs(np.cross(pos, vel)) / (r ** 2 + 1e-8)
        omega_target = v_target / (target_radius + 1e-8)
        omega_err = abs(omega - omega_target) / (omega_target + 1e-8)
        if r_error < 0.2:
            reward -= 4.0 * (omega_err ** 2)

        # --- Action smoothness (CRITICAL) ---
        if hasattr(compute_reward, "prev_thrust"):
            delta_u = thrust - compute_reward.prev_thrust
            delta_norm = np.linalg.norm(delta_u) / (thrust_scale + 1e-8)
            reward -= 4.0 * (delta_norm ** 2)

        compute_reward.prev_thrust = thrust.copy()

    elif reward_mode == "orbit_strict":
        reward += 2.0 * orbit_lock
        reward -= 1.5 * radial_trap_penalty
        if tangential_alignment < 0.35 and r_error < 0.20:
            reward -= 1.0
        reward += 2.0 * sustained_tan_force_bonus
        reward -= 1.5 * tangential_speed_penalty
        reward -= 1.2 * outward_escape_penalty
        reward -= 1.2 * near_target_radial_penalty
        reward += 1.5 * closure_bonus
        reward -= 1.5 * closure_penalty
        reward -= 2.0 * (energy_rel_error ** 2)
        if specific_energy > target_energy:
            reward -= 4.0 * ((specific_energy - target_energy) / (abs(target_energy) + 1e-12))

        if angle_cos > 0.8:
            reward -= 4.0 * (angle_cos - 0.8)
        if r_error < 0.15 and radial_alignment > 0.6:
            reward -= 3.0 * (radial_alignment - 0.6)
        if r > target_radius:
            reward -= 10.0 * (r_error ** 1.5)
        if r_error < 0.2:
            reward += 6.0 * tangential_alignment
        else:
            reward += 2.0 * tangential_alignment
        if r_error < 0.08 and abs(v_t_ratio - 1.0) < 0.08 and abs(vr_norm) < 0.05:
            reward += 6.0
        if r_error < 0.08:
            reward += 4.0 * tangential_alignment
            reward -= 4.0 * radial_alignment
        if r_error < 0.1 and abs(vr_norm) > 0.1:
            reward -= 5.0 * abs(vr_norm)

    # Soft squash keeps gradient signal while preventing huge negative plateaus.
    reward = 10.0 * np.tanh(reward / 10.0)

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
        "orbit_lock": float(orbit_lock),
        "radial_trap_penalty": float(radial_trap_penalty),
        "lock_gate": float(lock_gate),
        "v_t_ratio": float(v_t_ratio),
        "thrust_tangential": float(thrust_tangential),
        "sustained_tan_force_bonus": float(sustained_tan_force_bonus),
        "tangential_speed_penalty": float(tangential_speed_penalty),
        "outward_escape_penalty": float(outward_escape_penalty),
        "near_target_radial_penalty": float(near_target_radial_penalty),
        "phase_eff": float(phase_eff),
        "closure_bonus": float(closure_bonus),
        "closure_penalty": float(closure_penalty),
        "radial_growth_penalty": float(radial_growth_penalty),
        "energy_rel_error": float(energy_rel_error),
        "specific_energy": float(specific_energy),
        "target_energy": float(target_energy),
        "ecc_proxy": float(ecc_proxy),
    }

