import numpy as np

def simulate_orbit(
    steps=120000,
    dt=3600,
    G=6.67430e-11,
    M=1.989e30,
    mass=722,
    pos_init=np.array([0.0, 1.5e11]),
    vel_init=None,
    thrust_vector=None
):
    pos = pos_init.copy()

    if vel_init is None:
        r0 = np.linalg.norm(pos_init)
        v_circular = np.sqrt(G * M / r0)
        vel = np.array([v_circular, 0.0])
    else:
        vel = vel_init.copy()

    trajectory = np.zeros((steps, 2))

    for step in range(steps):
        t = step * dt
        r = np.linalg.norm(pos)

        if r < 1e3 or not np.isfinite(r):
            print(f"[Step {step}] Invalid radius r = {r}, pos = {pos}, breaking.")
            break

        gravity_force = -G * M * pos / (r ** 3)

        if callable(thrust_vector):
            thrust = thrust_vector(t, pos, vel)
        elif isinstance(thrust_vector, np.ndarray):
            thrust = thrust_vector
        else:
            thrust = np.array([0.0, 0.0])

        if not np.all(np.isfinite(thrust)):
            print(f"[Step {step}] Invalid thrust at t={t}: thrust = {thrust}")
            break

        # Optional: limit thrust to prevent explosion
        thrust = np.clip(thrust, -1e10, 1e10)

        acc = (gravity_force + thrust) / mass
        vel += acc * dt
        pos += vel * dt

        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            print(f"[Step {step}] Numerical instability at t={t}, pos = {pos}, vel = {vel}")
            break

        trajectory[step] = pos.copy()

    if step < steps - 1:
        trajectory = trajectory[:step]

    return trajectory


