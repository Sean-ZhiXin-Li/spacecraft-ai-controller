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
    """
    Simulate orbital motion under gravity and thrust.
    Returns np.array of shape (steps, 2), each row = position [x, y].
    """
    pos = pos_init.copy()

    # Initial velocity setup
    if vel_init is None:
        r = np.linalg.norm(pos_init)
        v_mag = np.sqrt(G * M / r)
        direction = np.array([-pos[1], pos[0]]) / r  # tangential unit vector
        vel = v_mag * direction
    else:
        vel = vel_init.copy()

    trajectory = []

    for step in range(steps):
        t = step * dt
        r = np.linalg.norm(pos)
        gravity_force = -G * M * pos / (r ** 3)

        # Dynamic or constant thrust
        if callable(thrust_vector):
            thrust = thrust_vector(t, pos, vel)
        elif isinstance(thrust_vector, np.ndarray):
            thrust = thrust_vector
        else:
            thrust = np.array([0.0, 0.0])

        # Update physics
        total_force = gravity_force + thrust
        acc = total_force / mass
        vel += acc * dt
        pos += vel * dt

        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            print(f"[Step {step}] Numerical instability detected, breaking simulation.")
            break

        trajectory.append(pos.copy())

    trajectory = np.array(trajectory)
    if trajectory.shape[0] < steps:
        print(f"Simulation terminated early at step {trajectory.shape[0]} due to numerical instability.")
    return trajectory

