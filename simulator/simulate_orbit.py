import numpy as np

def simulate_orbit(
    steps = 6000,
    dt = 0.1,
    G = 1.0,
    M = 1000.0,
    mass = 1.0,
    pos_init = np.array([100.0,0.0]),
    vel_init = None,
    thrust_vector = None
):
    """
    simulate the orbital trajectory a spacecraft under gravity and dynamic thrust.
    :param steps: Total number o simulation steps.
    :param dt: Time steps size.
    :param G : Gravitational constant.
    :param M : Mass of the central body (e.g., the sun).
    :param mass: Mass of the spacecraft.
    :param pos_init: Initial position of the spacecraft [x, y].
    :param vel_init: Initial velocity of the spacecraft [vx, vy]. If None use circular orbit velocity.
    :param thrust_vector:
            -If np.array : Constant thrust vector [Tx, Ty].
            -If function : Should be thrust_vector(t, pos, vel) - returns thrust np.array.
    :return:
            np.array: Trajectory of the spacecraft, shape = (steps, 2), where each row is a position [x, y].
    """
    pos = pos_init.copy()

    if vel_init is None:
        r = np.linalg.norm(pos_init)
        v_mag = np.sqrt(G * M / r)  # circular orbit speed
        direction = np.array([-pos[1],pos_init[0]]) / r  # tangential direction
        vel = v_mag * direction
    else:
        vel = vel_init.copy()

    trajectory = []

    for step in range(steps):
        t = step * dt
        r = np.linalg.norm(pos)
        gravity_force = -G * M * pos / (r ** 3)

        if callable(thrust_vector):
            thrust = thrust_vector(t, pos, vel)
        elif isinstance(thrust_vector, np.ndarray):
            thrust =thrust_vector
        else:
            thrust = np.array([0.0, 0.0])  #default: no thrust

        total_force = gravity_force + thrust
        acc = total_force / mass
        vel += acc * dt
        pos += vel * dt
        trajectory.append(pos.copy())

    return np.array(trajectory)

