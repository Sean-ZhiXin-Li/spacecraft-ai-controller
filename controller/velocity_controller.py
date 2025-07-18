import numpy as np

def radial_controller(t, pos, vel):
    """
    Thrust controller that aligns thrust direction with current velocity.
    Only activates after t > 3.0 second.
    :param t: Current (float).
    :param pos: Current position [x, y].
    :param vel: Current velocity [vx, vy].
    :return: np.ndarray: Thrust vector[Tx,Ty].
    """
    if t <= 3600 or np.linalg.norm(pos) == 0:
        return np.array([0.0, 0.0])
    unit_radial = pos / np.linalg.norm(pos)
    return 0.1 * unit_radial


def get_thrust(t, pos, vel, mode = "continuous"):
    """
    General thrust controller supporting multiple strategies:
        - "continuous": constant thrust in fixed direction.
        - "impulse": periodic thrust(e.g., 1s on every 5s).
        - "radial_controller": thrust directed radially away from the Sun (or central mass).
    :param t: Current (float).
    :param pos: Current position [x, y].
    :param vel: Current velocity [vx, vy].
    :param mode: control msde("continuous", "impulse", "radial_controller").
    :return: np.ndarray: Thrust vector[Tx, Ty].
    """
    thrust_strength = 0.1
    direction = np.array([1.0, 0.0])  # Default fixed direction (positive x-axis)

    if mode == "continuous":
        return thrust_strength * direction

    elif mode == "impulse":
        period = 5.0  # total period duration(seconds)
        duration = 1.0  # thrust on-duration within period
        if t % period < duration:
            return thrust_strength * direction
        else:
            return np.array([0.0, 0.0])

    elif mode == "velocity_direction":
        return radial_controller(t, pos, vel)

    else:
        return np.array([0.0, 0.0])  # Invalid mode -> no trust


def tangential_controller(t, pos, vel):
    """
    The thrust direction is tangential (perpendicular to the radial direction), which changes the angular momentum.
    """
    if np.linalg.norm(pos) == 0:
        return np.array([0.0, 0.0])
    radial_dir = pos / np.linalg.norm(pos)
    tangential_dir = np.array([-radial_dir[1], radial_dir[0]])  # Rotate counterclockwise by 90 degrees
    return 0.05 * tangential_dir

