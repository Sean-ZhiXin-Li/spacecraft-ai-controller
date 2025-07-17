import numpy as np

def evaluate_orbit_error(trajectory, target_radius):
    """
    Evaluate how far the spacecraft trajectory is from the ideal circular orbit.
    :param trajectory: np.array of shape(N, 2)
    :param target_radius: desired orbit radius (float)
    :return: mean radial error (float), std radial error (float)
    """
    radii = np.linalg.norm(trajectory, axis = 1)
    errors = np.abs(radii - target_radius)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error
