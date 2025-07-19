import numpy as np
from scipy.spatial.transform import Rotation as R

def add_attitude_noise(thrust_vec, max_angle_deg = 10):
    """
    Add small random rotation to thrust vector to simulate attitude error.
    :param thrust_vec: np.array [Tx, Ty], the original thrust direction.
    :param max_angle_deg: Maximum angle of deviation in degrees.
    :return: np.array, rotated thrust vector.
    """
    norm = np.linalg.norm(thrust_vec)
    if norm == 0:
        return thrust_vec

    # Create random 2D rotation
    angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return  rotation_matrix @ thrust_vec