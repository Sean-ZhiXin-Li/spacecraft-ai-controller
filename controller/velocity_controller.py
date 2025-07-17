import numpy as np

def velocity_direction_controller(t, pos, vel):
    if t > 3.0:
        unit_direction = vel / np.linalg.norm(vel)
        return 0.002 * unit_direction
    else:
        return np.array([0.0, 0.0])
