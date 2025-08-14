import numpy as np

class ZeroThrustController:
    def act(self, obs):
        # action in normalized thrust space [-1,1]^2
        return np.array([0.0, 0.0], dtype=np.float64)
