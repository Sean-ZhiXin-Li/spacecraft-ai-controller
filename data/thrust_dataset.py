import numpy as np
import os

class ThrustDataset:
    def __init__(self):
        self.data = []  # List to store (pos, vel, thrust)

    def add(self, pos, vel, thrust):
        self.data.append([
            pos[0], pos[1],
            vel[0], vel[1],
            thrust[0], thrust[1]
        ])

    def __call__(self, t, pos, vel, controller):
        """
        Allows the dataset object to act like a function that:
        1. Gets thrust from the controller
        2. Records the state and thrust
        3. Returns the thrust
        """
        thrust = controller(t, pos, vel)  # controller must be callable
        self.add(pos, vel, thrust)
        return thrust

    def save(self, path_prefix="dataset"):
        os.makedirs("data/dataset", exist_ok=True)
        array = np.array(self.data)
        np.save(f"data/dataset/{path_prefix}.npy", array)
        np.savetxt(f"data/dataset/{path_prefix}.csv", array,
                   delimiter=",",
                   header="pos_x,pos_y,vel_x,vel_y,thrust_x,thrust_y",
                   comments='')
        print(f"Dataset saved to: data/dataset/{path_prefix}.csv/.npy")
