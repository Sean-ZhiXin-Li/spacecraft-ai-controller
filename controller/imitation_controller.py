import joblib
import numpy as np

class ImitationController:
    """
    Imitation Controller using a pre-trained MLP model to generate thrust vector
    based on the current orbital state (position and velocity).

    This class is compatible with the common controller interface: __call__(t, pos, vel).
    """

    def __init__(self,
                 model_path="imitation_policy_model.joblib",
                 clip=True,
                 verbose=False):
        """
        Initialize the imitation controller.

        Args:
            model_path (str): Path to the saved joblib model file.
            clip (bool): Whether to clip the predicted thrust to [-1, 1].
            verbose (bool): If True, prints debug information when called.
        """
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.clip = clip
        self.verbose = verbose

    def __call__(self, t, pos, vel):
        """
        Closed-loop control interface. Given the current time, position, and velocity,
        returns a thrust vector [Tx, Ty] predicted by the model.

        Args:
            t (float): Current simulation time (not used but kept for compatibility).
            pos (np.ndarray): Current position vector [x, y].
            vel (np.ndarray): Current velocity vector [vx, vy].

        Returns:
            np.ndarray: Predicted thrust vector [Tx, Ty], clipped to [-1, 1] range.
        """
        # Concatenate position and velocity into a 4D input vector
        obs = np.array([*pos, *vel]).reshape(1, -1)

        # Predict the thrust using the loaded MLP model
        action = self.model.predict(obs).flatten()

        # Optionally clip action values to match environment's action space
        if self.clip:
            action = np.clip(action, -1.0, 1.0)

        # If verbose mode is enabled, print debugging information
        if self.verbose:
            print(f"[ImitationController] t={t:.2f}, pos={pos}, vel={vel}, thrust={action}")

        return action

