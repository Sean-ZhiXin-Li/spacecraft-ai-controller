import joblib
import numpy as np

class ImitationController:
    """
    Imitation Controller using a pre-trained MLP model + scaler
    to generate thrust vector based on the normalized orbital state.
    """

    def __init__(self,
                 model_path="imitation_policy_model_V5.joblib",
                 scaler_path="state_scaler_V5.joblib",
                 clip=True,
                 verbose=False):
        """
        Args:
            model_path (str): Path to trained MLP model.
            scaler_path (str): Path to StandardScaler for input normalization.
            clip (bool): Whether to clip output thrust.
            verbose (bool): Print debug info.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.clip = clip
        self.verbose = verbose

    def __call__(self, t, pos, vel):
        """
        Closed-loop control interface for V6 model (7 input features).
        Args:
            t (float): current time (not used)
            pos (np.ndarray): [x, y]
            vel (np.ndarray): [vx, vy]
        Returns:
            np.ndarray: [thrust_x, thrust_y]
        """
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        cos_theta = np.dot(pos, vel) / (r * v + 1e-8)

        # Assemble 7D input vector
        state = np.array([*pos, *vel, r, v, cos_theta], dtype=np.float32)
        scaled = self.scaler.transform([state])[0]
        thrust = self.model.predict([scaled])[0]

        if self.clip:
            thrust = np.clip(thrust, -1.0, 1.0)

        if self.verbose:
            print(f"[V6] t={t:.1f}, pos={pos}, vel={vel}, thrust={thrust}")

        return thrust

