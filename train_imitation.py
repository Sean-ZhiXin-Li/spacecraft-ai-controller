import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load all expert dataset files

def load_all_datasets(folder="data/data/dataset"):
    folder = os.path.abspath(folder)
    """
    Load and stack all expert_dataset_*.npy files from the dataset folder.
    Each row contains: [pos_x, pos_y, vel_x, vel_y, thrust_x, thrust_y]
    """
    files = sorted(glob.glob(os.path.join(folder, "expert_dataset_*.npy")))
    print("Matched files:", files)
    all_data = []
    for f in files:
        print(f" Loading dataset: {f}")
        data = np.load(f)
        all_data.append(data)
    return np.vstack(all_data)

# Step 2: Prepare training input and output

def prepare_data(data):
    """
    Split data into input (X) and output (y)
    X = [pos_x, pos_y, vel_x, vel_y]
    y = [thrust_x, thrust_y]
    Returns train/test split
    """
    X = data[:, :4]
    y = data[:, 4:]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the neural network

def train_model(X_train, y_train):
    """
    Train a neural network using MLPRegressor
    Input: 4D state vector
    Output: 2D thrust vector
    """
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),   # Two hidden layers with 64 neurons each
        activation='tanh',             # Nonlinear activation function
        solver='adam',                 # Optimizer
        max_iter=1000,
        random_state=42
    )
    print(" Training MLP model...")
    model.fit(X_train, y_train)
    print(" Training complete!")
    return model

# Step 4: Evaluate and visualize results
def evaluate(model, X_test, y_test):
    """
    Print test MSE and plot first 100 predicted vs actual thrust vectors
    using real position (pos_x, pos_y) as arrow start point.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f" Test MSE (mean squared error): {mse:.6f}")

    # Create plot
    plt.figure(figsize=(10, 8))

    for i in range(min(100, len(X_test))):
        pos_x, pos_y = X_test[i][:2]
        tx, ty = y_test[i]
        px, py = y_pred[i]

        # scale factor to amplify short thrust vectors (for visualization only)
        scale_factor = 1e11

        # true thrust in green
        plt.quiver(pos_x, pos_y, tx, ty,
                   angles='xy', scale_units='xy',
                   scale=1/scale_factor, color='green',
                   alpha=0.7, label='True' if i == 0 else "")

        # predicted thrust in red
        plt.quiver(pos_x, pos_y, px, py,
                   angles='xy', scale_units='xy',
                   scale=1/scale_factor, color='red',
                   alpha=0.5, label='Pred' if i == 0 else "")

    plt.title("Thrust Vectors on Position Plane: Green=True / Red=Predicted")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Main function

def main():
    print(" Starting imitation policy training...")
    data = load_all_datasets()
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    # Optional: save the trained model
    from joblib import dump
    dump(model, "imitation_policy_model.joblib")
    print(" Model saved to: imitation_policy_model.joblib")

if __name__ == "__main__":
    main()
