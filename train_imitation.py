import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load dataset
def load_all_datasets(data_dir, max_files=200):
    file_paths = glob.glob("data/data/preprocessed/merged_expert_dataset.npy")
    file_paths = file_paths[:max_files]
    all_data = [np.load(path) for path in file_paths]
    return np.vstack(all_data)

# Compute derived features
def compute_additional_features(pos, vel):
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    v = np.linalg.norm(vel, axis=1, keepdims=True)
    cos_theta = (pos[:, 0]*vel[:, 0] + pos[:, 1]*vel[:, 1]) / (r.flatten() * v.flatten() + 1e-8)
    cos_theta = cos_theta.reshape(-1, 1)
    return r, v, cos_theta

# Prepare data for training
def prepare_data(data, max_samples=100_000):
    if len(data) > max_samples:
        idx = np.random.choice(len(data), max_samples, replace=False)
        data = data[idx]
    pos = data[:, :2]
    vel = data[:, 2:4]
    thrust = data[:, 4:]

    r, v, cos_theta = compute_additional_features(pos, vel)
    X = np.hstack([pos, vel, r, v, cos_theta])
    y = thrust
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='tanh',
        solver='adam',
        learning_rate='adaptive',
        alpha=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=3000,
        random_state=42,
        verbose=True
    )
    print("Training MLP model (V6.2)...")
    model.fit(X_train, y_train)
    print("Training complete!")
    return model

# ======== Evaluation & Visualization ========
def evaluate(model, X_test, y_test, save_path="plots/thrust_vectors_v6_2.png"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6e}")

    # Only show samples with valid thrust
    valid = np.linalg.norm(y_test, axis=1) > 1e-6
    X_test = X_test[valid]
    y_test = y_test[valid]
    y_pred = y_pred[valid]

    # Plot true vs predicted thrust vectors
    plt.figure(figsize=(10, 8))
    for i in range(min(100, len(X_test))):
        pos_x, pos_y = X_test[i][:2]
        tx, ty = y_test[i]
        px, py = y_pred[i]
        plt.quiver(pos_x, pos_y, tx, ty, angles='xy', scale_units='xy',
                   color='green', width=0.004, alpha=0.6, label='True' if i == 0 else "")
        plt.quiver(pos_x, pos_y, px, py, angles='xy', scale_units='xy',
                   color='red', width=0.004, alpha=0.4, label='Predicted' if i == 0 else "")

    plt.title("Thrust Vectors – Green: True, Red: Predicted (V6.2)")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/expert_trajectories_v6_2",
                        help="Directory containing expert .npy trajectory files")
    parser.add_argument("--model_out", type=str, default="controller/imitation_policy_model_V6_2.joblib",
                        help="Path to save trained model")
    parser.add_argument("--scaler_out", type=str, default="controller/state_scaler_V6_2.joblib",
                        help="Path to save feature scaler")
    return parser.parse_args()

# Main entry
def main():
    args = parse_args()

    print("Loading data from:", args.data_dir)
    data = load_all_datasets(args.data_dir)
    X_train, X_test, y_train, y_test = prepare_data(data)

    print("Scaling input features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_model(X_train_scaled, y_train)
    evaluate(model, X_test_scaled, y_test)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump(model, args.model_out)
    dump(scaler, args.scaler_out)

    print("Model saved!")
    print("Model:", args.model_out)
    print("Scaler:", args.scaler_out)

if __name__ == "__main__":
    main()