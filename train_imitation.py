import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# Feature engineering

def compute_additional_features(pos, vel):
    r = np.linalg.norm(pos, axis=1).reshape(-1, 1)
    v = np.linalg.norm(vel, axis=1).reshape(-1, 1)
    cos_theta = np.sum(pos * vel, axis=1).reshape(-1, 1) / (r * v + 1e-8)
    return r, v, cos_theta

# Load expert dataset

def load_all_datasets(folder="data/data/dataset"):
    folder = os.path.abspath(folder)
    files = sorted(glob.glob(os.path.join(folder, "expert_dataset_*.npy")))
    print("Matched files:", files)
    all_data = []
    for f in files:
        print(f" Loading dataset: {f}")
        data = np.load(f)
        all_data.append(data)
    return np.vstack(all_data)

# Prepare input/output

def prepare_data(data):
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
    print(" Training MLP model (V6)...")
    model.fit(X_train, y_train)
    print(" Training complete!")
    return model

# Evaluation

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f" Test MSE: {mse:.6e}")

    valid = np.linalg.norm(y_test, axis=1) > 1e-6
    X_test = X_test[valid]
    y_test = y_test[valid]
    y_pred = y_pred[valid]

    plt.figure(figsize=(10, 8))
    for i in range(min(100, len(X_test))):
        pos_x, pos_y = X_test[i][:2]
        tx, ty = y_test[i]
        px, py = y_pred[i]
        plt.quiver(pos_x, pos_y, tx, ty, angles='xy', scale_units='xy',
                   color='green', width=0.004, alpha=0.6, label='True' if i == 0 else "")
        plt.quiver(pos_x, pos_y, px, py, angles='xy', scale_units='xy',
                   color='red', width=0.004, alpha=0.4, label='Predicted' if i == 0 else "")

    plt.title("Thrust Vectors – Green: True, Red: Predicted (V6)")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Main

def main():
    print(" Loading data...")
    data = load_all_datasets()
    X_train, X_test, y_train, y_test = prepare_data(data)

    print(" Scaling input features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_model(X_train_scaled, y_train)
    evaluate(model, X_test_scaled, y_test)

    os.makedirs("controller", exist_ok=True)
    dump(model, "controller/imitation_policy_model_V6.joblib")
    dump(scaler, "controller/state_scaler_V6.joblib")
    print("V6 model saved to controller/")
    print("Model: imitation_policy_model_V6.joblib")
    print("Scaler: state_scaler_V6.joblib")

if __name__ == "__main__":
    main()
