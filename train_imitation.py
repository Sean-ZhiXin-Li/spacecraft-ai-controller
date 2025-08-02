import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# Load all expert dataset files

def load_all_datasets(folder="data/dataset"):
    folder = os.path.abspath(folder)
    files = sorted(glob.glob(os.path.join(folder, "expert_dataset_*.npy")))
    print("Matched files:", files)
    all_data = []
    for f in files:
        print(f" Loading dataset: {f}")
        data = np.load(f)
        all_data.append(data)
    return np.vstack(all_data)

# Prepare training input and output

def prepare_data(data):
    X = data[:, :4]  # [pos_x, pos_y, vel_x, vel_y]
    y = data[:, 4:]  # [thrust_x, thrust_y]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the neural network

def train_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='tanh',
        solver='adam',
        learning_rate='adaptive',
        alpha=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=2000,
        random_state=42,
        verbose=True
    )
    print(" Training MLP model...")
    model.fit(X_train, y_train)
    print(" Training complete!")
    return model

#  Evaluate and visualize

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f" Test MSE: {mse:.6e}")

    plt.figure(figsize=(10, 8))
    for i in range(min(100, len(X_test))):
        pos_x, pos_y = X_test[i][:2]
        tx, ty = y_test[i]
        px, py = y_pred[i]

        # Green = true, Red = predicted
        plt.quiver(pos_x, pos_y, tx, ty,
                   angles='xy', scale_units='xy',
                   color='green', width=0.004,
                   alpha=0.6, label='True' if i == 0 else "")
        plt.quiver(pos_x, pos_y, px, py,
                   angles='xy', scale_units='xy',
                   color='red', width=0.004,
                   alpha=0.4, label='Predicted' if i == 0 else "")

    plt.title("Thrust Vectors – Green: True, Red: Predicted")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Main training pipeline

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

    # === Save model and scaler ===
    os.makedirs("controller", exist_ok=True)
    dump(model, "imitation_policy_model_V5.joblib")
    dump(scaler, "controller/state_scaler_V5.joblib")
    print(" Model saved to: imitation_policy_model_V5.joblib")
    print(" Scaler saved to: controller/state_scaler_V5.joblib")

if __name__ == "__main__":
    main()
