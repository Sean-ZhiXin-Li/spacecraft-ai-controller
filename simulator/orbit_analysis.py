import numpy as np
import matplotlib.pyplot as plt

def evaluate_orbit_error(trajectory, target_radius):
    """
    Compute the average and standard deviation of radial error
    from a target orbit radius.

    Args:
        trajectory (np.ndarray): (N, 2) array of positions [x, y].
        target_radius (float): Target orbital radius.

    Returns:
        mean_error (float): Mean deviation from target.
        std_error (float): Standard deviation of error.
    """
    radii = np.linalg.norm(trajectory, axis=1)
    errors = np.abs(radii - target_radius)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error


def plot_radius_error(trajectory, target_radius, save_path=None):
    """
    Plot radial error r(t) - target_radius over time.

    Args:
        trajectory (np.ndarray): (N, 2) array of positions.
        target_radius (float): Target orbital radius.
        save_path (str, optional): Path to save the plot. If None, display instead.
    """
    r_list = np.linalg.norm(trajectory, axis=1)
    t_list = np.arange(len(r_list))

    plt.figure(figsize=(8, 4))
    plt.plot(t_list, r_list - target_radius, label="Radial Error", color="purple")
    plt.axhline(0, color="gray", linestyle="--", label="Target Radius")

    plt.xlabel("Time Step")
    plt.ylabel("r(t) - target radius (m)")
    plt.title("Radial Deviation Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved radial error plot to: {save_path}")
    else:
        plt.show()


def plot_radius_error_with_analysis(trajectory, target_radius, window=1000, save_path=None):
    """
    Plot radial error with moving average and max error marker.

    Args:
        trajectory (np.ndarray): (N, 2) array of positions.
        target_radius (float): Desired radius.
        window (int): Window size for moving average.
        save_path (str, optional): Save path if specified.
    """
    r = np.linalg.norm(trajectory, axis=1)
    error = r - target_radius
    steps = np.arange(len(r))
    ma = np.convolve(error, np.ones(window) / window, mode='same')
    max_idx = np.argmax(np.abs(error))
    max_val = error[max_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, error, label="Radial Error", alpha=0.6)
    plt.plot(steps, ma, label=f"Moving Avg (window={window})", linestyle='--', color='orange')
    plt.scatter(max_idx, max_val, color='red', label=f"Max Error: {max_val:.2e}", zorder=5)
    plt.axhline(0, linestyle='--', color='gray')

    plt.xlabel("Time Step")
    plt.ylabel("r(t) - target (m)")
    plt.title("Enhanced Radial Error Analysis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    else:
        plt.show()


def plot_error_histogram(trajectory, target_radius, bins=50, save_path=None):
    """
    Plot histogram of absolute radial errors.

    Args:
        trajectory (np.ndarray): (N, 2) array of positions.
        target_radius (float): Target orbital radius.
        bins (int): Number of histogram bins.
        save_path (str, optional): Save path if specified.
    """
    r = np.linalg.norm(trajectory, axis=1)
    error = np.abs(r - target_radius)

    plt.figure(figsize=(8, 4))
    plt.hist(error, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Radial Error Distribution")
    plt.xlabel("Absolute Error (m)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    else:
        plt.show()


def analyze_error_stats(trajectory, target_radius):
    """
    Print mean, std, and max of radial error.

    Args:
        trajectory (np.ndarray): (N, 2) position array.
        target_radius (float): Target orbital radius.
    """
    r = np.linalg.norm(trajectory, axis=1)
    error = r - target_radius
    mean = np.mean(error)
    std = np.std(error)
    max_error = np.max(np.abs(error))

    print("Radial Error Statistics")
    print(f"  Mean Error : {mean:.2e} m")
    print(f"  Std  Dev   : {std:.2e} m")
    print(f"  Max  Error : {max_error:.2e} m")


