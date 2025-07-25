import numpy as np
import matplotlib.pyplot as plt

def evaluate_orbit_error(trajectory, target_radius):
    """
    Evaluate how far the spacecraft trajectory is from the desired orbit radius.

    Args:
        trajectory (np.ndarray): Shape (N, 2), position history of the spacecraft.
        target_radius (float): Desired circular orbit radius.

    Returns:
        mean_error (float): Mean radial deviation from the target radius.
        std_error (float): Standard deviation of the radial error.
    """
    # Compute radius r(t) = sqrt(x^2 + y^2) at each timestep
    radii = np.linalg.norm(trajectory, axis=1)

    # Compute absolute error at each timestep
    errors = np.abs(radii - target_radius)

    # Compute mean and standard deviation of the error
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    return mean_error, std_error


def plot_radius_error(trajectory, target_radius, save_path=None):
    """
    Plot the radial error (r(t) - target_radius) over time.

    Args:
        trajectory (np.ndarray): Shape (N, 2), position history of the spacecraft.
        target_radius (float): Desired orbit radius.
        save_path (str, optional): If provided, saves the figure to this path.
                                   If None, displays the plot interactively.
    """
    # Calculate radius at each time step
    r_list = np.linalg.norm(trajectory, axis=1)

    # Create time step index (0 to N-1)
    t_list = np.arange(len(r_list))

    # Create the figure
    plt.figure(figsize=(8, 4))

    # Plot the radial deviation from the target
    plt.plot(t_list, r_list - target_radius, label="Radial Error", color="purple")

    # Draw a horizontal line at y=0 representing perfect target radius
    plt.axhline(0, color="gray", linestyle="--", label="Target Radius")

    # Set axis labels and title
    plt.xlabel("Time Step")
    plt.ylabel("r(t) - target radius (meters)")
    plt.title("Radial Deviation Over Time")

    # Add grid and legend
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Saved radial error plot to: {save_path}")
    else:
        plt.show()

def plot_radius_error_with_analysis(trajectory, target_radius, window=1000, save_path=None):
    """
    Plot radial error curve with max-error marker and moving average.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    r = np.linalg.norm(trajectory, axis=1)
    error = r - target_radius
    steps = np.arange(len(r))

    # Moving average
    ma = np.convolve(error, np.ones(window) / window, mode='same')

    # Max error
    max_idx = np.argmax(np.abs(error))
    max_val = error[max_idx]

    # Plot
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
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
    Print basic stats of radial error.
    """
    r = np.linalg.norm(trajectory, axis=1)
    error = r - target_radius
    mean = np.mean(error)
    std = np.std(error)
    max_error = np.max(np.abs(error))
    print(" Error Stats")
    print(f"  Mean error: {mean:.2e}")
    print(f"  Std  error: {std:.2e}")
    print(f"  Max  error: {max_error:.2e}")

