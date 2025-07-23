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


