import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(
        trajectory,
        title="Orbit",
        target_radius=None,
        arrows=False,
        others=None
):
    """
    Plots the orbital trajectory of a spacecraft with optional target orbit, direction arrow, and additional comparison trajectories.
    :param trajectory: The main trajectory to plot, shape = (N, 2).
    :param title: Title of the plot.
    :param target_radius: If provided, draws a dashed circle representing a target orbit.
    :param arrows: If true, shows an arrow indicating the initial direction of motion.
    :param others: Optional list of other trajectories to compare.
                   Each item is a tuple (trajectories_array, label).
    Example usage:
                   plot_trajectory(traj, target_radius=100, arrows=True,
                                   others=[(baseline, "No Thrust")])
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Automatically obtain the maximum radius
    max_radius = np.max(np.linalg.norm(trajectory, axis=1))
    if target_radius is not None:
        max_radius = max(max_radius, target_radius)
    buffer = 0.2 * max_radius
    ax.set_xlim(-max_radius - buffer, max_radius + buffer)
    ax.set_ylim(-max_radius - buffer, max_radius + buffer)

    # Plot main trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            label='Main trajectory', linewidth=2)

    # Plot initial direction arrow (based on last two points)
    if arrows and len(trajectory) > 10:
        x0, y0 = trajectory[-10]
        x1, y1 = trajectory[-1]
        dx, dy = x1 - x0, y1 - y0
        ax.quiver(x0, y0, dx, dy,
                  angles='xy', scale_units='xy', scale=0.1,
                  color='blue', width=0.004, zorder=10)

    # Plot additional comparison trajectories
    if others:
        for traj, lbl in others:
            ax.plot(traj[:, 0], traj[:, 1], linestyle='--', label=lbl)

    # Plot central body (e.g., Sun)
    ax.scatter([0], [0], color='orange', label='Sun', s=100, zorder=5)

    # Plot target orbit as dashed circle (if provided)
    if target_radius is not None and target_radius > 0:
        circle = plt.Circle((0, 0), target_radius,
                            color='gray', linestyle='--', fill=False,
                            linewidth=1.5, label='Target Orbit', zorder=1)
        ax.add_patch(circle)

    # Configure plot appearance
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_radius_vs_time(trajectory, dt, title="Radius vs Time"):
    """
    Plot the radial distance r(t) of the spacecraft over time.
    :param trajectory: np.array of shape (N, 2), each row is position [x,y].
    :param dt: time step size used in simulation.
    :param title: plot title.
    """
    time = np.arange(len(trajectory)) * dt  # Time axis
    radii = np.linalg.norm(trajectory, axis=1)  # r(t)

    plt.figure(figsize=(8, 4))
    plt.plot(time, radii, label="r(t)", color='green')
    plt.axhline(np.mean(radii), color='gray', linestyle='--', label="Mean radius")
    plt.xlabel("Time")
    plt.ylabel("Radius (Distance from center)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
