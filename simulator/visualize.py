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


import numpy as np
import matplotlib.pyplot as plt

def plot_thrust_quiver(data, title="Thrust Vector Field", step=1000, save_path=None):
    """
    Plot a quiver diagram of thrust vectors over position.

    :param data: A NumPy array of shape (N, 6) or (N, 7) — each row is:
                 - [x, y, vx, vy, Tx, Ty]     ← if shape is (N, 6)
                 - [t, x, y, vx, vy, Tx, Ty]  ← if shape is (N, 7)
    :param title: Title of the plot.
    :param step: Sample every 'step' points to avoid overcrowding.
    :param save_path: Optional path to save the figure as PNG.
    """
    if data.ndim != 2:
        raise ValueError(f"Input data must be 2D, got shape {data.shape}")

    # Automatically handle shape (N, 7) → drop time column
    if data.shape[1] == 7:
        data = data[:, 1:]

    if data.shape[1] != 6:
        raise ValueError(f"Expected shape (N,6) after adjustment, got {data.shape}")

    pos_x = data[::step, 0]
    pos_y = data[::step, 1]
    thrust_x = data[::step, 4]
    thrust_y = data[::step, 5]

    scale = 1e8  # Adjust depending on your thrust magnitude

    plt.figure(figsize=(8, 8))
    plt.quiver(pos_x, pos_y, thrust_x, thrust_y,
               angles='xy', scale_units='xy', scale=scale,
               color='blue', alpha=0.6, width=0.003)
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    else:
        plt.show()
