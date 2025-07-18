import  matplotlib.pyplot as plt
import  numpy as np

def plot_trajectory(
        trajectory,
        title = "Orbit",
        target_radius = None,
        arrows  = False,
        others = None
):
    """
    Plots the orbital trajectory of a spacecraft with optional target orbit,  direction arrow, and additional comparison trajectories.
    :param trajectory: The main trajectory to plot, shape = (N, 2).
    :param title: Title of the plot.
    :param target_radius: If provided, draws a dashed circle representing a target orbit.
    :param arrows: If ture, shows an arrow indicating the initial direction of motion.
    :param others: Optional list of other trajectories to compare.
                    Each item id a tuple (trajectories_array, label).
    Example usage:
                    plot_trajectory(traj, target_radius = 100, arrow = True,
                                    others = [(baseline, "No Thrust")])
    """
    plt.figure(figsize=(8, 8))
    max_radius = np.max(np.linalg.norm(trajectory, axis=1))
    if max_radius > 1e9:
        plt.xlim(-1.2 * max_radius, 1.2 * max_radius)
        plt.ylim(-1.2 * max_radius, 1.2 * max_radius)

    # Plot main trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1],
             label = 'Main trajectory', linewidth = 2)

    # Plot initial direction arrow (based on first two points)
    if arrows and len(trajectory) > 10:
        x0, y0 = trajectory[-10]
        x1,y1 = trajectory[-1]
        dx, dy = x1 - x0, y1 - y0
        plt.quiver(x0, y0, dx, dy,
                   angles = 'xy',scale_units = 'xy', scale = 0.1, color = 'blue', width = 0.004, zorder = 10)

    # Plot additional comparison trajectories
    if others:
        for traj, lbl in others:
            plt.plot(traj[:, 0], traj[:, 1], linestyle = '--', label = lbl)

    # Plot central body(e.g., Sun)
    plt.scatter([0], [0], color = 'orange', label = 'Sun', s = 100, zorder = 5)

    # Plot target orbit as dashed circle(if provided)
    if target_radius:
        circle = plt.Circle((0, 0), target_radius, color = 'gray', linestyle = '--', fill = False, label = 'Target Orbit')
        plt.gca().add_artist(circle)

    # Configure plot appearance
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_radius_vs_time(trajectory, dt, title = "Radius vs Time"):
    """
    Plot the radial distance r(t) of the spacecraft over time.
    :param trajectory: np.array of shape (N, 2), each row id position [x,y].
    :param dt: time step size used in simulation.
    :param title: plot title.
    """
    time = np.arange(len(trajectory)) * dt  # Time axis
    radii = np.linalg.norm(trajectory, axis = 1)  # r(t)

    plt.figure(figsize=(8, 4))
    plt.plot(time, radii, label = "r(t)", color = 'green')
    plt.axhline(np.mean(radii), color = 'gray', linestyle = '--', label = "Mean rafius")
    plt.xlabel("Time")
    plt.ylabel("Radius(Distance from center)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
