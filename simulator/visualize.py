import  matplotlib.pyplot as plt

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
    plt.figure(figsize=(6, 6))

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

