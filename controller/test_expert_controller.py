import numpy as np
from controller.expert_controller import ExpertController
from simulator.visualize import plot_trajectory
import matplotlib.pyplot as plt

G = 6.67430e-11  #
M = 1.989e30
target_radius = 7.5e12
dt = 1000
steps = 600000
mass = 721.9


pos_init = np.array([0.0, target_radius * (1/3)])
r0 = np.linalg.norm(pos_init)
v_circular = np.sqrt(G * M / r0)
vel_init = np.array([v_circular, 0.0])

controller = ExpertController(
    target_radius=7.5e12,
    G=6.67430e-11,
    M=1.989e30,
    mass=1000,
    radial_gain=12.0,
    tangential_gain=8.0,
    damping_gain=4.0,
    thrust_limit=1.0,
    enable_damping=True
)

trajectory, velocities, accelerations, radii = [], [], [], []

pos = pos_init.copy()
vel = vel_init.copy()
t = 0.0

print(f"{'Step':>5} | {'r (m)':>12} | {'v (m/s)':>12} | {'thrust':>20} | {'a (m/s²)':>20}")
print("=" * 90)

for step in range(steps):
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    thrust = controller(t, pos, vel)
    acc = np.array(thrust) / mass

    trajectory.append(pos.copy())
    velocities.append(vel.copy())
    accelerations.append(acc.copy())
    radii.append(r)

    if step < 100:
        print(f"{step:5d} | {r:12.4e} | {v:12.4e} | {str(thrust):>20} | {str(acc):>20}")

    # 欧拉积分
    vel += acc * dt
    pos += vel * dt
    t += dt

trajectory = np.array(trajectory)
radii = np.array(radii)

plot_trajectory(
    trajectory=trajectory,
    title="ExpertController Orbit Debug",
    target_radius=target_radius,
    arrows=True
)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(radii)) * dt / 86400, radii, label='r(t)')
plt.axhline(y=target_radius, color='r', linestyle='--', label='Target Orbit')
plt.xlabel('Time (days)')
plt.ylabel('Orbital Radius (m)')
plt.title('Radius vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
