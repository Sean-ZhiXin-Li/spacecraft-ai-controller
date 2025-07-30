import numpy as np
from controller.expert_controller import ExpertController
from simulator.visualize import plot_trajectory
import matplotlib.pyplot as plt

# === 基础常数 ===
G = 6.67430e-11  # 万有引力常数
M = 1.989e30     # 恒星质量（太阳）
mass = 722       # 飞船质量 (kg)
target_radius = 7.5e12
dt = 300         # 时间步长 (s) - 减小以获得更平滑轨道
steps = 2000000   # 模拟步数（建议先测试短一点）

# === 初始状态 ===
pos_init = np.array([0.0, target_radius * (1/3)])  # 太阳正上方、距离为目标轨道1/3
r0 = np.linalg.norm(pos_init)
v_circular = np.sqrt(G * M / r0)
vel_init = np.array([v_circular, 0.0])  # 切向初速度

# === ExpertController 初始化 ===
controller = ExpertController(
    target_radius=target_radius,
    G=G,
    M=M,
    radial_gain=0.3,
    tangential_gain=0.05,
    thrust_cap=0.2,
    enable_error_feedback=True,
    enable_turn_penalty=True,
    enable_slowdown=True
)


# === 状态记录器 ===
trajectory, velocities, accelerations, radii = [], [], [], []

pos = pos_init.copy()
vel = vel_init.copy()
t = 0.0

# === 控制主循环 ===
print(f"{'Step':>5} | {'r (m)':>12} | {'v (m/s)':>12} | {'thrust':>20} | {'a (m/s²)':>20}")
print("=" * 90)

for step in range(steps):
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    thrust = controller(t, pos, vel)
    acc = thrust / mass

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

# === 轨道图像可视化 ===
plot_trajectory(
    trajectory=trajectory,
    title="ExpertController Orbit Debug",
    target_radius=target_radius,
    arrows=True
)

# === 半径随时间变化图（可选分析）===
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
