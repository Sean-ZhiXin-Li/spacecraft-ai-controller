import subprocess
import os

REWARD_SETUPS = [
    ("ppo_combined", "combined", 0.5, 5.0, 0.0),
    ("ppo_speed", "speed", 0.0, 0.0, 2.0),
    ("ppo_full", "full", 0.5, 5.0, 2.0),
]

def run_training(tag, reward_mode, w_radius, w_progress, w_speed):
    print(f"\n=== TRAIN {tag} ===")

    env = os.environ.copy()
    env["REWARD_MODE"] = reward_mode
    env["W_RADIUS"] = str(w_radius)
    env["W_PROGRESS"] = str(w_progress)
    env["W_SPEED"] = str(w_speed)
    env["LOG_DIR"] = f"ppo_orbit/{tag}"

    subprocess.run([
        "python", "ppo_orbit/ppo.py",
        "--epochs", "200",
        "--seed", "42",
    ], env=env, check=True)

def main():
    for tag, mode, wr, wp, ws in REWARD_SETUPS:
        run_training(tag, mode, wr, wp, ws)

    print("Done.")

if __name__ == "__main__":
    main()


