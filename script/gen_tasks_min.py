import os, json

def mkdir(p): os.makedirs(p, exist_ok=True)

OUT_DIR = os.path.join("ab", "day36", "task_specs")

def main():
    mkdir(OUT_DIR)

    # Task 1: medium circular orbit, medium mass/thrust
    t1 = {
        "name": "circular_r_1e12_demo",
        "orbit_type": "circular_r",
        "params": {"r": 1.0e12},
        "init_state": {"pos":[1.0e12, 0.0], "vel_angle_deg": 20.0, "vel_scale": 1.0},
        "mass": 5.0e9,                 # 5 billion kg (mega ship)
        "thrust_newton": 5.0e6,        # 5 MN thrust scale
        "r_target": 1.0e12,
        "max_steps": 20000,
        "seed": 20250820
    }

    # Task 2: larger orbit, same mega mass, stronger thrust
    t2 = {
        "name": "circular_r_7p5e12_demo",
        "orbit_type": "circular_r",
        "params": {"r": 7.5e12},
        "init_state": {"pos":[7.5e12, 0.0], "vel_angle_deg": 30.0, "vel_scale": 1.0},
        "mass": 5.0e9,                 # mega ship
        "thrust_newton": 2.0e7,        # 20 MN thrust scale
        "r_target": 7.5e12,
        "max_steps": 20000,
        "seed": 20250821
    }

    for t in (t1, t2):
        fp = os.path.join(OUT_DIR, f"{t['name']}.json")
        with open(fp, "w") as f:
            json.dump(t, f, indent=2)
        print("[gen] wrote", fp)

if __name__ == "__main__":
    main()
