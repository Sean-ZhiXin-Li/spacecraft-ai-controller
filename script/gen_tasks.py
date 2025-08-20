import os, json, argparse, random
from math import isfinite

def mkdir(p): os.makedirs(p, exist_ok=True)

def _safe_float(x, default):
    try:
        v = float(x)
        return v if isfinite(v) else default
    except Exception:
        return default

def circular_tasks(n, seed, radii, mass, thrust_list, max_steps):
    random.seed(seed)
    out = []
    for i in range(n):
        r = random.choice(radii)
        thrust_n = random.choice(thrust_list)
        t = {
            "name": f"circ_r_{int(r):d}_{i}",
            "orbit_type": "circular_r",
            "params": {"r": r},
            "init_state": {"pos":[r,0.0], "vel_angle_deg": 20.0, "vel_scale": 1.0},
            "mass": mass,
            "thrust_newton": thrust_n,
            "r_target": r,
            "max_steps": max_steps,
            "seed": seed + i
        }
        out.append(t)
    return out

def elliptic_tasks(n, seed, rp_choices, ra_ratio_choices, mass, thrust_list, max_steps):
    random.seed(seed + 1001)
    out = []
    for i in range(n):
        rp = random.choice(rp_choices)
        ra = rp * random.choice(ra_ratio_choices)  # ra > rp
        ecc = (ra - rp) / (ra + rp)
        r_target = 0.5 * (rp + ra)  # use semi-major axis as ref
        thrust_n = random.choice(thrust_list)
        t = {
            "name": f"elli_rp_{int(rp):d}_ra_{int(ra):d}_{i}",
            "orbit_type": "elliptic_e",
            "params": {"rp": rp, "ra": ra, "ecc": ecc},
            "init_state": {"pos":[rp,0.0], "vel_angle_deg": 15.0, "vel_scale": 1.0},
            "mass": mass,
            "thrust_newton": thrust_n,
            "r_target": r_target,
            "max_steps": max_steps,
            "seed": seed + 10000 + i
        }
        out.append(t)
    return out

def transfer_tasks(n, seed, radii, mass, thrust_list, max_steps):
    random.seed(seed + 2002)
    out = []
    Rs = list(radii)
    for i in range(n):
        r1, r2 = random.sample(Rs, 2)
        thrust_n = random.choice(thrust_list)
        t = {
            "name": f"transfer_{int(r1):d}_to_{int(r2):d}_{i}",
            "orbit_type": "transfer_hohmann",
            "params": {"r1": r1, "r2": r2},
            "init_state": {"pos":[r1,0.0], "vel_angle_deg": 0.0, "vel_scale": 1.0},
            "mass": mass,
            "thrust_newton": thrust_n,
            "r_target": r2,   # final target radius
            "max_steps": max_steps,
            "seed": seed + 20000 + i
        }
        out.append(t)
    return out

def perturbed_tasks(n, seed, base_r, mass, thrust_list, max_steps,
                    ang_choices=( -30, -20, -10, 10, 20, 30 ),
                    vel_scales=(0.8, 0.9, 1.0, 1.1, 1.2)):
    random.seed(seed + 3003)
    out = []
    for i in range(n):
        r = random.choice(base_r)
        ang = random.choice(ang_choices)
        vs = random.choice(vel_scales)
        thrust_n = random.choice(thrust_list)
        t = {
            "name": f"perturb_r_{int(r):d}_ang_{ang}_vs_{int(10*vs):02d}_{i}",
            "orbit_type": "perturbed",
            "params": {"r": r, "ang_deg": ang, "vel_scale": vs},
            "init_state": {"pos":[r,0.0], "vel_angle_deg": float(ang), "vel_scale": float(vs)},
            "mass": mass,
            "thrust_newton": thrust_n,
            "r_target": r,
            "max_steps": max_steps,
            "seed": seed + 30000 + i
        }
        out.append(t)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory for task JSON files")
    ap.add_argument("--seed", type=int, default=20250820)

    # counts
    ap.add_argument("--num_circular", type=int, default=30)
    ap.add_argument("--num_elliptic", type=int, default=30)
    ap.add_argument("--num_transfer", type=int, default=20)
    ap.add_argument("--num_perturbed", type=int, default=10)

    # physical scales
    ap.add_argument("--mass", type=float, default=5e9, help="spacecraft mass in kg (mega ship)")
    ap.add_argument("--max_steps", type=int, default=60000)

    # radius presets
    ap.add_argument("--radii", nargs="+", type=float,
                    default=[5e11, 1e12, 2e12, 5e12, 7.5e12, 1e13])

    # thrust presets (Newtons) — mix weak and strong to ensure diversity
    ap.add_argument("--thrust_list", nargs="+", type=float,
                    default=[5e6, 1e7, 2e7, 5e7, 1e8])

    # elliptic specifics
    ap.add_argument("--rp_choices", nargs="+", type=float,
                    default=[5e11, 1e12, 1.5e12, 2e12])
    ap.add_argument("--ra_ratio_choices", nargs="+", type=float,
                    default=[1.5, 2.0, 2.5, 3.0])

    args = ap.parse_args()

    out_dir = args.out_dir
    mkdir(out_dir)

    # build sets
    tasks = []
    tasks += circular_tasks(args.num_circular, args.seed, args.radii, args.mass, args.thrust_list, args.max_steps)
    tasks += elliptic_tasks(args.num_elliptic, args.seed, args.rp_choices, args.ra_ratio_choices,
                            args.mass, args.thrust_list, args.max_steps)
    tasks += transfer_tasks(args.num_transfer, args.seed, args.radii, args.mass, args.thrust_list, args.max_steps)
    tasks += perturbed_tasks(args.num_perturbed, args.seed, args.radii, args.mass, args.thrust_list, args.max_steps)

    # write
    for t in tasks:
        fp = os.path.join(out_dir, f"{t['name']}.json")
        with open(fp, "w") as f:
            json.dump(t, f, indent=2)
    print(f"[gen_tasks] wrote {len(tasks)} tasks to {out_dir}")

if __name__ == "__main__":
    main()
