import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_run(path):
    data = np.load(path)

    print(f"\nLoading: {path}")
    print("Keys:", data.files)

    result = {}

    # Directly load available metrics
    result["r"] = data["r"]
    result["v_r"] = data["vr"]
    result["cos_tr"] = data["cos_tr"]
    result["cos_tt"] = data["cos_tt"]

    if "target_r" in data.files:
        result["target_r"] = data["target_r"]

    result["time"] = np.arange(len(result["r"]))

    return result


def plot_radial_velocity(all_results, save_path):
    plt.figure(figsize=(8, 5))

    for name, result in all_results.items():
        t = result["time"]
        v_r = result["v_r"]
        plt.plot(t, v_r, label=name)

    plt.xlabel("Time step")
    plt.ylabel("Radial velocity $v_r$")
    plt.title("Radial Velocity vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cos_tr_vs_vr(all_results, save_path):
    plt.figure(figsize=(8, 5))

    for name, result in all_results.items():
        cos_tr = result["cos_tr"]
        v_r = result["v_r"]
        plt.scatter(cos_tr, v_r, s=10, alpha=0.5, label=name)

    plt.xlabel("cos(thrust, radial)")
    plt.ylabel("Radial velocity $v_r$")
    plt.title("cos_tr vs v_r")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def summarize_runs(all_results):
    print("\n=== Day 7 Summary ===")

    for name, result in all_results.items():
        mean_vr = np.mean(result["v_r"])
        min_vr = np.min(result["v_r"])
        mean_cos_tr = np.mean(result["cos_tr"])
        mean_cos_tt = np.mean(result["cos_tt"])

        print(f"\n[{name}]")
        print(f"mean_vr     = {mean_vr:.6e}")
        print(f"min_vr      = {min_vr:.6e}")
        print(f"mean_cos_tr = {mean_cos_tr:.6e}")
        print(f"mean_cos_tt = {mean_cos_tt:.6e}")

def plot_radius_vs_time(all_results, save_path):
    plt.figure(figsize=(8, 5))

    for name, result in all_results.items():
        t = result["time"]
        r = result["r"]
        plt.plot(t, r, label=name)

        if "target_r" in result:
            target_r = result["target_r"]
            if np.isscalar(target_r) or np.ndim(target_r) == 0:
                plt.axhline(float(target_r), linestyle="--", alpha=0.6)
            else:
                plt.plot(t, target_r, linestyle="--", alpha=0.6)

    plt.xlabel("Time step")
    plt.ylabel("Radius r")
    plt.title("Radius vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def summarize_radius_error(all_results):
    print("\n=== Radius Error Summary ===")
    for name, result in all_results.items():
        if "target_r" not in result:
            continue

        r = result["r"]
        target_r = result["target_r"]

        if np.isscalar(target_r) or np.ndim(target_r) == 0:
            target = float(target_r)
            err = r - target
        else:
            target = target_r
            err = r - target_r

        mean_abs_err = np.mean(np.abs(err))
        final_abs_err = np.abs(err[-1])
        max_abs_err = np.max(np.abs(err))

        mean_rel_err = np.mean(np.abs(err / target))
        final_rel_err = np.abs(err[-1] / target[-1] if np.ndim(target) > 0 else err[-1] / target)

        print(f"\n[{name}]")
        print(f"mean_abs_r_err   = {mean_abs_err:.6e}")
        print(f"final_abs_r_err  = {final_abs_err:.6e}")
        print(f"max_abs_r_err    = {max_abs_err:.6e}")
        print(f"mean_rel_r_err   = {mean_rel_err:.6e}")
        print(f"final_rel_r_err  = {final_rel_err:.6e}")

def plot_radius_error_vs_time(all_results, save_path):
    plt.figure(figsize=(8, 5))

    for name, result in all_results.items():
        if "target_r" not in result:
            continue

        t = result["time"]
        r = result["r"]
        target_r = result["target_r"]

        if np.isscalar(target_r) or np.ndim(target_r) == 0:
            err = r - float(target_r)
        else:
            err = r - target_r

        plt.plot(t, err, label=name)

    plt.xlabel("Time step")
    plt.ylabel("Radius error (r - target_r)")
    plt.title("Radius Error vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_relative_radius_error_vs_time(all_results, save_path):
    plt.figure(figsize=(8, 5))

    for name, result in all_results.items():
        if "target_r" not in result:
            continue

        t = result["time"]
        r = result["r"]
        target_r = result["target_r"]

        if np.isscalar(target_r) or np.ndim(target_r) == 0:
            target = float(target_r)
            rel_err = (r - target) / target
        else:
            rel_err = (r - target_r) / target_r

        plt.plot(t, rel_err, label=name)

    plt.xlabel("Time step")
    plt.ylabel("Relative radius error")
    plt.title("Relative Radius Error vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    runs = {
            "ppo": "analysis/runs/run_964254/traj.npz",
            "gated": "analysis/runs/run_854705/traj.npz",
            "always_on": "analysis/runs/run_835152/traj.npz",
    }

    all_results = {}

    for name, path in runs.items():
        rollout = load_run(path)
        all_results[name] = rollout

    out_dir = Path("analysis/figs/day7_mechanism")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_radial_velocity(
        all_results,
        out_dir / "radial_velocity_vs_time.png",
    )

    plot_cos_tr_vs_vr(
        all_results,
        out_dir / "cos_tr_vs_vr.png",
    )

    plot_radius_vs_time(
        all_results,
        out_dir / "radius_vs_time.png",
    )

    plot_radius_error_vs_time(
        all_results,
        out_dir / "radius_error_vs_time.png",
    )

    plot_relative_radius_error_vs_time(
        all_results,
        out_dir / "relative_radius_error_vs_time.png",
    )

    print("\nSaved plots:")
    print("analysis/figs/day7_mechanism/radial_velocity_vs_time.png")
    print("analysis/figs/day7_mechanism/cos_tr_vs_vr.png")
    print("analysis/figs/day7_mechanism/radius_vs_time.png")
    print("analysis/figs/day7_mechanism/radius_error_vs_time.png")
    print("analysis/figs/day7_mechanism/relative_radius_error_vs_time.png")
    summarize_runs(all_results)
    summarize_radius_error(all_results)



if __name__ == "__main__":
    main()

