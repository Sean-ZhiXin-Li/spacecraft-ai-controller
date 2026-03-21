from pathlib import Path
import subprocess
import os

CONTROLLERS = ["always_on", "gated"]
R0_LEVELS = [1.01]
FIXED_THRUST = 100000

RESULTS_DIR = Path("analysis/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = "analysis/results/week03_r0_sweep_summary.csv"


def run_single_config(controller: str, thrust: float, r0_over_target: float) -> dict:
    """
    Run one experiment configuration and return one summary row.
    """

    env = os.environ.copy()
    env["THRUST_NEWTON"] = str(thrust)
    env["R0_OVER_TARGET"] = str(r0_over_target)
    env["CONTROLLER_VARIANT"] = controller
    env["ABLATION_CSV"] = OUTPUT_CSV

    print(f"[RUN] controller={controller}, thrust={thrust}, r0={r0_over_target}")

    cmd = [
        "python",
        "src/quick_compare_v3_v4.py",
    ]

    print(f"[CMD] {' '.join(cmd)}")

    completed = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )

    print(f"[RETURN CODE] {completed.returncode}")

    if completed.stdout:
        print("[STDOUT]")
        print(completed.stdout)

    if completed.stderr:
        print("[STDERR]")
        print(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for controller={controller}, thrust={thrust}, r0={r0_over_target}"
        )


def main():
    # Optional: start fresh each time for W03
    out_path = Path(OUTPUT_CSV)
    if out_path.exists():
        out_path.unlink()
        print(f"[RESET] removed old file: {out_path}")

    for controller in CONTROLLERS:
        for r0 in R0_LEVELS:
            run_single_config(
                controller=controller,
                thrust=FIXED_THRUST,
                r0_over_target=r0
            )

    print(f"[DONE] results written by runner to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
