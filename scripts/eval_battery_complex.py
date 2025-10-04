import os, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "csv"), exist_ok=True)

    df = pd.read_csv(args.in_csv)

    # Summary by (orbit_type, controller)
    g = df.groupby(["orbit_type","controller"]).agg(
        success_rate=("success","mean"),
        mean_r_err=("r_err","mean"),
        n=("task_name","count")
    ).reset_index().sort_values(["orbit_type","controller"])
    g.to_csv(os.path.join(args.out_dir, "csv", "summary.csv"), index=False)

    # Hardest 20 for the expert:spiral_in (failed and largest r_err)
    filt = (df["controller"]=="expert:spiral_in") & (df["success"]==0)
    hardest = df[filt].sort_values("r_err", ascending=False).head(20)
    hardest[["task_name","orbit_type","r_err","return"]].to_csv(
        os.path.join(args.out_dir, "csv", "worst_20.csv"), index=False
    )

    print("[eval] summary -> csv/summary.csv")
    print("[eval] worst_20 -> csv/worst_20.csv")

if __name__ == "__main__":
    main()
