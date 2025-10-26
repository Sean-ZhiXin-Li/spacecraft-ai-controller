import os, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    print("[npz] keys:", list(d.files))
    for k in d.files:
        arr = d[k]
        print(f" - {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 1:
            print(f"   stats: min={arr.min():.3e}, max={arr.max():.3e}")
        elif arr.ndim == 2 and arr.shape[1] <= 4:
            print(f"   sample last row: {arr[-1]}")

    if args.meta and os.path.exists(args.meta):
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fields = ["steps_recorded","seed","env_factory","policy","extra"]
        meta_pick = {k: meta.get(k) for k in fields}
        print("[meta]", json.dumps(meta_pick, indent=2))

if __name__ == "__main__":
    main()
