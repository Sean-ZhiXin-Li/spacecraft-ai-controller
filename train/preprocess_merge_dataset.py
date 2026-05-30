import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "data" / "preprocessed"


def main():
    file_paths = sorted(DATASET_DIR.glob("expert_dataset_*.npy"))
    print("Matched files:", file_paths)

    # Read and merge
    all_data = []
    for path in file_paths:
        arr = np.load(path)
        all_data.append(arr)
    all_data = np.concatenate(all_data, axis=0)

    # Save as a large file
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREPROCESSED_DIR / "merged_expert_dataset.npy", all_data)

    print("Merged dataset saved.")


if __name__ == "__main__":
    main()
