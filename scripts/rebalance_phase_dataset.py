from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "analysis" / "phase_controller_dataset"
INPUT_NPZ = DATASET_DIR / "phase_controller_dataset.npz"
INPUT_META = DATASET_DIR / "phase_controller_dataset_metadata.json"
OUTPUT_NPZ = DATASET_DIR / "phase_controller_dataset_balanced.npz"
OUTPUT_META = DATASET_DIR / "phase_controller_dataset_balanced_metadata.json"

PHASE_NAMES = {
    0: "DESCENT",
    1: "CAPTURE",
    2: "LOCK",
}


def main() -> None:
    rng = np.random.default_rng(42)
    data = np.load(INPUT_NPZ)
    phase_ids = data["phase_ids"]
    metadata = json.loads(INPUT_META.read_text(encoding="utf-8"))

    phase_indices = {phase_id: np.flatnonzero(phase_ids == phase_id) for phase_id in PHASE_NAMES}
    target_count = max(len(phase_indices[1]), len(phase_indices[2]))

    selected_indices = []
    rebalance_info = {}
    for phase_id, indices in phase_indices.items():
        if len(indices) >= target_count:
            picked = rng.choice(indices, size=target_count, replace=False)
        else:
            picked = rng.choice(indices, size=target_count, replace=True)
        picked = np.sort(picked)
        selected_indices.append(picked)
        rebalance_info[PHASE_NAMES[phase_id]] = {
            "original_count": int(len(indices)),
            "balanced_count": int(len(picked)),
            "sampling": "downsample" if len(indices) >= target_count else "upsample",
        }

    final_indices = np.sort(np.concatenate(selected_indices))
    balanced = {key: data[key][final_indices] for key in data.files}
    np.savez_compressed(OUTPUT_NPZ, **balanced)

    output_meta = {
        "source_dataset": INPUT_NPZ.as_posix(),
        "balanced_dataset": OUTPUT_NPZ.as_posix(),
        "target_count_per_phase": int(target_count),
        "phase_stats": rebalance_info,
        "source_setups": metadata,
        "total_samples": int(len(final_indices)),
        "rng_seed": 42,
    }
    OUTPUT_META.write_text(json.dumps(output_meta, indent=2), encoding="utf-8")

    print(json.dumps(output_meta, indent=2))
    print(f"Saved balanced dataset to: {OUTPUT_NPZ}")
    print(f"Saved balanced metadata to: {OUTPUT_META}")


if __name__ == "__main__":
    main()
