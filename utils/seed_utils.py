"""Deterministic seeding helpers."""
import os, random
from typing import Optional

def set_global_seed(seed: int, use_torch: bool = True) -> None:
    """Set seeds for python, numpy, torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            # Torch not installed or CUDA unavailable; skip silently.
            pass

def get_seed_from_env(default_seed: int = 51) -> int:
    """Pick seed from env if provided, else default."""
    val = os.environ.get("RUN_SEED")
    try:
        return int(val) if val is not None else default_seed
    except Exception:
        return default_seed
