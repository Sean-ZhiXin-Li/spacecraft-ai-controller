from dataclasses import dataclass
from typing import List, Dict, Any
import json, glob, os, random

@dataclass
class TaskSpec:
    """Typed container for a single task specification."""
    name: str
    orbit_type: str
    params: Dict[str, Any]
    init_state: Dict[str, Any]
    mass: float
    thrust_newton: float
    r_target: float
    max_steps: int
    seed: int

class TaskSampler:
    """
    Load all task JSON files from a directory and provide a .sample() API
    to return a single TaskSpec each time (random or sequential).
    """
    def __init__(self, dir_path: str, mode: str = "random"):
        """
        Args:
            dir_path: directory where *.json task files are stored.
            mode: "random" or "sequential".
        """
        self.dir_path = dir_path
        self.paths: List[str] = sorted(glob.glob(os.path.join(dir_path, "*.json")))
        if not self.paths:
            raise FileNotFoundError(f"No task JSONs found in: {dir_path}")
        self.mode = mode
        self._idx = 0

    def _load(self, path: str) -> TaskSpec:
        with open(path, "r") as f:
            data = json.load(f)
        # Minimal validation + defaults
        req = ["name","orbit_type","params","init_state","mass","thrust_newton","r_target","max_steps","seed"]
        for k in req:
            if k not in data:
                raise KeyError(f"Task JSON missing key: {k} in {path}")
        return TaskSpec(**data)

    def sample(self) -> TaskSpec:
        """Return one TaskSpec according to the sampling mode."""
        if self.mode == "random":
            path = random.choice(self.paths)
        else:
            path = self.paths[self._idx % len(self.paths)]
            self._idx += 1
        return self._load(path)
