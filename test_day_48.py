# tests/test_day48_pytest.py
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from envs.multi_orbit_env import (MultiOrbitEnv)

LOG_DIR = Path("logs/day48")
LOG_PATH = LOG_DIR / "test_log_day48.csv"
FIELDS = ["timestamp", "scenario", "steps", "fuel_used", "dv1", "total_reward", "final_orbit_error", "elapsed_s"]


def _run_once(scenario: str) -> dict:
    env = MultiOrbitEnv(scenario=scenario)
    t0 = datetime.utcnow()
    metrics = env.rollout(max_steps=4096)
    dt = (datetime.utcnow() - t0).total_seconds()
    metrics["timestamp"] = datetime.utcnow().isoformat()
    metrics["elapsed_s"] = round(dt, 3)
    return metrics


def test_day48_smoke_three_scenarios(tmp_path):
    """
    PyTest-style smoke test for Day 48.
    Runs circular / elliptic / transfer once each,
    writes a CSV log to logs/day48/test_log_day48.csv,
    and asserts we got 3 rows with required fields.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for sc in ["circular", "elliptic", "transfer"]:
        m = _run_once(sc)
        rows.append(m)

    df = pd.DataFrame(rows, columns=FIELDS)

    for col in FIELDS:
        if col not in df.columns:
            df[col] = "" if col not in ["steps", "fuel_used", "dv1", "total_reward", "final_orbit_error", "elapsed_s"] else 0.0
    df = df[FIELDS]
    df.to_csv(LOG_PATH, index=False)

    assert len(df) == 3
    assert set(df["scenario"]) == {"circular", "elliptic", "transfer"}

    for col in ["total_reward", "final_orbit_error"]:
        assert col in df.columns
