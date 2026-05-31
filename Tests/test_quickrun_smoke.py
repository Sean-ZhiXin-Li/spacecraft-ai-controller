"""Pytest smoke test that calls quickrun as a subprocess."""
import os
import subprocess
import sys
from pathlib import Path

def test_quickrun_exits_zero():
    cmd = [sys.executable, "scripts/quickrun.py", "--steps", "1200", "--preset", "voyager1"]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH")]))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, "quickrun failed; see stdout for metrics"
