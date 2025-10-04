"""Pytest smoke test that calls quickrun as a subprocess."""
import subprocess, sys

def test_quickrun_exits_zero():
    cmd = [sys.executable, "scripts/quickrun.py", "--steps", "1200", "--preset", "voyager1"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, "quickrun failed; see stdout for metrics"
