from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_shadow_calibration import (
    CALIBRATION_OUTPUT_PATH,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_OFFLINE_REPLAY_COUNT,
    TRACE_SET_OUTPUT_PATH,
    analyze_candidates,
    atomic_publish_new_directory,
    build_calibration_payloads,
    calibration_candidates,
    load_and_validate_config,
    load_trace_set,
    validate_calibration_payloads,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _require_clean() -> None:
    for command, label in ((["git", "diff", "--quiet"], "tracked changes"), (["git", "diff", "--cached", "--quiet"], "staged changes")):
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise RuntimeError(f"frozen calibration blocked by {label}")


def validate(*, require_trace_set: bool, require_clean: bool) -> dict[str, object]:
    config = load_and_validate_config(ROOT)
    if len(calibration_candidates(config)) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("candidate grid count mismatch")
    if require_trace_set:
        load_trace_set(ROOT)
    if (ROOT / CALIBRATION_OUTPUT_PATH).exists():
        raise RuntimeError("calibration result directory already exists")
    if require_clean:
        _require_clean()
        if not _git("rev-parse", "HEAD"):
            raise RuntimeError("committed trace-set HEAD is required")
    return config


def execute() -> None:
    config = validate(require_trace_set=True, require_clean=True)
    trace_set_commit = _git("rev-parse", "HEAD")
    implementation_commit = _git("rev-parse", "HEAD~1")
    index, rows = load_trace_set(ROOT)
    results, selected = analyze_candidates(config, index, rows)
    payloads = build_calibration_payloads(
        ROOT, config, index, rows, results, selected,
        implementation_commit=implementation_commit,
        trace_set_commit=trace_set_commit,
    )
    target = atomic_publish_new_directory(
        ROOT, CALIBRATION_OUTPUT_PATH, payloads, validate_calibration_payloads
    )
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print(f"candidate_count={EXPECTED_CANDIDATE_COUNT}")
    print(f"offline_replay_count={EXPECTED_OFFLINE_REPLAY_COUNT}")
    print("physical_executions=0")
    print(f"selected_candidate={selected.candidate.candidate_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure offline Stage 1B-B shadow calibration analyzer.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-calibration", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_calibration)):
        parser.print_help()
        return 0
    try:
        if args.plan:
            config = validate(require_trace_set=False, require_clean=False)
            print(f"candidate_count={len(calibration_candidates(config))}")
            print("trace_count=13")
            print("offline_replay_count=2808")
            print("physical_execution=false")
        elif args.validate_only:
            require_trace = (ROOT / TRACE_SET_OUTPUT_PATH).exists()
            validate(require_trace_set=require_trace, require_clean=False)
            print(f"CALIBRATION_STATIC_VALIDATION: passed; trace_set_available={str(require_trace).lower()}; execution=false")
        else:
            execute()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
