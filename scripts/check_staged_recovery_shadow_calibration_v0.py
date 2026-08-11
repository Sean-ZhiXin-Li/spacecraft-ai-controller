from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_shadow_calibration import (
    CALIBRATION_OUTPUT_PATH,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_TRACE_COUNT,
    TRACE_SET_OUTPUT_PATH,
    calibration_candidates,
    load_and_validate_config,
    trace_definitions,
    validate_calibration_payloads,
    validate_trace_set_payloads,
)


def payloads(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*") if item.is_file()
    }


def validate_static() -> None:
    config = load_and_validate_config(ROOT)
    if len(calibration_candidates(config)) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("candidate grid count mismatch")
    if len(trace_definitions(ROOT)) != EXPECTED_TRACE_COUNT:
        raise RuntimeError("trace definition count mismatch")
    print("CALIBRATION_STATIC_VALIDATION: passed; traces=13; candidates=216; active_authority=false")


def validate_trace_set() -> None:
    validate_trace_set_payloads(payloads(ROOT / TRACE_SET_OUTPUT_PATH))
    print("TRACE_SET_VALIDATION: passed; pairs=13; observed_traces=13; physical_equivalence_failures=0")


def validate_calibration() -> None:
    validate_calibration_payloads(payloads(ROOT / CALIBRATION_OUTPUT_PATH))
    print("CALIBRATION_VALIDATION: passed; candidates=216; offline_replays=2808; physical_executions=0")


def print_selected() -> None:
    value = json.loads((ROOT / CALIBRATION_OUTPUT_PATH / "engineering_candidate.json").read_text("utf-8"))
    for key in sorted(value):
        print(f"{key}={json.dumps(value[key], sort_keys=True, separators=(',', ':'))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Stage 1B-B calibration checker.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-trace-set", action="store_true")
    modes.add_argument("--validate-calibration", action="store_true")
    modes.add_argument("--print-selected-candidate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.validate_static, args.validate_trace_set, args.validate_calibration, args.print_selected_candidate)):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            validate_static()
        elif args.validate_trace_set:
            validate_trace_set()
        elif args.validate_calibration:
            validate_calibration()
        else:
            print_selected()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
