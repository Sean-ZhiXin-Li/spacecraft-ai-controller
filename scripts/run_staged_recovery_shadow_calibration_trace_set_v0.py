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
    EXPECTED_PHYSICAL_EXECUTION_COUNT,
    EXPECTED_TRACE_COUNT,
    TRACE_SET_OUTPUT_PATH,
    atomic_publish_new_directory,
    build_trace_set_payloads,
    capture_trace_pair,
    load_and_validate_config,
    trace_definitions,
    validate_trace_set_payloads,
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _require_clean() -> None:
    for command, label in ((["git", "diff", "--quiet"], "tracked changes"), (["git", "diff", "--cached", "--quiet"], "staged changes")):
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise RuntimeError(f"frozen trace-set execution blocked by {label}")


def validate(*, require_clean: bool) -> tuple[object, ...]:
    load_and_validate_config(ROOT)
    definitions = trace_definitions(ROOT)
    if len(definitions) != EXPECTED_TRACE_COUNT:
        raise RuntimeError("trace definition count mismatch")
    if (ROOT / TRACE_SET_OUTPUT_PATH).exists():
        raise RuntimeError("trace-set result directory already exists")
    if require_clean:
        _require_clean()
        if not _head():
            raise RuntimeError("committed implementation HEAD is required")
    return definitions


def execute() -> None:
    definitions = validate(require_clean=True)
    implementation_commit = _head()
    traces = tuple(
        capture_trace_pair(ROOT, definition, implementation_commit=implementation_commit)
        for definition in definitions
    )
    payloads = build_trace_set_payloads(
        ROOT, traces, implementation_commit=implementation_commit
    )
    target = atomic_publish_new_directory(
        ROOT, TRACE_SET_OUTPUT_PATH, payloads, validate_trace_set_payloads
    )
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print(f"trace_pairs={EXPECTED_TRACE_COUNT}")
    print(f"bounded_executions={EXPECTED_PHYSICAL_EXECUTION_COUNT}")
    print("physical_equivalence_failures=0")
    print("unauthorized_physical_effects=0")
    print("automatic_retry=false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Stage 1B-B 13-pair shadow trace-set runner.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-trace-set", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_trace_set)):
        parser.print_help()
        return 0
    try:
        definitions = validate(require_clean=False) if not args.execute_frozen_trace_set else None
        if args.plan:
            for item in definitions:
                print(f"trace={item.trace_id}")
            print("pairs=13")
            print("bounded_executions=26")
            print("maximum_transitions=32")
            print("execution=false")
        elif args.validate_only:
            print("TRACE_SET_STATIC_VALIDATION: passed; definitions=13; execution=false")
        else:
            execute()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
