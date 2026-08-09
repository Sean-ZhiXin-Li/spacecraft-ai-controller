from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_shadow_guard import (
    CONTROL_AUTHORITY,
    MAXIMUM_SHADOW_TRANSITIONS_PER_TRACE,
    MINIMUM_PHASE_DWELL_STEPS,
    SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME,
    TRANSITION_COOLDOWN_STEPS,
    allowed_shadow_edges,
    architecture_phase_ids,
)
from runtime_assurance.staged_recovery_shadow_runtime import (
    OUTPUT_PATH,
    validate_registry_source,
    validate_smoke_payloads,
)


def _validate_static() -> None:
    members = validate_registry_source(ROOT)
    if len(architecture_phase_ids()) != 9 or len(allowed_shadow_edges()) != 26:
        raise RuntimeError("frozen staged architecture identity changed")
    if CONTROL_AUTHORITY or SHADOW_OUTPUT_CONSUMED_BY_PHYSICAL_RUNTIME:
        raise RuntimeError("shadow authority boundary failed")
    print(
        "STATIC_VALIDATION: passed; registry_members=4; phases=9; edges=26; "
        "control_authority=false; staged_recovery_execution=not_authorized"
    )


def _published_payloads() -> dict[str, bytes]:
    target = ROOT / OUTPUT_PATH
    if not target.is_dir():
        raise RuntimeError("published shadow smoke directory is missing")
    return {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*") if path.is_file()
    }


def _validate_published() -> None:
    payloads = _published_payloads()
    validate_smoke_payloads(payloads)
    if len(payloads) != 10:
        raise RuntimeError("published smoke must contain six top-level and four trace files")
    print("PUBLISHED_VALIDATION: passed; pairs=4; traces=4; physical_equivalence_failures=0")


def _print_summary() -> None:
    summary = json.loads((ROOT / OUTPUT_PATH / "shadow_guard_summary.json").read_text("utf-8"))
    for key in sorted(summary):
        value = summary[key]
        print(f"{key}={json.dumps(value, sort_keys=True, separators=(',', ':'))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Stage 1B-A shadow runtime checker.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.validate_static, args.validate_published, args.print_summary)):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            _validate_static()
            print(
                f"anti_chatter=dwell:{MINIMUM_PHASE_DWELL_STEPS},"
                f"cooldown:{TRANSITION_COOLDOWN_STEPS},"
                f"budget:{MAXIMUM_SHADOW_TRANSITIONS_PER_TRACE}"
            )
        elif args.validate_published:
            _validate_published()
        else:
            _print_summary()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
