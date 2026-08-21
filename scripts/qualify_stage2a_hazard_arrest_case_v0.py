from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.stage2a_hazard_arrest_runner import (
    QUALIFICATION_OUTPUT_PATH,
    build_qualification_payloads,
    validate_qualification_payloads,
    validate_static_sources,
)
from runtime_assurance.staged_recovery_shadow_calibration import (
    atomic_publish_new_directory,
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _require_clean() -> None:
    for command, label in (
        (["git", "diff", "--quiet"], "tracked changes"),
        (["git", "diff", "--cached", "--quiet"], "staged changes"),
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            raise RuntimeError(f"qualification blocked by {label}")


def validate(*, require_clean: bool, require_output_absent: bool) -> dict[str, object]:
    report = validate_static_sources(ROOT)
    if require_output_absent and (ROOT / QUALIFICATION_OUTPUT_PATH).exists():
        raise RuntimeError("qualification output directory already exists")
    if require_clean:
        _require_clean()
        if not _head():
            raise RuntimeError("committed implementation HEAD is required")
    return report


def execute() -> None:
    validate(require_clean=True, require_output_absent=True)
    payloads = build_qualification_payloads(ROOT, implementation_commit=_head())
    target = atomic_publish_new_directory(
        ROOT,
        QUALIFICATION_OUTPUT_PATH,
        payloads,
        validate_qualification_payloads,
    )
    manifest = json.loads(payloads["qualification_manifest.json"])
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print(f"states_inspected={manifest['states_inspected']}")
    print(f"offline_prediction_evaluations={manifest['offline_prediction_evaluations']}")
    print(f"eligible_boundary_count={manifest['eligible_boundary_count']}")
    print(f"selection_status={manifest['selection_status']}")
    print("physical_executions=0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Stage 2A frozen-case qualification."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-qualification", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_qualification)):
        parser.print_help()
        return 0
    try:
        if args.execute_frozen_qualification:
            execute()
        else:
            report = validate(
                require_clean=False,
                require_output_absent=not (ROOT / QUALIFICATION_OUTPUT_PATH).exists(),
            )
            if args.plan:
                print("source_trace_count=13")
                print("normal_branch_ids=zero_action_reference_v0,tangential_error_correction_v0")
                print("hazard_action_source=velocity_opposed_thrust_v0")
                print("selection_order=registry_member_id,source_trace_id,prefix_transition_count,normal_branch_id")
                print(f"output={QUALIFICATION_OUTPUT_PATH.as_posix()}")
                print("physical_executions=0")
            else:
                print(
                    "STAGE2A_QUALIFICATION_STATIC: passed; "
                    f"traces={report['source_trace_count']}; physical_executions=0"
                )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
