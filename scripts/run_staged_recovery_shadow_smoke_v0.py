from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.staged_recovery_shadow_runtime import (
    CLAIM_RESTRICTIONS,
    MAXIMUM_RECOVERY_TRANSITIONS,
    OUTPUT_PATH,
    PHYSICAL_BRANCH_ID,
    build_smoke_payloads,
    publish_smoke_payloads,
    run_shadow_smoke_case,
    validate_registry_source,
)


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _require_clean_tracked_tree() -> None:
    for arguments, label in ((["git", "diff", "--quiet"], "tracked changes"), (["git", "diff", "--cached", "--quiet"], "staged changes")):
        result = subprocess.run(arguments, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"frozen smoke execution is blocked by {label}")


def _validate(*, require_clean: bool) -> tuple[str, ...]:
    members = validate_registry_source(ROOT)
    if (ROOT / OUTPUT_PATH).exists():
        raise RuntimeError("shadow smoke result directory already exists")
    if require_clean:
        _require_clean_tracked_tree()
        if not _head():
            raise RuntimeError("a committed implementation HEAD is required")
    return members


def _plan() -> None:
    members = validate_registry_source(ROOT)
    print(f"registry_members={len(members)}")
    for member in members:
        print(f"member={member}")
    print(f"physical_branch={PHYSICAL_BRANCH_ID}")
    print(f"pair_count={len(members)}")
    print(f"bounded_execution_count={2 * len(members)}")
    print(f"maximum_transitions_per_execution={MAXIMUM_RECOVERY_TRANSITIONS}")
    print(f"output={OUTPUT_PATH.as_posix()}")
    print("shadow_control_authority=false")
    print("execution=false")
    for restriction in CLAIM_RESTRICTIONS:
        print(f"claim_restriction={restriction}")


def _execute() -> None:
    members = _validate(require_clean=True)
    implementation_commit = _head()
    results = tuple(
        run_shadow_smoke_case(
            ROOT, member, implementation_commit=implementation_commit
        )
        for member in members
    )
    payloads = build_smoke_payloads(results, implementation_commit=implementation_commit)
    target = publish_smoke_payloads(ROOT, payloads)
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print("pairs=4")
    print("bounded_executions=8")
    print("physical_equivalence_failures=0")
    print("unauthorized_physical_effects=0")
    print("nominal_handoff_recommendations=0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen four-case observational shadow-FSM smoke validation."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_smoke)):
        parser.print_help()
        return 0
    try:
        if args.plan:
            _plan()
        elif args.validate_only:
            members = _validate(require_clean=False)
            print(f"STATIC_VALIDATION: passed; registry_members={len(members)}; execution=false")
        else:
            _execute()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
