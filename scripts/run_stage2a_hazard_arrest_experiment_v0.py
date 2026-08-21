from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.stage2a_hazard_arrest_runner import (
    EXPERIMENT_OUTPUT_PATH,
    QUALIFICATION_OUTPUT_PATH,
    build_experiment_payloads,
    execute_selected_experiment,
    load_selected_experiment,
    protected_evidence_hashes,
    validate_experiment_payloads,
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
            raise RuntimeError(f"frozen experiment blocked by {label}")


def validate(*, require_clean: bool, require_selected: bool) -> dict[str, object] | None:
    validate_static_sources(ROOT)
    if (ROOT / EXPERIMENT_OUTPUT_PATH).exists():
        raise RuntimeError("experiment output directory already exists")
    selected = None
    if (ROOT / QUALIFICATION_OUTPUT_PATH).exists():
        selected = load_selected_experiment(ROOT)
    elif require_selected:
        raise RuntimeError("frozen qualification result is missing")
    if require_clean:
        _require_clean()
        if not _head():
            raise RuntimeError("committed selection HEAD is required")
    return selected


def execute() -> None:
    selected = validate(require_clean=True, require_selected=True)
    assert selected is not None
    selection_commit = _head()
    protected_before = protected_evidence_hashes(ROOT)
    experiment = execute_selected_experiment(
        ROOT,
        selected,
        implementation_commit=selection_commit,
    )
    protected_after = protected_evidence_hashes(ROOT)
    payloads = build_experiment_payloads(
        experiment,
        implementation_commit=selection_commit,
        selection_commit=selection_commit,
        protected_before=protected_before,
        protected_after=protected_after,
    )
    target = atomic_publish_new_directory(
        ROOT,
        EXPERIMENT_OUTPUT_PATH,
        payloads,
        validate_experiment_payloads,
    )
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print("formal_invocation_count=1")
    print("baseline_bounded_runs=1")
    print("active_bounded_runs=1")
    print("automatic_retry_count=0")
    print(
        "baseline_physical_transitions="
        f"{experiment.baseline_summary['physical_transition_count']}"
    )
    print(
        "active_physical_transitions="
        f"{experiment.active_summary['physical_transition_count']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen Stage 2A one-intervention measured experiment runner."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-experiment", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_experiment)):
        parser.print_help()
        return 0
    try:
        if args.execute_frozen_experiment:
            execute()
        else:
            selected = validate(require_clean=False, require_selected=False)
            if args.plan:
                print(f"qualification_present={str(selected is not None).lower()}")
                print("bounded_runs=2")
                print("maximum_interventions=1")
                print(f"output={EXPERIMENT_OUTPUT_PATH.as_posix()}")
                print("execution=false")
            else:
                print(
                    "STAGE2A_EXPERIMENT_STATIC: passed; execution=false; "
                    f"qualification_present={str(selected is not None).lower()}"
                )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
