from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.recovery_experiment_preflight import (  # noqa: E402
    BRANCH_STATE_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    preflight_report_lines,
    run_recovery_experiment_preflight,
)
from runtime_assurance.recovery_experiment_runner import (  # noqa: E402
    RecoveryExperimentExecutionError,
    run_frozen_recovery_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, validate, preflight, or explicitly execute the single frozen "
            "Recovery Action Branching Nonformal v0 experiment."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="Print the frozen experiment plan without executing or writing.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate static contracts without executing or writing.",
    )
    mode.add_argument(
        "--experiment-preflight",
        action="store_true",
        help="Require clean-tree execution readiness without executing or writing.",
    )
    mode.add_argument(
        "--execute-frozen-experiment",
        action="store_true",
        help=(
            "Execute all frozen branches exactly once and atomically publish the "
            "complete nonformal bundle. No scientific overrides are accepted."
        ),
    )
    return parser


def _print_plan(contract) -> None:
    print(f"EXPERIMENT_ID {contract.experiment_id}")
    print(f"SOURCE_CASE {contract.case_id}")
    print(f"BRANCH_STATE_HASH {contract.branch_state_hash}")
    print("BRANCH_ORDER " + " | ".join(contract.branch_ids))
    print(f"RECOVERY_HORIZON {contract.recovery_horizon}")
    print(f"TOTAL_HORIZON {contract.total_horizon}")
    print(f"HAZARD_COMPARATOR {contract.hazard_comparator} {contract.hazard_threshold}")
    print("STOP_PRIORITY " + " > ".join(contract.stop_priority))
    print("RESERVED_OUTPUTS " + " | ".join(contract.output_filenames))
    print("EXECUTION_DISABLED true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = (
        args.plan
        or args.validate_only
        or args.experiment_preflight
        or args.execute_frozen_experiment
    )
    if not selected:
        parser.print_help()
        print("\nNo mode selected. No recovery transition executed. No artifact written.")
        return 2

    if args.execute_frozen_experiment:
        try:
            result = run_frozen_recovery_experiment(
                manifest_path=PROJECT_ROOT / MANIFEST_RELATIVE_PATH,
                branch_state_path=PROJECT_ROOT / BRANCH_STATE_RELATIVE_PATH,
                output_directory=(
                    PROJECT_ROOT / "analysis/recovery_action_branching_nonformal_v0"
                ),
            )
        except RecoveryExperimentExecutionError as exc:
            print(f"EXECUTION_ABORTED {exc}", file=sys.stderr)
            print(f"BRANCH {exc.branch_id or 'not_started'}", file=sys.stderr)
            print(
                f"ATTEMPTED_RECOVERY_STEP "
                f"{exc.attempted_recovery_step if exc.attempted_recovery_step is not None else 'none'}",
                file=sys.stderr,
            )
            print(
                f"REALIZED_TRANSITIONS {exc.realized_transitions}",
                file=sys.stderr,
            )
            print("No automatic retry performed.", file=sys.stderr)
            return 1
        print("EXECUTION_COMPLETE true")
        print(f"IMPLEMENTATION_COMMIT {result.preflight_report.implementation_commit}")
        print(f"BRANCH_RECORDS {len(result.bundle.records)}")
        print(f"DECISION_EVENTS {len(result.bundle.decision_events)}")
        for record in result.bundle.records:
            print(
                "BRANCH_RESULT "
                f"{record.branch_id} terminal={record.terminal_reason} "
                f"transitions={record.recovery_transition_count} "
                f"overspeed={record.overspeed_status} "
                f"crossing={record.crossed_target_radius} "
                f"recoverable={record.phase34_compatible_recoverable_crossing} "
                f"recovery_success={record.recovery_success}"
            )
        for filename, digest in result.publication.artifact_hashes:
            print(f"ARTIFACT_HASH {filename} {digest}")
        print("No retry performed.")
        return 0

    report, contract = run_recovery_experiment_preflight(
        repository_root=PROJECT_ROOT,
        require_clean_repository=bool(args.experiment_preflight),
    )
    if args.plan:
        if contract is None:
            for line in preflight_report_lines(report):
                print(line)
            return 1
        _print_plan(contract)
        return 0 if report.ready else 1
    for line in preflight_report_lines(report):
        print(line)
    print("No recovery transition executed. No artifact written.")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
