from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.recovery_experiment_preflight import (  # noqa: E402
    preflight_report_lines,
    run_recovery_experiment_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or validate Recovery Action Branching v0 without executing "
            "a simulator transition."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="Print the frozen four-branch plan; never execute or write.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate static contracts and imports; never execute or write.",
    )
    mode.add_argument(
        "--experiment-preflight",
        action="store_true",
        help=(
            "Run the full clean-tree and publication-readiness preflight; "
            "never execute or write."
        ),
    )
    return parser


def _print_plan(contract) -> None:
    print(f"EXPERIMENT_ID {contract.experiment_id}")
    print(f"SOURCE_CASE {contract.case_id}")
    print(f"BRANCH_STATE_HASH {contract.branch_state_hash}")
    print("BRANCHES")
    for branch_id in contract.branch_ids:
        print(f"  {branch_id}")
    print(f"RECOVERY_HORIZON {contract.recovery_horizon}")
    print(f"TOTAL_HORIZON {contract.total_horizon}")
    print("STOP_PRIORITY " + " > ".join(contract.stop_priority))
    print("RESERVED_OUTPUTS")
    for filename in contract.output_filenames:
        print(f"  {contract.output_directory.rstrip('/')}/{filename}")
    print("EXECUTION_DISABLED true")
    print("No simulator transition executed. No artifact written.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.plan or args.validate_only or args.experiment_preflight):
        parser.print_help()
        print("\nNo mode selected. No simulator transition executed. No artifact written.")
        return 2

    require_clean = bool(args.experiment_preflight)
    report, contract = run_recovery_experiment_preflight(
        repository_root=PROJECT_ROOT,
        require_clean_repository=require_clean,
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
    if args.experiment_preflight:
        print("No simulator transition executed. No artifact written.")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
