from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_assurance.recovery_branch_executor import (  # noqa: E402
    DEFAULT_BRANCH_STATE_PATH,
    SUPPORTED_BRANCH_IDS,
    RecoveryBranchExecutionResult,
    RecoveryBranchExecutorError,
    execute_recovery_branch,
    load_frozen_branch_state,
)
from runtime_assurance.recovery_stop_conditions import (  # noqa: E402
    INVALID_RECOVERY_EVALUATION,
    NOT_EVALUATED,
    RECOVERY_SUCCESS,
    RecoveryStopConditionReport,
    evaluate_recovery_stop_conditions,
)
from simulator.phase34_35_transition import CartesianState2D  # noqa: E402


RUNNER_RECORD_SCHEMA_VERSION = "recovery_branch_runner_record_v0"
MAX_RUNNER_HORIZON_STEPS = 32
DIAGNOSTIC_LOG_NAME = "recovery_branch_diagnostics.jsonl"

_PROTECTED_OUTPUT_DIRECTORIES = (
    PROJECT_ROOT / "analysis" / "final_veto_ablation_v0",
    PROJECT_ROOT / "analysis" / "phase34_post_cross_sync",
    PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion",
    PROJECT_ROOT / "analysis" / "phase36b_transfer_family_benchmark",
    PROJECT_ROOT / "analysis" / "phase36c_non_crossing_geometry_diagnosis",
    PROJECT_ROOT / "analysis" / "phase37a_radial_commit_timing",
    PROJECT_ROOT / "analysis" / "phase37b_weak_tangential_subset",
    PROJECT_ROOT / "analysis" / "recovery_action_branching_nonformal_v0",
)


class RecoveryBranchRunnerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryBranchRunnerRecord:
    branch_id: str
    step: int
    state_hash: str
    action: tuple[float, float] | None
    transition_executed: bool
    terminal_reason: str
    valid: bool


@dataclass(frozen=True, slots=True)
class RecoveryBranchRunResult:
    branch_id: str
    records: tuple[RecoveryBranchRunnerRecord, ...]
    terminal_reason: str
    transition_count: int
    valid: bool
    recovery_success_status: str
    diagnostic_log_path: Path | None


def _validate_horizon(horizon_steps: object) -> int:
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps < 1
        or horizon_steps > MAX_RUNNER_HORIZON_STEPS
    ):
        raise RecoveryBranchRunnerError(
            "horizon_steps must be an integer from 1 through "
            f"{MAX_RUNNER_HORIZON_STEPS}; long recovery horizons are disabled"
        )
    return horizon_steps


def _validate_branch_id(branch_id: str) -> None:
    if branch_id not in SUPPORTED_BRANCH_IDS:
        raise RecoveryBranchRunnerError(f"unsupported recovery branch: {branch_id!r}")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_output_directory(output_dir: str | Path) -> Path:
    resolved = Path(output_dir).resolve()
    for protected_directory in _PROTECTED_OUTPUT_DIRECTORIES:
        if _is_within(resolved, protected_directory.resolve()):
            raise RecoveryBranchRunnerError(
                f"output directory overlaps protected evidence: {protected_directory}"
            )
    return resolved


def build_recovery_branch_plan(
    branch_id: str,
    *,
    branch_state_path: str | Path,
    horizon_steps: int,
    output_dir: str | Path | None,
) -> dict[str, object]:
    _validate_branch_id(branch_id)
    checked_horizon = _validate_horizon(horizon_steps)
    branch_state = load_frozen_branch_state(branch_state_path)
    checked_output = (
        validate_output_directory(output_dir) if output_dir is not None else None
    )
    return {
        "branch_id": branch_id,
        "branch_state_hash": branch_state["canonical_branch_state_hash"],
        "execution_authorized": False,
        "horizon_steps": checked_horizon,
        "is_formal_experiment": False,
        "output_dir": str(checked_output) if checked_output is not None else None,
        "recovery_success_evaluation": NOT_EVALUATED,
    }


def _realized_speed_ratio(
    state: CartesianState2D | None,
    branch_state: Mapping[str, object],
) -> float | None:
    if state is None:
        return None
    configuration = branch_state["simulator_configuration"]
    constants = configuration["simulator_constants"]
    target_speed = float(constants["target_circular_speed"])
    epsilon = float(constants["speed_ratio_denominator_epsilon"])
    speed = math.hypot(state.vx, state.vy)
    return speed / (target_speed + epsilon)


def _runner_record(
    execution: RecoveryBranchExecutionResult,
    *,
    step: int,
    stop_report: RecoveryStopConditionReport,
) -> RecoveryBranchRunnerRecord:
    state_hash = execution.next_state_hash or execution.previous_state_hash
    return RecoveryBranchRunnerRecord(
        branch_id=execution.branch_id,
        step=step,
        state_hash=state_hash,
        action=execution.action,
        transition_executed=execution.executed,
        terminal_reason=stop_report.terminal_reason or "continue",
        valid=execution.valid,
    )


def _record_payload(record: RecoveryBranchRunnerRecord) -> dict[str, object]:
    return {
        "action": list(record.action) if record.action is not None else None,
        "branch_id": record.branch_id,
        "is_formal_experiment": False,
        "record_schema_version": RUNNER_RECORD_SCHEMA_VERSION,
        "state_hash": record.state_hash,
        "step": record.step,
        "terminal_reason": record.terminal_reason,
        "transition_executed": record.transition_executed,
        "valid": record.valid,
    }


def write_diagnostic_log(
    output_dir: str | Path,
    records: tuple[RecoveryBranchRunnerRecord, ...],
) -> Path:
    checked_output = validate_output_directory(output_dir)
    checked_output.mkdir(parents=True, exist_ok=True)
    output_path = checked_output / DIAGNOSTIC_LOG_NAME
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic log: {output_path}")
    temporary_path = checked_output / f".{DIAGNOSTIC_LOG_NAME}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        _record_payload(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def run_recovery_branch(
    branch_id: str,
    *,
    branch_state_path: str | Path,
    horizon_steps: int,
    output_dir: str | Path,
    write_diagnostics: bool = False,
) -> RecoveryBranchRunResult:
    _validate_branch_id(branch_id)
    checked_horizon = _validate_horizon(horizon_steps)
    checked_output = validate_output_directory(output_dir)
    branch_state = load_frozen_branch_state(branch_state_path)
    branch_ordering = branch_state["branch_ordering"]
    prefix_transitions = int(branch_ordering["realized_prefix_transition_count"])
    simulator_constants = branch_state["simulator_configuration"][
        "simulator_constants"
    ]
    total_horizon = int(simulator_constants["max_steps"])
    overspeed_threshold = float(branch_state["threshold"])

    records: list[RecoveryBranchRunnerRecord] = []
    current_state: CartesianState2D | None = None
    recovery_transition_count = 0
    terminal_reason: str | None = None
    valid = True
    final_stop_report: RecoveryStopConditionReport | None = None

    for step in range(1, checked_horizon + 1):
        try:
            execution = execute_recovery_branch(
                branch_state,
                branch_id,
                horizon_steps=1,
                current_state=current_state,
            )
        except RecoveryBranchExecutorError as exc:
            raise RecoveryBranchRunnerError(
                f"recovery execution failed at step {step}: {exc}"
            ) from exc
        if execution.executed:
            recovery_transition_count += 1
            current_state = execution.next_state
        elif current_state is None:
            current_state = execution.previous_state

        final_stop_report = evaluate_recovery_stop_conditions(
            execution_terminal_reason=execution.terminal_reason,
            next_state=execution.next_state,
            realized_speed_ratio=_realized_speed_ratio(
                execution.next_state,
                branch_state,
            ),
            overspeed_threshold=overspeed_threshold,
            recovery_transition_count=recovery_transition_count,
            recovery_horizon_steps=checked_horizon,
            total_transition_count=prefix_transitions + recovery_transition_count,
            total_horizon_steps=total_horizon,
        )
        record = _runner_record(
            execution,
            step=step,
            stop_report=final_stop_report,
        )
        records.append(record)
        valid = valid and record.valid
        if final_stop_report.terminal_reason is not None:
            terminal_reason = final_stop_report.terminal_reason
            break

    if final_stop_report is None or terminal_reason is None:
        raise RecoveryBranchRunnerError("runner ended without a terminal condition")
    recovery_success_status = final_stop_report.status_for(RECOVERY_SUCCESS)
    diagnostic_log_path = (
        write_diagnostic_log(checked_output, tuple(records))
        if write_diagnostics
        else None
    )
    return RecoveryBranchRunResult(
        branch_id=branch_id,
        records=tuple(records),
        terminal_reason=terminal_reason,
        transition_count=recovery_transition_count,
        valid=valid,
        recovery_success_status=recovery_success_status,
        diagnostic_log_path=diagnostic_log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, validate, or explicitly run one bounded nonformal recovery branch."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print one branch plan only.")
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and optional output path without execution.",
    )
    mode.add_argument(
        "--execute-nonformal",
        action="store_true",
        help="Explicitly execute one bounded branch; never a formal experiment.",
    )
    parser.add_argument("--branch-id", choices=sorted(SUPPORTED_BRANCH_IDS))
    parser.add_argument(
        "--branch-state",
        type=Path,
        default=DEFAULT_BRANCH_STATE_PATH,
    )
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write only the bounded nonformal JSONL diagnostic log.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.plan or args.validate_only or args.execute_nonformal):
        parser.print_help()
        print("\nNo rollout executed.")
        return 0
    try:
        if args.validate_only:
            load_frozen_branch_state(args.branch_state)
            _validate_horizon(args.horizon_steps)
            if args.output_dir is not None:
                validate_output_directory(args.output_dir)
            print("VALIDATION PASS: frozen branch state and runner inputs are valid")
            print("No rollout executed.")
            return 0
        if args.branch_id is None:
            raise RecoveryBranchRunnerError("--branch-id is required for plan or execution")
        if args.plan:
            plan = build_recovery_branch_plan(
                args.branch_id,
                branch_state_path=args.branch_state,
                horizon_steps=args.horizon_steps,
                output_dir=args.output_dir,
            )
            print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
            print("No rollout executed.")
            return 0
        if args.output_dir is None:
            raise RecoveryBranchRunnerError(
                "--output-dir is required for explicit nonformal execution"
            )
        result = run_recovery_branch(
            args.branch_id,
            branch_state_path=args.branch_state,
            horizon_steps=args.horizon_steps,
            output_dir=args.output_dir,
            write_diagnostics=args.write_diagnostics,
        )
        print(
            json.dumps(
                {
                    "branch_id": result.branch_id,
                    "is_formal_experiment": False,
                    "recovery_success_status": result.recovery_success_status,
                    "terminal_reason": result.terminal_reason,
                    "transition_count": result.transition_count,
                    "valid": result.valid,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (FileExistsError, FileNotFoundError, RecoveryBranchRunnerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
