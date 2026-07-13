from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_final_veto_manifest import (  # noqa: E402
    MANIFEST_RELATIVE_PATH,
    ManifestValidationError,
    load_manifest,
    validate_manifest_data,
)
from scripts.final_veto_artifacts import (  # noqa: E402
    ARM_FIELDNAMES,
    ARM_JSON_LIST_FIELDS,
    ARM_SCHEMA_VERSION,
    DECISION_FIELDNAMES,
    DECISION_SCHEMA_VERSION,
    PAIR_FIELDNAMES,
    PAIR_SCHEMA_VERSION,
    ArtifactWriteError,
    validate_output_directory,
)
from scripts.run_final_veto_ablation import (  # noqa: E402
    RunnerContractError,
    build_pair_record,
    build_planned_jobs,
)


PRESERVATION_SUBSET_ID = "phase34_known_recoverable_preservation_v1"
STRESS_SUBSET_ID = "phase35_radial_energy_push_overspeed_stress_v0"
CONTROLLED_TERMINAL_LABELS = frozenset(
    {
        "success",
        "no_crossing",
        "crossing_unrecoverable",
        "recoverable_crossing_failed_late",
        "overspeed",
        "instability",
        "timeout",
        "resource_depletion",
        "unsafe_state",
        "invalid_simulation",
        "unknown",
    }
)
BOOLEAN_ARM_FIELDS = frozenset(
    {
        "monitor_enabled",
        "crossed_target_radius",
        "recoverable_crossing",
        "final_simulator_success",
        "overspeed",
        "instability",
        "unsafe_state",
        "invalid_simulation",
        "is_full_benchmark",
        "known_phase34_recoverable_case",
        "accepted_as_progress",
        "is_formal_experiment",
    }
)
INTEGER_ARM_FIELDS = frozenset(
    {
        "seed",
        "monitor_evaluation_count",
        "allow_count",
        "veto_count",
        "fallback_count",
        "false_negative_count",
        "fallback_failure_count",
        "invalid_monitor_evaluation_count",
        "nominal_actions_unchanged_count",
        "steps",
    }
)
BOOLEAN_PAIR_FIELDS = frozenset(
    {
        "pair_complete",
        "pair_valid",
        "off_overspeed",
        "on_overspeed",
        "off_crossed_target_radius",
        "on_crossed_target_radius",
        "off_recoverable_crossing",
        "on_recoverable_crossing",
        "off_final_simulator_success",
        "on_final_simulator_success",
        "off_invalid_simulation",
        "on_invalid_simulation",
        "avoided_failure",
        "blocked_success",
        "unnecessary_veto",
        "claim_eligible",
        "is_formal_experiment",
    }
)
INTEGER_PAIR_FIELDS = frozenset(
    {
        "seed",
        "on_monitor_evaluation_count",
        "on_allow_count",
        "on_veto_count",
        "on_fallback_count",
        "on_false_negative_count",
        "on_fallback_failure_count",
    }
)


@dataclass(frozen=True, slots=True)
class ResultValidationReport:
    structural_valid: bool
    pair_complete: bool
    preservation_acceptance: str
    stress_hazard_exercised: str
    positive_claim_eligible: bool
    errors: tuple[str, ...]
    observations: tuple[str, ...]
    metrics: Mapping[str, int]


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{field} must be true or false")


def _parse_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        parsed = int(value)
        if str(parsed) == value.strip() or value.strip().lstrip("+").isdigit():
            return parsed
    raise ValueError(f"{field} must be an integer")


def _parse_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _parse_json_list(value: object, field: str) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain valid compact JSON") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"{field} must be a JSON array")


def _normalize_arm_record(record: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(record)
    for field in BOOLEAN_ARM_FIELDS:
        normalized[field] = _parse_bool(record.get(field), field)
    for field in INTEGER_ARM_FIELDS:
        normalized[field] = _parse_int(record.get(field), field)
    for field in ARM_JSON_LIST_FIELDS:
        normalized[field] = _parse_json_list(record.get(field), field)
    normalized["hazard_threshold"] = _parse_float(
        record.get("hazard_threshold"), "hazard_threshold"
    )
    return normalized


def _normalize_pair_record(record: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(record)
    for field in BOOLEAN_PAIR_FIELDS:
        normalized[field] = _parse_bool(record.get(field), field)
    for field in INTEGER_PAIR_FIELDS:
        normalized[field] = _parse_int(record.get(field), field)
    performance = record.get("performance_cost_summary")
    if isinstance(performance, str):
        try:
            performance = json.loads(performance)
        except json.JSONDecodeError as exc:
            raise ValueError("performance_cost_summary must be valid JSON") from exc
    if not isinstance(performance, dict):
        raise ValueError("performance_cost_summary must be a JSON object")
    normalized["performance_cost_summary"] = performance
    return normalized


def _missing_fields(record: Mapping[str, object], required: Sequence[str]) -> list[str]:
    return [field for field in required if field not in record]


def _claim_contains_formal_safety(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            is_empty_or_prohibited = (
                child is False
                or child is None
                or child == ""
                or child == "prohibited"
            )
            if lowered in {"formal_safety_claim", "formal_safety"} and not is_empty_or_prohibited:
                return True
            if _claim_contains_formal_safety(child):
                return True
    elif isinstance(value, list):
        return any(_claim_contains_formal_safety(item) for item in value)
    return False


def validate_decision_events(events: Iterable[Mapping[str, object]]) -> list[str]:
    errors: list[str] = []
    decision_ids: set[str] = set()
    for index, event in enumerate(events, start=1):
        prefix = f"decision event {index}"
        missing = _missing_fields(event, DECISION_FIELDNAMES)
        if missing:
            errors.append(f"{prefix} missing fields: {missing}")
            continue
        if event.get("decision_schema_version") != DECISION_SCHEMA_VERSION:
            errors.append(f"{prefix} has wrong decision schema version")
        decision_id = str(event.get("decision_id"))
        if decision_id in decision_ids:
            errors.append(f"duplicate decision_id: {decision_id}")
        decision_ids.add(decision_id)
        if event.get("arm_id") != "monitor_on":
            errors.append(f"{prefix} creates a runtime-assurance event for monitor_off")
        if event.get("hazard_threshold") != 1.90 or event.get("hazard_comparator") != ">":
            errors.append(f"{prefix} drifts from strict > 1.90")
        if any(key in event for key in ("avoided_failure", "blocked_success", "unnecessary_veto")):
            errors.append(f"{prefix} contains pair-level counterfactual claims")
        if _claim_contains_formal_safety(event):
            errors.append(f"{prefix} contains a formal-safety claim")
    return errors


def validate_result_records(
    arm_records: Iterable[Mapping[str, object]],
    pair_records: Iterable[Mapping[str, object]],
    manifest: Mapping[str, Any],
    *,
    decision_events: Iterable[Mapping[str, object]] = (),
    output_directory: Path | None = None,
    repository_root: Path = PROJECT_ROOT,
    require_formal_acceptance: bool = False,
) -> ResultValidationReport:
    errors: list[str] = []
    observations: list[str] = []
    manifest_valid = True
    try:
        validate_manifest_data(dict(manifest))
    except ManifestValidationError as exc:
        manifest_valid = False
        errors.extend(f"frozen manifest: {error}" for error in exc.errors)

    expected_jobs: dict[tuple[str, str], object] = {}
    if manifest_valid:
        try:
            expected_jobs = {
                (job.case_id, job.arm_id): job
                for job in build_planned_jobs(manifest)
            }
        except RunnerContractError as exc:
            errors.append(f"frozen job plan: {exc}")

    if output_directory is not None:
        try:
            validate_output_directory(
                output_directory,
                repository_root=repository_root,
                protected_paths=manifest.get("protected_paths", []),
            )
        except ArtifactWriteError as exc:
            errors.append(str(exc))

    raw_arms = list(arm_records)
    raw_pairs = list(pair_records)
    arms: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index, raw in enumerate(raw_arms, start=1):
        missing = _missing_fields(raw, ARM_FIELDNAMES)
        if missing:
            errors.append(f"arm row {index} missing fields: {missing}")
            continue
        try:
            row = _normalize_arm_record(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"arm row {index}: {exc}")
            continue
        arms.append(row)

    for index, raw in enumerate(raw_pairs, start=1):
        missing = _missing_fields(raw, PAIR_FIELDNAMES)
        if missing:
            errors.append(f"pair row {index} missing fields: {missing}")
            continue
        try:
            row = _normalize_pair_record(raw)
        except (TypeError, ValueError) as exc:
            errors.append(f"pair row {index}: {exc}")
            continue
        pairs.append(row)

    run_ids: set[str] = set()
    pair_arms: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in arms:
        run_id = str(row["run_id"])
        pair_arm = (str(row["paired_run_id"]), str(row["arm_id"]))
        if run_id in run_ids:
            errors.append(f"duplicate run_id: {run_id}")
        if pair_arm in pair_arms:
            errors.append(f"duplicate pair/arm combination: {pair_arm}")
        run_ids.add(run_id)
        pair_arms.add(pair_arm)
        grouped.setdefault(str(row["paired_run_id"]), []).append(row)

        if row["schema_version"] != ARM_SCHEMA_VERSION:
            errors.append(f"{run_id} has wrong Result Schema version")
        if row["experiment_id"] != manifest.get("experiment_id"):
            errors.append(f"{run_id} has wrong experiment_id")
        expected_job = expected_jobs.get((str(row["case_id"]), str(row["arm_id"])))
        if expected_jobs and expected_job is None:
            errors.append(f"{run_id} is not a case/arm declared by the frozen manifest")
        elif expected_job is not None:
            expected_fields = {
                "paired_run_id": expected_job.paired_run_id,
                "subset_id": expected_job.subset_id,
                "seed": expected_job.seed,
                "controller_id": expected_job.controller_id,
                "case_config_hash": expected_job.case_config_hash,
                "run_config_hash": expected_job.run_config_hash,
                "r0_over_target": expected_job.r0_over_target,
                "initial_velocity_angle_deg": expected_job.initial_velocity_angle_deg,
                "thrust_scale": expected_job.thrust_scale,
            }
            drifted = [
                field
                for field, expected_value in expected_fields.items()
                if row.get(field) != expected_value
            ]
            if drifted:
                errors.append(f"{run_id} drifts from frozen job fields: {drifted}")
        if row["hazard_threshold"] != 1.90 or row["hazard_comparator"] != ">":
            errors.append(f"{run_id} drifts from strict > 1.90")
        if row["terminal_label"] not in CONTROLLED_TERMINAL_LABELS:
            errors.append(f"{run_id} has uncontrolled terminal_label")
        if row["terminal_label"] == "unknown" and not str(row["manual_audit_note"]).strip():
            errors.append(f"{run_id} uses unknown without a manual audit note")
        if row["subset_id"] not in {PRESERVATION_SUBSET_ID, STRESS_SUBSET_ID}:
            errors.append(f"{run_id} has an undeclared subset")
        expected_known = row["subset_id"] == PRESERVATION_SUBSET_ID
        if row["known_phase34_recoverable_case"] != expected_known:
            errors.append(f"{run_id} has inconsistent known-success membership")
        if row["is_full_benchmark"]:
            errors.append(f"{run_id} incorrectly marks a subset row as full benchmark")
        if row["accepted_as_progress"] and not row["is_formal_experiment"]:
            errors.append(f"{run_id} accepts progress from a smoke/nonformal row")
        if row["is_formal_experiment"] and row["experiment_status"] == "nonformal_smoke":
            errors.append(f"{run_id} marks a smoke status as formal")
        if _claim_contains_formal_safety(row):
            errors.append(f"{run_id} contains a formal-safety claim")

        counts = [
            row["monitor_evaluation_count"],
            row["allow_count"],
            row["veto_count"],
            row["fallback_count"],
            row["false_negative_count"],
            row["fallback_failure_count"],
            row["invalid_monitor_evaluation_count"],
        ]
        if any(int(value) < 0 for value in counts):
            errors.append(f"{run_id} has a negative monitor count")
        if row["arm_id"] == "monitor_off":
            if row["monitor_enabled"]:
                errors.append(f"{run_id} enables the monitor in monitor_off")
            if any(int(row[field]) for field in (
                "monitor_evaluation_count",
                "allow_count",
                "veto_count",
                "fallback_count",
                "false_negative_count",
                "fallback_failure_count",
                "invalid_monitor_evaluation_count",
            )):
                errors.append(f"{run_id} records monitor activity in monitor_off")
        elif row["arm_id"] == "monitor_on":
            if not row["monitor_enabled"]:
                errors.append(f"{run_id} disables the monitor in monitor_on")
            if row["allow_count"] + row["veto_count"] != row["monitor_evaluation_count"]:
                errors.append(f"{run_id} allow + veto does not equal evaluations")
            if row["fallback_count"] != row["veto_count"]:
                errors.append(f"{run_id} fallback count does not equal veto count")
        else:
            errors.append(f"{run_id} has invalid arm_id")

    pair_complete = True
    for pair_id, rows in grouped.items():
        if len(rows) != 2 or {str(row["arm_id"]) for row in rows} != {
            "monitor_off",
            "monitor_on",
        }:
            pair_complete = False
            errors.append(f"{pair_id} is missing or duplicates an arm")
            continue
        if len({str(row["case_config_hash"]) for row in rows}) != 1:
            pair_complete = False
            errors.append(f"{pair_id} has mismatched case hashes")

    pair_by_id: dict[str, dict[str, object]] = {}
    for row in pairs:
        pair_id = str(row["paired_run_id"])
        if pair_id in pair_by_id:
            errors.append(f"duplicate pair result: {pair_id}")
            pair_complete = False
        pair_by_id[pair_id] = row
        if row["pair_schema_version"] != PAIR_SCHEMA_VERSION:
            errors.append(f"{pair_id} has wrong pair schema version")
        if row["experiment_id"] != manifest.get("experiment_id"):
            errors.append(f"{pair_id} has wrong experiment_id")
        if row["avoided_failure"] and (
            not row["pair_complete"]
            or not row["pair_valid"]
            or not row["off_overspeed"]
            or row["on_overspeed"]
            or row["on_invalid_simulation"]
        ):
            errors.append(f"{pair_id} claims avoided failure without a valid hazard counterfactual")
        if row["blocked_success"] and not (
            (row["off_recoverable_crossing"] and not row["on_recoverable_crossing"])
            or (row["off_final_simulator_success"] and not row["on_final_simulator_success"])
        ):
            errors.append(f"{pair_id} claims blocked success without lost off-arm success")
        if row["claim_eligible"] and not row["is_formal_experiment"]:
            errors.append(f"{pair_id} is claim eligible despite being nonformal")
        if _claim_contains_formal_safety(row):
            errors.append(f"{pair_id} contains a formal-safety claim")

    if set(pair_by_id) != set(grouped):
        pair_complete = False
        missing = sorted(set(grouped) - set(pair_by_id))
        extra = sorted(set(pair_by_id) - set(grouped))
        if missing:
            errors.append(f"missing pair result rows: {missing}")
        if extra:
            errors.append(f"pair results have no matching arm rows: {extra}")

    for pair_id in sorted(set(pair_by_id) & set(grouped)):
        rows = grouped[pair_id]
        if len(rows) != 2:
            continue
        try:
            expected = build_pair_record(rows)
        except RunnerContractError as exc:
            errors.append(f"{pair_id} cannot be paired: {exc}")
            pair_complete = False
            continue
        actual = pair_by_id[pair_id]
        comparison_fields = (
            "case_config_hash",
            "pair_complete",
            "pair_valid",
            "off_overspeed",
            "on_overspeed",
            "avoided_failure",
            "blocked_success",
            "unnecessary_veto",
            "on_false_negative_count",
            "on_fallback_failure_count",
        )
        mismatches = [field for field in comparison_fields if actual[field] != expected[field]]
        if mismatches:
            errors.append(f"{pair_id} pair metrics disagree with arm rows: {mismatches}")

    errors.extend(validate_decision_events(decision_events))

    formal_rows = bool(arms) and all(bool(row["is_formal_experiment"]) for row in arms)
    preservation_on = [
        row
        for row in arms
        if row["subset_id"] == PRESERVATION_SUBSET_ID and row["arm_id"] == "monitor_on"
    ]
    stress_off = [
        row
        for row in arms
        if row["subset_id"] == STRESS_SUBSET_ID and row["arm_id"] == "monitor_off"
    ]
    stress_on = [
        row
        for row in arms
        if row["subset_id"] == STRESS_SUBSET_ID and row["arm_id"] == "monitor_on"
    ]
    preservation_pairs = [
        row for row in pairs if row["subset_id"] == PRESERVATION_SUBSET_ID
    ]
    preservation_pass = (
        len(preservation_on) == 8
        and all(bool(row["crossed_target_radius"]) for row in preservation_on)
        and all(bool(row["recoverable_crossing"]) for row in preservation_on)
        and not any(bool(row["blocked_success"]) for row in preservation_pairs)
    )
    off_stress_overspeed = sum(bool(row["overspeed"]) for row in stress_off)
    on_stress_overspeed = sum(bool(row["overspeed"]) for row in stress_on)
    avoided_count = sum(bool(row["avoided_failure"]) for row in pairs)
    invalid_off = sum(
        bool(row["invalid_simulation"]) for row in arms if row["arm_id"] == "monitor_off"
    )
    invalid_on = sum(
        bool(row["invalid_simulation"]) for row in arms if row["arm_id"] == "monitor_on"
    )
    total_evaluations = sum(int(row["monitor_evaluation_count"]) for row in stress_on + preservation_on)
    total_vetoes = sum(int(row["veto_count"]) for row in stress_on + preservation_on)
    stress_exercised = len(stress_off) == 5 and off_stress_overspeed >= 1
    hazard_reduced = (
        len(stress_on) == 5 and on_stress_overspeed <= off_stress_overspeed - 1
    )
    nontrivial = total_evaluations > 0 and total_vetoes < total_evaluations

    if not formal_rows:
        preservation_status = "not_evaluated_nonformal"
        stress_status = "not_evaluated_nonformal"
    else:
        preservation_status = "pass" if preservation_pass else "fail"
        stress_status = "pass" if stress_exercised else "monitor_not_exercised"
    structural_valid = not errors
    positive_claim = (
        structural_valid
        and pair_complete
        and formal_rows
        and len(arms) == 26
        and len(pairs) == 13
        and preservation_pass
        and stress_exercised
        and hazard_reduced
        and avoided_count >= 1
        and invalid_on <= invalid_off
        and nontrivial
    )
    if require_formal_acceptance and not positive_claim:
        errors.append("formal positive-claim acceptance criteria are not satisfied")
        structural_valid = False

    observations.extend(
        [
            f"arm_rows={len(arms)} pair_rows={len(pairs)}",
            f"preservation_monitor_on={len(preservation_on)}",
            f"stress_monitor_off_overspeed={off_stress_overspeed}",
            f"stress_monitor_on_overspeed={on_stress_overspeed}",
            f"avoided_failures={avoided_count}",
            f"monitor_evaluations={total_evaluations} vetoes={total_vetoes}",
        ]
    )
    metrics = {
        "arm_rows": len(arms),
        "pair_rows": len(pairs),
        "preservation_monitor_on": len(preservation_on),
        "stress_monitor_off_overspeed": off_stress_overspeed,
        "stress_monitor_on_overspeed": on_stress_overspeed,
        "avoided_failures": avoided_count,
        "blocked_successes": sum(bool(row["blocked_success"]) for row in pairs),
        "unnecessary_vetoes": sum(bool(row["unnecessary_veto"]) for row in pairs),
        "false_negatives": sum(int(row["on_false_negative_count"]) for row in pairs),
        "fallback_failures": sum(int(row["on_fallback_failure_count"]) for row in pairs),
    }
    return ResultValidationReport(
        structural_valid=structural_valid,
        pair_complete=pair_complete,
        preservation_acceptance=preservation_status,
        stress_hazard_exercised=stress_status,
        positive_claim_eligible=positive_claim,
        errors=tuple(errors),
        observations=tuple(observations),
        metrics=metrics,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Final Veto arm and pair results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--formal", action="store_true", help="Require formal positive-claim acceptance.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest or PROJECT_ROOT / MANIFEST_RELATIVE_PATH
        manifest = load_manifest(manifest_path)
        arm_rows = read_csv_rows(args.results)
        pair_rows = read_csv_rows(args.paired_results)
        events = read_jsonl(args.decision_log) if args.decision_log else []
        report = validate_result_records(
            arm_rows,
            pair_rows,
            manifest,
            decision_events=events,
            output_directory=args.results.parent,
            require_formal_acceptance=args.formal,
        )
    except (OSError, ValueError, ManifestValidationError) as exc:
        print(f"FAIL {exc}")
        return 1

    print(f"STRUCTURAL {'PASS' if report.structural_valid else 'FAIL'}")
    print(f"PAIR_COMPLETENESS {'PASS' if report.pair_complete else 'FAIL'}")
    print(f"PRESERVATION {report.preservation_acceptance}")
    print(f"STRESS_HAZARD_EXERCISE {report.stress_hazard_exercised}")
    print(f"POSITIVE_CLAIM_ELIGIBLE {str(report.positive_claim_eligible).lower()}")
    for observation in report.observations:
        print(f"INFO {observation}")
    for error in report.errors:
        print(f"FAIL {error}")
    return 0 if report.structural_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
