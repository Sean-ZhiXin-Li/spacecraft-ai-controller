from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MANIFEST_RELATIVE_PATH = Path("analysis/final_veto_ablation_v0/manifest.json")
EXPECTED_SCHEMA_VERSION = "final_veto_ablation_manifest_v0"
EXPECTED_EXPERIMENT_STATUS = "design_frozen_not_run"

REQUIRED_TOP_LEVEL_FIELDS = {
    "manifest_schema_version",
    "experiment_id",
    "experiment_status",
    "created_date",
    "source_commit",
    "repository",
    "benchmark_id",
    "benchmark_version",
    "design_document",
    "monitor",
    "hazard",
    "fallback",
    "arms",
    "preservation_set",
    "diagnostic_stress_set",
    "pairing",
    "acceptance_criteria",
    "output_contract",
    "protected_paths",
    "allowed_claims",
    "prohibited_claims",
    "notes",
}

EXPECTED_PRESERVATION_CASES = {
    (1.00, 150, 8000),
    (1.00, 165, 8000),
    (1.00, 170, 8000),
    (1.00, 175, 8000),
    (1.00, 150, 10000),
    (1.00, 165, 10000),
    (1.00, 170, 10000),
    (1.00, 175, 10000),
}

EXPECTED_STRESS_CASES = {
    (0.98, 150, 8000),
    (0.98, 150, 10000),
    (0.98, 165, 10000),
    (0.98, 170, 10000),
    (0.98, 175, 10000),
}

REQUIRED_PAIRING_RULES = {
    "exactly_two_arms_per_case",
    "one_monitor_off",
    "one_monitor_on",
    "stable_paired_run_id",
    "identical_case_hash",
    "identical_config_hash",
    "deterministic_seed_recorded",
    "missing_pair_is_incomplete_evidence",
}

REQUIRED_ACCEPTANCE_CRITERIA = {
    "protected_historical_guard_passes",
    "no_protected_path_modified",
    "exact_preservation_case_count",
    "exact_stress_case_count",
    "complete_counterfactual_pairs",
    "preservation_monitor_on_crossing",
    "preservation_monitor_on_recoverable_crossing",
    "preservation_blocked_successes_zero",
    "invalid_simulation_nonincrease",
    "stress_monitor_off_hazard_exercised",
    "stress_hazard_reduction",
    "paired_avoided_failure",
    "nontrivial_action_execution",
    "blocked_successes_reported",
    "unnecessary_vetoes_reported",
    "false_negatives_reported",
    "fallback_failures_reported",
    "claim_sets_separate",
    "no_formal_safety_claim",
}

# These booleans are freeze-time metadata recorded at manifest source_commit.
# They are not a live publication-readiness assertion; the runner queries
# current Git behavior independently before any formal execution.
EXPECTED_OUTPUT_PATHS = {
    "analysis/final_veto_ablation_v0/results.csv": True,
    "analysis/final_veto_ablation_v0/paired_results.csv": True,
    "analysis/final_veto_ablation_v0/decision_log.jsonl": False,
    "analysis/final_veto_ablation_v0/summary.md": False,
    "analysis/final_veto_ablation_v0/comparison.png": True,
}

REQUIRED_PROTECTED_PATHS = {
    "analysis/phase34_post_cross_sync/",
    "analysis/phase35_crossing_basin_expansion/",
    "analysis/phase36b_transfer_family_benchmark/",
    "analysis/phase36c_non_crossing_geometry_diagnosis/",
    "analysis/phase37a_radial_commit_timing/",
    "analysis/phase37b_weak_tangential_subset/",
    "scripts/check_phase_results.py",
}

FORBIDDEN_MEASURED_OUTCOME_KEYS = {
    "measured_results",
    "observed_results",
    "measured_metrics",
    "run_results",
    "monitor_off_overspeed_count",
    "monitor_on_overspeed_count",
    "avoided_failure_count",
    "blocked_success_count",
    "unnecessary_veto_count",
    "false_negative_count",
    "fallback_failure_count",
}


class ManifestValidationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError("could not locate repository root containing .git and scripts/")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ManifestValidationError([f"manifest does not exist: {path}"]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError([f"manifest is not valid UTF-8 JSON: {path}: {exc}"]) from exc
    if not isinstance(data, dict):
        raise ManifestValidationError(["manifest root must be a JSON object"])
    return data


def stable_case_id(subset_id: str, r0: object, angle: object, thrust: object) -> str:
    r0_token = f"{float(r0):.2f}".replace(".", "p")
    angle_token = f"{float(angle):g}".replace(".", "p")
    thrust_token = f"{float(thrust):g}".replace(".", "p")
    return f"{subset_id}__r0_{r0_token}__angle_{angle_token}__thrust_{thrust_token}"


def case_parameters(case: dict[str, Any]) -> tuple[float, int, int]:
    return (
        round(float(case["r0_over_target"]), 8),
        int(case["initial_velocity_angle_deg"]),
        int(case["thrust_scale"]),
    )


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _check(condition: bool, pass_message: str, error_message: str, passes: list[str], errors: list[str]) -> None:
    if condition:
        passes.append(pass_message)
    else:
        errors.append(error_message)


def _forbidden_key_locations(value: object, prefix: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_MEASURED_OUTCOME_KEYS:
                locations.append(child_path)
            locations.extend(_forbidden_key_locations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_forbidden_key_locations(child, f"{prefix}[{index}]"))
    return locations


def _validate_case_set(
    label: str,
    section: dict[str, Any],
    expected_subset_id: str,
    expected_cases: set[tuple[float, int, int]],
    passes: list[str],
    errors: list[str],
) -> tuple[set[str], set[tuple[float, int, int]]]:
    cases = _list(section.get("cases"))
    expected_count = len(expected_cases)
    _check(
        section.get("subset_id") == expected_subset_id,
        f"{label} subset_id is frozen",
        f"{label} subset_id must be {expected_subset_id!r}",
        passes,
        errors,
    )
    _check(
        section.get("case_count") == expected_count and len(cases) == expected_count,
        f"{label} contains exactly {expected_count} cases",
        f"{label} must declare and contain exactly {expected_count} cases",
        passes,
        errors,
    )

    ids: list[str] = []
    parameters: list[tuple[float, int, int]] = []
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            errors.append(f"{label} case {index} must be an object")
            continue
        try:
            params = case_parameters(raw_case)
            expected_id = stable_case_id(expected_subset_id, *params)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label} case {index} has invalid parameters: {exc}")
            continue
        case_id = raw_case.get("case_id")
        if case_id != expected_id:
            errors.append(f"{label} case {index} case_id is not stable: expected {expected_id!r}, got {case_id!r}")
        if not isinstance(raw_case.get("seed"), int) or isinstance(raw_case.get("seed"), bool):
            errors.append(f"{label} case {index} must record an integer deterministic seed")
        ids.append(str(case_id))
        parameters.append(params)

    unique_ids = set(ids)
    unique_parameters = set(parameters)
    _check(
        len(unique_ids) == len(ids),
        f"{label} case IDs are unique",
        f"{label} contains duplicate case IDs",
        passes,
        errors,
    )
    _check(
        unique_parameters == expected_cases,
        f"{label} parameter combinations match the frozen set",
        f"{label} parameter combinations differ from the frozen set",
        passes,
        errors,
    )
    return unique_ids, unique_parameters


def _is_within_output_directory(path_text: str) -> bool:
    path = PurePosixPath(path_text)
    prefix = PurePosixPath("analysis/final_veto_ablation_v0")
    return not path.is_absolute() and ".." not in path.parts and path.parts[: len(prefix.parts)] == prefix.parts


def _overlaps_protected_path(path_text: str, protected_paths: Iterable[str]) -> bool:
    normalized = PurePosixPath(path_text).as_posix().rstrip("/")
    for protected in protected_paths:
        protected_normalized = PurePosixPath(protected).as_posix().rstrip("/")
        if normalized == protected_normalized or normalized.startswith(f"{protected_normalized}/"):
            return True
    return False


def validate_manifest_data(data: dict[str, Any]) -> list[str]:
    passes: list[str] = []
    errors: list[str] = []

    missing_fields = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(data))
    _check(
        not missing_fields,
        "all required top-level fields are present",
        f"missing required top-level fields: {missing_fields}",
        passes,
        errors,
    )
    _check(
        data.get("manifest_schema_version") == EXPECTED_SCHEMA_VERSION,
        f"schema version is {EXPECTED_SCHEMA_VERSION}",
        f"manifest_schema_version must be {EXPECTED_SCHEMA_VERSION!r}",
        passes,
        errors,
    )
    _check(
        data.get("experiment_status") == EXPECTED_EXPERIMENT_STATUS,
        "experiment status confirms the design has not run",
        f"experiment_status must be {EXPECTED_EXPERIMENT_STATUS!r}",
        passes,
        errors,
    )
    _check(
        bool(re.fullmatch(r"[0-9a-f]{40}", str(data.get("source_commit", "")))),
        "source commit is a full 40-character Git SHA",
        "source_commit must be a lowercase 40-character hexadecimal Git SHA",
        passes,
        errors,
    )

    monitor = _mapping(data.get("monitor"))
    trigger = _mapping(monitor.get("veto_trigger"))
    hazard = _mapping(data.get("hazard"))
    fallback = _mapping(data.get("fallback"))
    prediction_requirements = set(_list(trigger.get("requirements")))
    required_prediction_requirements = {
        "same_current_state",
        "same_nominal_action",
        "same_simulator_constants",
        "same_action_clamp",
        "same_integration_order",
        "same_target_circular_speed_normalization",
    }

    _check(
        monitor.get("implemented") is False,
        "monitor is explicitly not implemented",
        "monitor.implemented must be false at manifest freeze time",
        passes,
        errors,
    )
    _check(
        monitor.get("prediction_horizon_steps") == 1,
        "prediction horizon is frozen at one step",
        "monitor.prediction_horizon_steps must equal 1",
        passes,
        errors,
    )
    _check(
        hazard.get("threshold") == 1.90 and trigger.get("threshold") == 1.90,
        "realized and predicted overspeed thresholds are frozen at 1.90",
        "hazard and veto-trigger thresholds must both equal 1.90",
        passes,
        errors,
    )
    _check(
        hazard.get("comparator") == ">" and trigger.get("comparator") == ">",
        "realized and predicted comparators are frozen at >",
        "hazard and veto-trigger comparators must both be '>'",
        passes,
        errors,
    )
    _check(
        required_prediction_requirements <= prediction_requirements,
        "one-step prediction equality requirements are complete",
        "veto trigger is missing one or more rollout-equality requirements",
        passes,
        errors,
    )
    _check(
        fallback.get("action") == [0.0, 0.0] and fallback.get("duration_steps") == 1,
        "fallback is frozen as one zero-action step",
        "fallback must be action [0.0, 0.0] for exactly one step",
        passes,
        errors,
    )
    _check(
        fallback.get("proven_safe") is False,
        "fallback is not marked proven safe",
        "fallback.proven_safe must be false",
        passes,
        errors,
    )
    _check(
        hazard.get("formal_safety_boundary") is False,
        "hazard threshold is not marked as a formal safety boundary",
        "hazard.formal_safety_boundary must be false",
        passes,
        errors,
    )

    preservation_ids, preservation_parameters = _validate_case_set(
        "preservation set",
        _mapping(data.get("preservation_set")),
        "phase34_known_recoverable_preservation_v1",
        EXPECTED_PRESERVATION_CASES,
        passes,
        errors,
    )
    stress_ids, stress_parameters = _validate_case_set(
        "diagnostic stress set",
        _mapping(data.get("diagnostic_stress_set")),
        "phase35_radial_energy_push_overspeed_stress_v0",
        EXPECTED_STRESS_CASES,
        passes,
        errors,
    )
    _check(
        preservation_ids.isdisjoint(stress_ids) and preservation_parameters.isdisjoint(stress_parameters),
        "preservation and stress sets are disjoint",
        "a case appears in both preservation and diagnostic stress sets",
        passes,
        errors,
    )

    arm_ids = [arm.get("arm_id") for arm in _list(data.get("arms")) if isinstance(arm, dict)]
    _check(
        len(arm_ids) == 2 and set(arm_ids) == {"monitor_off", "monitor_on"},
        "exactly monitor_off and monitor_on arms are declared",
        "arms must contain exactly one monitor_off and one monitor_on declaration",
        passes,
        errors,
    )
    pairing = _mapping(data.get("pairing"))
    _check(
        pairing.get("arms_per_case") == 2 and REQUIRED_PAIRING_RULES <= set(_list(pairing.get("requirements"))),
        "pairing requirements are complete",
        "pairing contract is missing required off/on completeness rules",
        passes,
        errors,
    )

    acceptance = _mapping(data.get("acceptance_criteria"))
    criterion_ids = {
        item.get("criterion_id")
        for item in _list(acceptance.get("rules"))
        if isinstance(item, dict)
    }
    _check(
        acceptance.get("status") == "future_evaluation_rules_not_results"
        and REQUIRED_ACCEPTANCE_CRITERIA <= criterion_ids,
        "all required future acceptance protections are declared",
        "acceptance criteria are missing required future protections or are represented as results",
        passes,
        errors,
    )

    protected_paths = set(_list(data.get("protected_paths")))
    _check(
        REQUIRED_PROTECTED_PATHS <= protected_paths,
        "all required protected paths are declared",
        "protected_paths is missing one or more protected historical locations",
        passes,
        errors,
    )

    output_contract = _mapping(data.get("output_contract"))
    artifacts = _list(output_contract.get("future_artifacts"))
    actual_output_map = {
        item.get("path"): item.get("currently_ignored_by_gitignore")
        for item in artifacts
        if isinstance(item, dict)
    }
    _check(
        actual_output_map == EXPECTED_OUTPUT_PATHS,
        "future output paths and freeze-time ignore metadata are frozen",
        "future output paths or their recorded freeze-time .gitignore metadata differ from the frozen contract",
        passes,
        errors,
    )
    all_output_paths = [str(path) for path in actual_output_map]
    all_output_paths.append(str(output_contract.get("manifest_path", "")))
    _check(
        all(_is_within_output_directory(path) for path in all_output_paths),
        "manifest and future outputs are isolated under analysis/final_veto_ablation_v0/",
        "one or more output paths escape analysis/final_veto_ablation_v0/",
        passes,
        errors,
    )
    _check(
        not any(_overlaps_protected_path(path, protected_paths) for path in all_output_paths),
        "no manifest or future output path overlaps a protected path",
        "one or more output paths overlap a protected historical path",
        passes,
        errors,
    )

    allowed_claims = _list(data.get("allowed_claims"))
    prohibited_claims = _list(data.get("prohibited_claims"))
    _check(
        bool(allowed_claims) and all(isinstance(item, str) and item for item in allowed_claims),
        "allowed claims are explicitly scoped",
        "allowed_claims must be a non-empty list of strings",
        passes,
        errors,
    )
    _check(
        bool(prohibited_claims)
        and "formal_safety" in prohibited_claims
        and all(isinstance(item, str) and item for item in prohibited_claims),
        "prohibited claims include formal safety",
        "prohibited_claims must be non-empty and include formal_safety",
        passes,
        errors,
    )

    forbidden_locations = _forbidden_key_locations(data)
    _check(
        not forbidden_locations,
        "manifest contains no measured ablation outcome fields",
        f"manifest contains measured-outcome keys at: {forbidden_locations}",
        passes,
        errors,
    )

    if errors:
        raise ManifestValidationError(errors)
    return passes


def validate_manifest(path: Path) -> list[str]:
    return validate_manifest_data(load_manifest(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the frozen Final Veto ablation manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path; defaults to analysis/final_veto_ablation_v0/manifest.json under the repository root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = find_repository_root()
        manifest_path = args.manifest.resolve() if args.manifest else repository_root / MANIFEST_RELATIVE_PATH
        passes = validate_manifest(manifest_path)
    except (FileNotFoundError, ManifestValidationError) as exc:
        errors = exc.errors if isinstance(exc, ManifestValidationError) else [str(exc)]
        for error in errors:
            print(f"FAIL {error}")
        print(f"Final Veto manifest validation FAILED with {len(errors)} issue(s).")
        return 1

    for message in passes:
        print(f"PASS {message}")
    print(f"Final Veto manifest validation PASSED with {len(passes)} checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
