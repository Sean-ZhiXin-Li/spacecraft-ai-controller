from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MANIFEST_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/manifest.json"
)
EXPECTED_SCHEMA_VERSION = "recovery_action_branching_manifest_v0"
EXPECTED_EXPERIMENT_ID = "recovery_action_branching_nonformal_v0"
EXPECTED_EXPERIMENT_STATUS = "design_frozen_not_run"
EXPECTED_CREATED_DATE = "2026-07-18"

REQUIRED_TOP_LEVEL_FIELDS = {
    "manifest_schema_version",
    "experiment_id",
    "experiment_status",
    "is_formal_experiment",
    "created_date",
    "source_commit",
    "repository",
    "design_document",
    "metrics_document",
    "design_state",
    "source_evidence",
    "source_case",
    "hazard",
    "branch_point",
    "coordinate_convention",
    "branches",
    "recovery_action_rejection",
    "horizons",
    "recovery_success",
    "stop_conditions",
    "required_metrics",
    "comparison_contract",
    "output_contract",
    "protected_paths",
    "allowed_claims",
    "prohibited_claims",
    "notes",
}

EXPECTED_SOURCE_CASE = {
    "case_id": "phase35_radial_energy_push_overspeed_stress_v0__r0_0p98__angle_150__thrust_8000",
    "subset_id": "phase35_radial_energy_push_overspeed_stress_v0",
    "r0_over_target": 0.98,
    "initial_velocity_angle_deg": 150,
    "thrust_scale": 8000,
    "seed": 0,
}

EXPECTED_SOURCE_CONTEXT = {
    "phase": "phase35",
    "controller_id": "phase35_crossing_basin_expansion",
    "upstream_variant": "radial_energy_push",
    "post_cross_context": "phase34_radius_priority",
}

EXPECTED_FINAL_VETO_EVIDENCE_PATHS = {
    "analysis/final_veto_ablation_v0/manifest.json",
    "analysis/final_veto_ablation_v0/results.csv",
    "analysis/final_veto_ablation_v0/paired_results.csv",
    "analysis/final_veto_ablation_v0/decision_log.jsonl",
    "analysis/final_veto_ablation_v0/summary.md",
    "analysis/final_veto_ablation_v0/comparison.png",
}

REQUIRED_BRANCH_STATE_FIELDS = {
    "step",
    "state_vector",
    "position",
    "velocity",
    "phase",
    "active_stage",
    "nominal_proposed_action",
    "predicted_nominal_next_state",
    "predicted_nominal_speed_ratio",
    "hazard_threshold",
    "hazard_comparator",
    "monitor_decision",
    "implementation_commit",
    "simulator_constants_hash",
    "case_configuration_hash",
    "canonical_branch_state_hash",
}

REQUIRED_BRANCH_VALIDITY_RULES = {
    "all_state_values_are_finite",
    "nominal_action_is_finite",
    "predicted_nominal_next_state_is_finite",
    "predicted_nominal_speed_ratio_is_finite",
    "monitor_evaluation_is_valid",
    "implementation_commit_is_recorded",
    "simulator_constants_hash_is_recorded",
    "case_configuration_hash_is_recorded",
}

EXPECTED_BRANCH_IDS = {
    "zero_action_reference_v0",
    "velocity_opposed_thrust_v0",
    "tangential_error_correction_v0",
    "explicit_abort_v0",
}

EXPECTED_STOP_LABELS = {
    "recovery_success": ("recovery_success", "success"),
    "realized_overspeed": ("overspeed", "overspeed"),
    "instability": ("instability", "instability"),
    "unsafe_state": ("unsafe_state", "unsafe_state"),
    "invalid_simulation": ("invalid_simulation", "invalid_simulation"),
    "invalid_recovery_evaluation": (
        "invalid_recovery_evaluation",
        "unknown_with_manual_audit",
    ),
    "recovery_action_rejection": (
        "recovery_action_rejected",
        "unknown_with_manual_audit",
    ),
    "explicit_abort": ("explicit_recovery_abort", "unknown_with_manual_audit"),
    "recovery_horizon_exhaustion": ("recovery_horizon_exhausted", "timeout"),
    "total_horizon_exhaustion": ("total_horizon_exhausted", "timeout"),
}

EXPECTED_STOP_PRIORITY = [
    "invalid_simulation",
    "invalid_recovery_evaluation",
    "realized_overspeed",
    "instability",
    "unsafe_state",
    "recovery_action_rejection",
    "explicit_abort",
    "recovery_success",
    "recovery_horizon_exhaustion",
    "total_horizon_exhaustion",
]

REQUIRED_RECOVERY_SUCCESS_CONDITIONS = {
    "declared_hazard_avoided",
    "not_invalid_simulation",
    "not_invalid_recovery_evaluation",
    "target_radius_crossing",
    "phase34_compatible_recoverable_crossing",
    "recovery_target_reached_within_10000_transitions",
}

REQUIRED_MARGIN_METRICS = {
    "overspeed_headroom",
    "action_saturation_margin",
    "available_correction_authority",
    "required_to_available_correction_ratio",
}

REQUIRED_COST_METRICS = {
    "recovery_steps",
    "total_steps",
    "normalized_action_effort",
    "delta_v_proxy",
    "crossing_delay",
    "final_radius_error",
    "final_radial_velocity_error",
    "final_tangential_velocity_error",
    "task_abandonment_status",
}

REQUIRED_INTERVENTION_METRICS = {
    "evaluation_count",
    "allow_count",
    "veto_count",
    "recovery_action_rejection_count",
    "first_intervention_step",
    "last_intervention_step",
    "longest_veto_streak",
    "veto_segment_count",
    "action_suppression_duration",
}

REQUIRED_OUTCOME_METRICS = {
    "hazard_avoided",
    "recovery_success",
    "simulator_success",
    "terminal_label",
    "branch_terminal_label",
    "recovery_outcome_taxonomy",
    "task_recovery",
    "explicit_abort",
    "invalid_evaluation",
    "new_failure_caused_by_recovery_action",
}

REQUIRED_COMPARISON_RULES = {
    "identical_canonical_branch_state_hash",
    "identical_case_configuration_hash",
    "identical_simulator_constants_hash",
    "identical_nominal_prefix",
    "identical_seed",
    "only_branch_decision_differs",
    "explicit_abort_has_zero_post_branch_transitions",
}

EXPECTED_OUTPUT_PATHS = {
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
    "analysis/recovery_action_branching_nonformal_v0/results.csv",
    "analysis/recovery_action_branching_nonformal_v0/decision_log.jsonl",
    "analysis/recovery_action_branching_nonformal_v0/summary.md",
    "analysis/recovery_action_branching_nonformal_v0/comparison.png",
}
BRANCH_STATE_OUTPUT_PATH = (
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)
EXPERIMENT_RESULT_OUTPUT_PATHS = EXPECTED_OUTPUT_PATHS - {
    BRANCH_STATE_OUTPUT_PATH
}

REQUIRED_PROTECTED_PATHS = {
    "analysis/final_veto_ablation_v0/",
    "analysis/phase34_post_cross_sync/",
    "analysis/phase35_crossing_basin_expansion/",
    "analysis/phase36b_transfer_family_benchmark/",
    "analysis/phase36c_non_crossing_geometry_diagnosis/",
    "analysis/phase37a_radial_commit_timing/",
    "analysis/phase37b_weak_tangential_subset/",
    "scripts/check_phase_results.py",
}

REQUIRED_PROHIBITED_CLAIMS = {
    "formal_safety",
    "universal_recovery",
    "controller_superiority",
    "benchmark_wide_effectiveness",
    "cross_case_generalization",
    "hardware_validity",
    "deployment_readiness",
    "cross_embodiment_validation",
    "proof_action_magnitude_0p25_is_optimal",
    "proof_recovery_horizon_10000_is_sufficient",
}

EXPECTED_ALLOWED_CLAIMS = {
    "branch_specific_diagnostic_outcome_for_one_predeclared_case",
    "whether_a_branch_avoided_the_declared_overspeed_hazard_in_that_case",
    "whether_a_branch_reached_the_predeclared_recovery_target_in_that_case",
    "relative_intervention_burden_and_cost_for_the_four_branches_in_that_case",
}

FORBIDDEN_RESULT_KEYS = {
    "measured_outcomes",
    "measured_results",
    "observed_results",
    "observed_metrics",
    "result_rows",
    "results",
    "branch_winner",
    "winning_branch",
    "best_branch",
    "branch_rankings",
    "branch_scores",
    "recovery_success_count",
    "hazard_avoided_count",
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
        raise ManifestValidationError(
            [f"manifest is not valid UTF-8 JSON: {path}: {exc}"]
        ) from exc
    if not isinstance(data, dict):
        raise ManifestValidationError(["manifest root must be a JSON object"])
    return data


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _check(
    condition: bool,
    pass_message: str,
    error_message: str,
    passes: list[str],
    errors: list[str],
) -> None:
    if condition:
        passes.append(pass_message)
    else:
        errors.append(error_message)


def _forbidden_key_locations(value: object, prefix: str = "$") -> list[str]:
    locations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_RESULT_KEYS:
                locations.append(child_path)
            locations.extend(_forbidden_key_locations(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(_forbidden_key_locations(child, f"{prefix}[{index}]"))
    return locations


def _is_within_output_directory(path_text: str) -> bool:
    path = PurePosixPath(path_text)
    prefix = PurePosixPath("analysis/recovery_action_branching_nonformal_v0")
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts[: len(prefix.parts)] == prefix.parts
    )


def _overlaps_protected_path(
    path_text: str, protected_paths: Iterable[str]
) -> bool:
    normalized = PurePosixPath(path_text).as_posix().rstrip("/")
    for protected in protected_paths:
        protected_normalized = PurePosixPath(protected).as_posix().rstrip("/")
        if normalized == protected_normalized or normalized.startswith(
            f"{protected_normalized}/"
        ):
            return True
    return False


def _branch_map(data: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    branches = [item for item in _list(data.get("branches")) if isinstance(item, dict)]
    ids = [str(item.get("branch_id")) for item in branches]
    return ids, {str(item.get("branch_id")): item for item in branches}


def _future_output_paths(data: dict[str, Any]) -> list[str]:
    output = _mapping(data.get("output_contract"))
    return [
        str(item.get("path", ""))
        for item in _list(output.get("future_artifacts"))
        if isinstance(item, dict)
    ]


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
        data.get("experiment_id") == EXPECTED_EXPERIMENT_ID,
        f"experiment ID is {EXPECTED_EXPERIMENT_ID}",
        f"experiment_id must be {EXPECTED_EXPERIMENT_ID!r}",
        passes,
        errors,
    )
    _check(
        data.get("experiment_status") == EXPECTED_EXPERIMENT_STATUS
        and data.get("is_formal_experiment") is False,
        "experiment is frozen as nonformal and not run",
        "experiment must have status design_frozen_not_run and is_formal_experiment=false",
        passes,
        errors,
    )
    _check(
        data.get("created_date") == EXPECTED_CREATED_DATE
        and bool(re.fullmatch(r"[0-9a-f]{40}", str(data.get("source_commit", "")))),
        "creation date and source commit are frozen",
        "created_date must be 2026-07-18 and source_commit must be a lowercase 40-character Git SHA",
        passes,
        errors,
    )

    design_state = _mapping(data.get("design_state"))
    expected_design_state = {
        "runner_implemented": False,
        "simulation_executed": False,
        "measured_outcomes_present": False,
        "result_rows_present": False,
        "branch_winner_selected": False,
        "positive_recovery_claim_made": False,
    }
    _check(
        design_state == expected_design_state,
        "design state contains no implementation, run, result, winner, or positive claim",
        "design_state must explicitly keep implementation, execution, outcomes, rows, winner, and positive claim false",
        passes,
        errors,
    )
    _check(
        data.get("design_document")
        == "docs/experiments/recovery_action_branching_nonformal_v0.md"
        and data.get("metrics_document") == "docs/theory/recovery_metrics_v0.md",
        "design and metric document references are frozen",
        "design_document or metrics_document differs from the frozen contract",
        passes,
        errors,
    )

    source_evidence = _mapping(data.get("source_evidence"))
    _check(
        source_evidence.get("frozen_final_veto_manifest")
        == "analysis/final_veto_ablation_v0/manifest.json"
        and set(_list(source_evidence.get("frozen_final_veto_evidence_paths")))
        == EXPECTED_FINAL_VETO_EVIDENCE_PATHS
        and source_evidence.get("use") == "read_only_case_selection_and_design_context"
        and source_evidence.get("modification_permitted") is False,
        "frozen Final Veto evidence is referenced read-only",
        "source_evidence must reference the complete frozen Final Veto package as read-only context",
        passes,
        errors,
    )

    source_case = _mapping(data.get("source_case"))
    _check(
        all(source_case.get(key) == value for key, value in EXPECTED_SOURCE_CASE.items()),
        "source case exactly matches the frozen first stress case",
        "source_case differs from the frozen r0=0.98 angle=150 thrust=8000 seed=0 case",
        passes,
        errors,
    )
    _check(
        _mapping(source_case.get("nominal_controller_context"))
        == EXPECTED_SOURCE_CONTEXT,
        "source controller context is frozen",
        "source nominal controller context differs from Phase35 radial_energy_push with Phase34 radius_priority",
        passes,
        errors,
    )

    hazard = _mapping(data.get("hazard"))
    _check(
        hazard.get("hazard_target") == "overspeed"
        and hazard.get("threshold") == 1.90
        and hazard.get("comparator") == ">"
        and hazard.get("prediction_horizon_steps") == 1
        and hazard.get("monitor_id") == "one_step_overspeed_veto_v0"
        and hazard.get("formal_safety_boundary") is False,
        "overspeed hazard remains strict > 1.90 over one step",
        "hazard must remain overspeed with strict > 1.90, one-step prediction, and no formal boundary claim",
        passes,
        errors,
    )

    branch_point = _mapping(data.get("branch_point"))
    _check(
        branch_point.get("selection_ordinal") == "first"
        and branch_point.get("required_monitor_decision") == "veto"
        and branch_point.get("before_nominal_action_execution") is True
        and branch_point.get("before_final_veto_fallback_execution") is True
        and "predicted one-step speed_ratio > 1.90"
        in str(branch_point.get("definition", "")),
        "branch point is the first strict-threshold veto before any action executes",
        "branch point timing, threshold definition, or required veto decision is incomplete",
        passes,
        errors,
    )
    _check(
        set(_list(branch_point.get("required_fields")))
        == REQUIRED_BRANCH_STATE_FIELDS,
        "branch-state fields are complete and exact",
        "branch_point.required_fields differs from the frozen branch-state contract",
        passes,
        errors,
    )
    _check(
        set(_list(branch_point.get("validity_requirements")))
        == REQUIRED_BRANCH_VALIDITY_RULES,
        "branch-state validity requirements are complete",
        "branch point is missing finite-value, monitor-validity, commit, or hash requirements",
        passes,
        errors,
    )

    hash_contract = _mapping(branch_point.get("canonical_hash_contract"))
    _check(
        hash_contract.get("encoding") == "utf-8"
        and hash_contract.get("json_sort_keys") is True
        and hash_contract.get("json_separators") == [",", ":"]
        and hash_contract.get("allow_nan") is False
        and hash_contract.get("hash_algorithm") == "sha256"
        and hash_contract.get("hash_field_excluded_from_input")
        == "canonical_branch_state_hash"
        and hash_contract.get("hash_input_is_complete_branch_state_object_without_hash_field")
        is True
        and hash_contract.get("all_branches_require_byte_equivalent_canonical_input")
        is True
        and hash_contract.get("all_branches_require_identical_canonical_branch_state_hash")
        is True,
        "canonical branch-state hashing contract is complete",
        "canonical branch-state hashing must freeze UTF-8 canonical JSON, SHA-256, excluded hash field, and byte-equivalent branch inputs",
        passes,
        errors,
    )

    coordinates = _mapping(data.get("coordinate_convention"))
    _check(
        coordinates.get("frame_id") == "inertial_cartesian_2d"
        and coordinates.get("state_vector_order") == ["x", "y", "vx", "vy"]
        and coordinates.get("action_vector_order") == ["action_x", "action_y"],
        "Cartesian frame and state/action ordering are frozen",
        "coordinate convention must use inertial_cartesian_2d with exact state and action ordering",
        passes,
        errors,
    )
    _check(
        coordinates.get("radial_unit_vector") == "e_r = r / r_norm"
        and coordinates.get("positive_tangential_unit_vector")
        == "e_t = (-e_r_y, e_r_x)"
        and coordinates.get("positive_tangential_orientation")
        == "counterclockwise_90_degree_rotation_of_e_r"
        and coordinates.get("signed_tangential_speed") == "v_t = dot(v, e_t)"
        and coordinates.get("signed_tangential_error")
        == "tangential_error = v_t - target_circular_speed",
        "radial and positive tangential sign conventions are unambiguous",
        "radial/tangential formulas or positive orientation are ambiguous or changed",
        passes,
        errors,
    )
    sign = _mapping(coordinates.get("sign_convention"))
    invalid_conditions = set(_list(coordinates.get("invalid_evaluation_conditions")))
    required_invalid_conditions = {
        "position_norm_is_zero",
        "position_or_velocity_contains_nonfinite_value",
        "derived_unit_vector_contains_nonfinite_value",
        "target_circular_speed_is_nonfinite",
        "action_contains_nonfinite_value",
    }
    _check(
        sign.get("positive") == 1
        and sign.get("negative") == -1
        and sign.get("zero") == 0
        and sign.get("zero_tolerance") == 1e-12
        and sign.get("zero_tolerance_units") == "m/s"
        and coordinates.get("velocity_zero_tolerance") == 1e-12
        and required_invalid_conditions == invalid_conditions,
        "zero tolerances and invalid-vector handling are frozen",
        "coordinate convention must freeze 1e-12 tolerances and all invalid-vector conditions",
        passes,
        errors,
    )

    branch_ids, branches = _branch_map(data)
    _check(
        len(branch_ids) == 4
        and len(set(branch_ids)) == 4
        and set(branch_ids) == EXPECTED_BRANCH_IDS,
        "exactly four unique frozen branch IDs are declared",
        "branches must contain exactly one declaration for each frozen branch ID",
        passes,
        errors,
    )
    _check(
        len(branches) == 4
        and all(branch.get("implemented") is False for branch in branches.values()),
        "all branch policies remain explicitly unimplemented",
        "every frozen branch must have implemented=false",
        passes,
        errors,
    )

    zero = branches.get("zero_action_reference_v0", {})
    _check(
        zero.get("pre_clip_action") == [0.0, 0.0]
        and zero.get("action_formula") == "u = (0.0, 0.0)"
        and zero.get("record_pre_clip_action") is True
        and zero.get("record_post_clip_action") is True
        and zero.get("final_veto_evaluation")
        == "required_unchanged_before_execution"
        and zero.get("on_veto") == "apply_recovery_action_rejection_contract",
        "zero-action reference branch contract is frozen",
        "zero_action_reference_v0 action, logging, veto evaluation, or rejection behavior changed",
        passes,
        errors,
    )

    velocity = branches.get("velocity_opposed_thrust_v0", {})
    _check(
        velocity.get("action_formula") == "u = -0.25 * v / v_norm"
        and velocity.get("coordinate_frame") == "inertial_cartesian_2d"
        and velocity.get("action_magnitude_before_clipping") == 0.25
        and velocity.get("velocity_zero_tolerance") == 1e-12
        and velocity.get("velocity_zero_rule")
        == "if_v_norm_lte_velocity_zero_tolerance_use_action_0_0"
        and velocity.get("record_pre_clip_action") is True
        and velocity.get("record_post_clip_action") is True
        and velocity.get("final_veto_evaluation")
        == "required_unchanged_before_execution"
        and velocity.get("interpretation_limit")
        == "heuristic_velocity_opposed_thrust_not_proven_braking",
        "velocity-opposed branch magnitude, frame, zero rule, and claim limit are frozen",
        "velocity_opposed_thrust_v0 differs from the frozen 0.25 inertial action contract",
        passes,
        errors,
    )

    tangential = branches.get("tangential_error_correction_v0", {})
    _check(
        tangential.get("tangential_error_formula")
        == "tangential_error = v_t - target_circular_speed"
        and tangential.get("action_formula")
        == "u = -0.25 * sign(tangential_error) * e_t"
        and tangential.get("action_magnitude_before_clipping") == 0.25
        and tangential.get("tangential_error_zero_tolerance") == 1e-12
        and tangential.get("zero_error_rule")
        == "if_abs_tangential_error_lte_tolerance_use_action_0_0"
        and tangential.get("cartesian_mapping")
        == "u_x = -0.25 * sign(tangential_error) * e_t_x; u_y = -0.25 * sign(tangential_error) * e_t_y"
        and tangential.get("record_radial_component") is True
        and tangential.get("record_tangential_component") is True
        and tangential.get("radial_correction_added") is False
        and tangential.get("final_veto_evaluation")
        == "required_unchanged_before_execution",
        "tangential-error branch sign, magnitude, Cartesian mapping, and no-radial rule are frozen",
        "tangential_error_correction_v0 has ambiguous sign, magnitude, mapping, tolerance, or radial behavior",
        passes,
        errors,
    )

    abort = branches.get("explicit_abort_v0", {})
    _check(
        abort.get("policy_type") == "terminal_decision_without_transition"
        and abort.get("execute_further_transition") is False
        and abort.get("recovery_transitions") == 0
        and abort.get("terminal_decision_event_required") is True
        and abort.get("branch_terminal_label") == "explicit_recovery_abort"
        and abort.get("task_recovery") is False
        and abort.get("fallback_action_executed") is False
        and abort.get("recovery_outcome_rule")
        == "hazard_avoided_through_termination_only_if_no_declared_hazard_occurred_before_termination",
        "explicit-abort branch has zero transitions, no fallback, and explicit terminal evidence",
        "explicit_abort_v0 must terminate without transition or fallback and cannot claim task recovery",
        passes,
        errors,
    )

    rejection = _mapping(data.get("recovery_action_rejection"))
    required_rejection_evidence = {
        "recovery_proposed_action_pre_clip",
        "recovery_proposed_action_post_clip",
        "predicted_recovery_next_state",
        "predicted_recovery_speed_ratio",
        "hazard_threshold",
        "hazard_comparator",
        "monitor_decision",
        "monitor_reason",
    }
    _check(
        rejection.get("trigger")
        == "unchanged_final_veto_rejects_proposed_recovery_action"
        and rejection.get("branch_terminal_label") == "recovery_action_rejected"
        and rejection.get("record_recovery_action_rejected") is True
        and rejection.get("execute_rejected_action") is False
        and rejection.get("substitute_zero_action") is False
        and rejection.get("recursive_branch_selection") is False
        and rejection.get("terminate_current_branch") is True
        and set(_list(rejection.get("preserve_evidence_fields")))
        == required_rejection_evidence,
        "rejected recovery actions terminate locally without fallback or recursion and preserve evidence",
        "recovery-action rejection contract is incomplete or permits execution, zero fallback, or recursive selection",
        passes,
        errors,
    )

    horizons = _mapping(data.get("horizons"))
    _check(
        horizons.get("total_episode_horizon_realized_transitions") == 100000
        and horizons.get("recovery_horizon_realized_transitions") == 10000
        and horizons.get("recovery_transition_counter_initial_value") == 0
        and horizons.get("first_realized_branch_transition_number") == 1
        and horizons.get("rejected_action_recovery_transition_count") == 0
        and horizons.get("explicit_abort_recovery_transition_count") == 0
        and horizons.get("recovery_transition_increment_rule")
        == "increment_only_after_a_branch_selected_transition_is_realized",
        "total and recovery horizons and realized-transition counting are frozen",
        "horizons must remain 100000 total and 10000 recovery realized transitions with zero-count rejection and abort",
        passes,
        errors,
    )

    recovery_success = _mapping(data.get("recovery_success"))
    _check(
        recovery_success.get("definition_id") == "recovery_success_v0"
        and recovery_success.get("logical_operator") == "all"
        and set(_list(recovery_success.get("required_conditions")))
        == REQUIRED_RECOVERY_SUCCESS_CONDITIONS
        and recovery_success.get("crossing_must_occur_at_or_after_branch_point")
        is True
        and recovery_success.get("hazard_avoidance_alone_is_recovery_success")
        is False
        and set(_list(recovery_success.get("report_separately")))
        == {
            "simulator_defined_success",
            "target_radius_crossing",
            "phase34_compatible_recoverable_crossing",
            "recovery_success",
            "retreat_or_termination",
            "hazard_outcome",
        },
        "Recovery Success v0 requires hazard avoidance, validity, crossing, and recoverability within horizon",
        "recovery_success contract is incomplete or treats no overspeed alone as recovery",
        passes,
        errors,
    )

    stop = _mapping(data.get("stop_conditions"))
    conditions = {
        item.get("condition_id"): (
            item.get("branch_terminal_label"),
            item.get("controlled_terminal_label_mapping"),
        )
        for item in _list(stop.get("conditions"))
        if isinstance(item, dict)
    }
    _check(
        stop.get("priority_order") == EXPECTED_STOP_PRIORITY
        and set(conditions) == set(EXPECTED_STOP_LABELS),
        "all stop conditions and simultaneous-condition priority are frozen",
        "stop condition set or priority order differs from the frozen contract",
        passes,
        errors,
    )
    _check(
        conditions == EXPECTED_STOP_LABELS,
        "branch terminal labels and controlled-taxonomy mappings are complete",
        "one or more stop conditions has an incorrect branch label or controlled taxonomy mapping",
        passes,
        errors,
    )

    metrics = _mapping(data.get("required_metrics"))
    margin_rules = _mapping(metrics.get("margin_rules"))
    _check(
        set(_list(metrics.get("margin"))) == REQUIRED_MARGIN_METRICS
        and set(_list(metrics.get("cost"))) == REQUIRED_COST_METRICS
        and set(_list(metrics.get("intervention"))) == REQUIRED_INTERVENTION_METRICS
        and set(_list(metrics.get("outcomes"))) == REQUIRED_OUTCOME_METRICS
        and margin_rules.get("unknown_multi_step_quantities") == "null_not_guessed"
        and metrics.get("missing_value_rule")
        == "unsupported_or_inapplicable_values_are_null_with_a_reason_and_are_never_guessed",
        "required margin, cost, intervention, outcome, and missing-value metrics are frozen",
        "required_metrics is missing fields or permits guessed unknown quantities",
        passes,
        errors,
    )

    comparison = _mapping(data.get("comparison_contract"))
    _check(
        comparison.get("comparison_type")
        == "four_branch_common_state_nonformal_diagnostic"
        and comparison.get("branch_count") == 4
        and set(_list(comparison.get("requirements")))
        == REQUIRED_COMPARISON_RULES
        and comparison.get("not_monitor_off_vs_monitor_on_pairing") is True
        and comparison.get("no_branch_expected_to_win_in_advance") is True,
        "four-branch common-state comparison contract is complete",
        "comparison contract must require identical shared state/configuration and no advance winner",
        passes,
        errors,
    )

    output = _mapping(data.get("output_contract"))
    future_artifacts = [
        item for item in _list(output.get("future_artifacts")) if isinstance(item, dict)
    ]
    output_paths = {str(item.get("path", "")) for item in future_artifacts}
    _check(
        output.get("base_directory")
        == "analysis/recovery_action_branching_nonformal_v0/"
        and output.get("manifest_path")
        == "analysis/recovery_action_branching_nonformal_v0/manifest.json"
        and output_paths == EXPECTED_OUTPUT_PATHS,
        "manifest and five future artifact paths are exact",
        "output contract must contain the exact isolated manifest, branch state, CSV, JSONL, summary, and plot paths",
        passes,
        errors,
    )
    _check(
        output.get("manifest_exists_at_design_freeze") is True
        and len(future_artifacts) == 5
        and all(item.get("must_not_exist_at_design_freeze") is True for item in future_artifacts)
        and output.get("publication_rule")
        == "only_manifest_json_exists_at_design_freeze",
        "only the manifest is permitted to exist at design freeze",
        "future artifacts must all be marked absent and only manifest.json may exist at design freeze",
        passes,
        errors,
    )
    all_output_paths = [str(output.get("manifest_path", "")), *sorted(output_paths)]
    _check(
        all(_is_within_output_directory(path) for path in all_output_paths),
        "all artifact paths remain inside the isolated nonformal directory",
        "one or more output paths escapes analysis/recovery_action_branching_nonformal_v0/",
        passes,
        errors,
    )

    protected_paths = set(_list(data.get("protected_paths")))
    _check(
        REQUIRED_PROTECTED_PATHS <= protected_paths,
        "frozen Final Veto and Phase34-37 paths are protected",
        "protected_paths is missing Final Veto or a protected Phase34-37 location",
        passes,
        errors,
    )
    _check(
        not any(
            _overlaps_protected_path(path, protected_paths)
            for path in all_output_paths
        ),
        "no recovery-branching artifact overlaps a protected path",
        "one or more recovery-branching output paths overlaps frozen Final Veto or Phase34-37 evidence",
        passes,
        errors,
    )

    allowed_claims = set(_list(data.get("allowed_claims")))
    prohibited_claims = set(_list(data.get("prohibited_claims")))
    _check(
        allowed_claims == EXPECTED_ALLOWED_CLAIMS
        and REQUIRED_PROHIBITED_CLAIMS == prohibited_claims,
        "allowed one-case diagnostics and all prohibited claims are explicit",
        "claim lists must contain exactly the scoped diagnostics and required prohibitions",
        passes,
        errors,
    )

    forbidden_locations = _forbidden_key_locations(data)
    _check(
        not forbidden_locations,
        "manifest contains no measured results, branch winner, ranking, or score fields",
        f"manifest contains forbidden result or winner keys at: {forbidden_locations}",
        passes,
        errors,
    )

    if errors:
        raise ManifestValidationError(errors)
    return passes


def validate_future_outputs_absent(
    data: dict[str, Any], repository_root: Path
) -> str:
    existing = [
        path
        for path in _future_output_paths(data)
        if (repository_root / PurePosixPath(path)).exists()
    ]
    if existing:
        raise ManifestValidationError(
            [f"future design-freeze artifacts already exist: {existing}"]
        )
    return "all five future artifacts remain uncreated"


def validate_post_extraction_pre_experiment_outputs(
    data: dict[str, Any], repository_root: Path
) -> list[str]:
    declared_paths = set(_future_output_paths(data))
    if declared_paths != EXPECTED_OUTPUT_PATHS:
        raise ManifestValidationError(
            ["declared future output paths do not match the frozen contract"]
        )
    branch_state_path = repository_root / PurePosixPath(BRANCH_STATE_OUTPUT_PATH)
    if not branch_state_path.is_file():
        raise ManifestValidationError(
            ["authorized frozen branch_state.json is missing"]
        )
    existing_result_paths = [
        path
        for path in sorted(EXPERIMENT_RESULT_OUTPUT_PATHS)
        if (repository_root / PurePosixPath(path)).exists()
    ]
    if existing_result_paths:
        raise ManifestValidationError(
            [
                "recovery experiment result artifacts exist before authorization: "
                f"{existing_result_paths}"
            ]
        )
    try:
        try:
            from scripts.check_recovery_branch_state import validate_branch_state
        except ModuleNotFoundError:
            from check_recovery_branch_state import validate_branch_state

        validate_branch_state(branch_state_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestValidationError(
            [f"authorized branch-state artifact is invalid: {exc}"]
        ) from exc
    return [
        "authorized branch_state.json exists and validates",
        "recovery result, decision-log, summary, and plot artifacts remain uncreated",
    ]


def validate_manifest(
    path: Path,
    *,
    repository_root: Path | None = None,
    require_future_outputs_absent: bool = False,
    require_branch_state_ready: bool = False,
) -> list[str]:
    data = load_manifest(path)
    passes = validate_manifest_data(data)
    if require_future_outputs_absent:
        root = repository_root or find_repository_root(path)
        passes.append(validate_future_outputs_absent(data, root))
    if require_branch_state_ready:
        root = repository_root or find_repository_root(path)
        passes.extend(validate_post_extraction_pre_experiment_outputs(data, root))
    return passes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen nonformal Recovery Action Branching v0 manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest path; defaults to "
            "analysis/recovery_action_branching_nonformal_v0/manifest.json "
            "under the repository root."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = find_repository_root()
        manifest_path = (
            args.manifest.resolve()
            if args.manifest
            else repository_root / MANIFEST_RELATIVE_PATH
        )
        passes = validate_manifest(
            manifest_path,
            repository_root=repository_root,
            require_branch_state_ready=True,
        )
    except (FileNotFoundError, ManifestValidationError) as exc:
        errors = exc.errors if isinstance(exc, ManifestValidationError) else [str(exc)]
        for error in errors:
            print(f"FAIL {error}")
        print(
            "Recovery Action Branching manifest validation "
            f"FAILED with {len(errors)} issue(s)."
        )
        return 1

    for message in passes:
        print(f"PASS {message}")
    print(
        "Recovery Action Branching manifest validation "
        f"PASSED with {len(passes)} checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
