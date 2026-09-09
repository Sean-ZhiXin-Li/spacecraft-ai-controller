from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage2a_safe_alternative_usefulness_v0 as usefulness  # noqa: E402
from scripts.analyze_stage2a_hazard_trigger_relevance_v0 import (  # noqa: E402
    canonical_sha256,
    directory_aggregate_hash,
)


AUDIT_ID = "stage2a_recovery_objective_definition_v0"
SCHEMA_VERSION = "stage2a_recovery_objective_definition_audit_v0"
COMPLETED_DATE = "2026-09-09"
OUTPUT_PATH = Path("analysis/stage2a_recovery_objective_definition_v0")
USEFULNESS_PATH = Path("analysis/stage2a_safe_alternative_usefulness_v0")
POST_VETO_PATH = Path("analysis/stage2a_post_veto_alternative_audit_v0")
GUARD_EVIDENCE_PATH = Path("analysis/staged_recovery_guard_evidence_v0")
INSTRUMENTATION_PATH = Path("analysis/staged_recovery_instrumentation_v0")
USEFULNESS_MANIFEST_HASH = "4efcd5874110b5244c9f33bd2d88c08d11207b8dc97ed11959ab1e2ae79fa104"
POST_VETO_MANIFEST_HASH = "6000f10cf924781cd051808a435804ca272d819e76b76c69a15c02e9d91d278e"
GUARD_EVIDENCE_MANIFEST_HASH = "df96854e1caecd45560f8d7e78136bc751bbf3087eab78a35ee6a0051c5ba648"
INSTRUMENTATION_MANIFEST_HASH = "c4947e623e7f9a83de16163f58c5a0da7a3f7b10ee3b10ce88f4eae4805f122c"
SOURCE_HEAD = "78a201841fedd1e829d8784296e6c0958a64cf6a"
JSON_REPORTS = (
    "objective_inventory.json",
    "metric_definition.json",
    "conflict_analysis.json",
    "unknown_evidence.json",
)
ALL_FILENAMES = (*JSON_REPORTS, "summary.md")


class RecoveryObjectiveDefinitionAuditError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryObjectiveDefinitionAuditError(
            f"expected JSON object: {path.as_posix()}"
        )
    return value


def _canonical_document(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("canonical_payload_hash", None)
    return canonical_sha256(payload)


def _with_hash(document: dict[str, object]) -> dict[str, object]:
    result = dict(document)
    result["canonical_payload_hash"] = canonical_sha256(result)
    return result


def source_snapshot(repository_root: Path) -> dict[str, str]:
    return {
        "stage2a_safe_alternative_usefulness": directory_aggregate_hash(
            repository_root / USEFULNESS_PATH
        ),
        "stage2a_post_veto_alternative": directory_aggregate_hash(
            repository_root / POST_VETO_PATH
        ),
        "staged_recovery_guard_evidence": directory_aggregate_hash(
            repository_root / GUARD_EVIDENCE_PATH
        ),
        "staged_recovery_instrumentation": directory_aggregate_hash(
            repository_root / INSTRUMENTATION_PATH
        ),
    }


def validate_sources(repository_root: Path) -> dict[str, object]:
    usefulness_payloads = usefulness.load_published_payloads(repository_root)
    usefulness_manifest = json.loads(usefulness_payloads["audit_manifest.json"])
    if (
        usefulness_manifest.get("canonical_manifest_hash") != USEFULNESS_MANIFEST_HASH
        or usefulness_manifest.get("physical_executions") != 0
        or usefulness_manifest.get("controller_executions") != 0
        or usefulness_manifest.get("Stage_2A_authority_granted") is not False
    ):
        raise RecoveryObjectiveDefinitionAuditError("Stage 2A-I source identity mismatch")

    post_veto_manifest = _json(repository_root / POST_VETO_PATH / "audit_manifest.json")
    if post_veto_manifest.get("canonical_manifest_hash") != POST_VETO_MANIFEST_HASH:
        raise RecoveryObjectiveDefinitionAuditError("Stage 2A-H source identity mismatch")

    guard_manifest = _json(repository_root / GUARD_EVIDENCE_PATH / "analysis_manifest.json")
    if (
        guard_manifest.get("canonical_payload_hash") != GUARD_EVIDENCE_MANIFEST_HASH
        or guard_manifest.get("new_runtime_execution") is not False
        or guard_manifest.get("phase_guard_policy") != "not_frozen"
    ):
        raise RecoveryObjectiveDefinitionAuditError("Stage 1A source identity mismatch")
    thresholds = guard_manifest.get("exact_inherited_thresholds")
    if thresholds != {
        "overspeed": {"comparator": ">", "threshold": 1.9},
        "phase34_radial_velocity_ratio": {
            "comparator": "inclusive_absolute_<=",
            "threshold": 0.02,
        },
        "phase34_radius_error_ratio": {
            "comparator": "inclusive_absolute_<=",
            "threshold": 0.0025,
        },
        "phase34_tangential_velocity_error_ratio": {
            "comparator": "inclusive_absolute_<=",
            "threshold": 0.25,
        },
    }:
        raise RecoveryObjectiveDefinitionAuditError("inherited threshold contract mismatch")

    instrumentation_manifest = _json(
        repository_root / INSTRUMENTATION_PATH / "instrumentation_manifest.json"
    )
    if instrumentation_manifest.get("canonical_payload_hash") != INSTRUMENTATION_MANIFEST_HASH:
        raise RecoveryObjectiveDefinitionAuditError("Stage 0A source identity mismatch")

    progress = json.loads(usefulness_payloads["progress_metrics.json"])
    recoverability = json.loads(usefulness_payloads["recoverability_comparison.json"])
    if progress.get("exact_one_step_observation_count") != 9:
        raise RecoveryObjectiveDefinitionAuditError("Stage 2A-I progress identity mismatch")
    if progress.get("combined_scalar_progress_score_created") is not False:
        raise RecoveryObjectiveDefinitionAuditError("Stage 2A-I created a combined score")
    if recoverability.get("recovery_success_claim_authorized") is not False:
        raise RecoveryObjectiveDefinitionAuditError("Stage 2A-I recovery claim mismatch")
    return {
        "snapshot": source_snapshot(repository_root),
        "usefulness_payloads": usefulness_payloads,
    }


def metric_definitions() -> list[dict[str, object]]:
    return [
        {
            "metric_id": "evidence_validity",
            "source_fields": ["simulation_validity", "recovery_evaluation_validity"],
            "evidence_level": "externally_supplied",
            "objective_role": "safety_precondition",
            "direction": "both_must_be_valid",
            "threshold": None,
            "use_in_future_semantics": "hard_precondition",
            "non_meaning": "Validity does not imply safety or recovery usefulness.",
        },
        {
            "metric_id": "predicted_action_speed_feasibility",
            "source_fields": ["predicted_speed_ratio", "final_veto_decision"],
            "evidence_level": "predicted_and_externally_supplied",
            "objective_role": "safety_constraint",
            "direction": "predicted_speed_ratio_at_or_below_boundary_and_veto_allow",
            "threshold": {"value": 1.9, "comparator": "<=", "source": "Final_Veto_v0"},
            "use_in_future_semantics": "hard_feasibility_gate",
            "non_meaning": "A safe one-step proposal is not necessarily useful or recoverable.",
        },
        {
            "metric_id": "realized_overspeed_state",
            "source_fields": ["realized_speed_ratio", "realized_overspeed"],
            "evidence_level": "derived_from_measured_state",
            "objective_role": "safety_constraint",
            "direction": "avoid_strict_overspeed",
            "threshold": {"value": 1.9, "comparator": ">", "source": "Final_Veto_v0"},
            "use_in_future_semantics": "hazard_state_detection",
            "non_meaning": "Clear realized speed does not establish future proposal safety.",
        },
        {
            "metric_id": "overspeed_headroom",
            "source_fields": ["overspeed_headroom", "predicted_overspeed_headroom"],
            "evidence_level": "derived",
            "objective_role": "safety_margin_progress",
            "direction": "increase_signed_margin_relative_to_1p90",
            "threshold": {"value": 1.9, "comparator": "reference_boundary"},
            "use_in_future_semantics": "secondary_safety_margin_not_a_mission_score",
            "non_meaning": "More headroom alone does not imply better orbital recovery.",
        },
        {
            "metric_id": "absolute_radius_gap",
            "source_fields": ["absolute_target_radius_error", "radius_error_ratio"],
            "evidence_level": "derived",
            "objective_role": "recovery_progress",
            "direction": "decrease_absolute_gap",
            "threshold": {
                "value": 0.0025,
                "comparator": "absolute_ratio_<=",
                "source": "Phase34_recoverability",
            },
            "use_in_future_semantics": "directional_progress_then_exact_component_milestone",
            "non_meaning": "Gap reduction does not guarantee controlled arrival or crossing.",
        },
        {
            "metric_id": "radial_velocity_component",
            "source_fields": [
                "signed_target_radius_error",
                "radial_velocity",
                "radial_velocity_ratio",
            ],
            "evidence_level": "derived",
            "objective_role": "recovery_progress_and_arrival_condition",
            "direction": (
                "move_toward_target_while_reducing_absolute_radial_velocity_ratio_for_arrival"
            ),
            "threshold": {
                "value": 0.02,
                "comparator": "absolute_ratio_<=",
                "source": "Phase34_recoverability",
            },
            "use_in_future_semantics": "two_distinct_atoms_not_one_scalar",
            "non_meaning": "Target-directed motion can still be too fast for recoverability.",
        },
        {
            "metric_id": "absolute_tangential_velocity_error",
            "source_fields": [
                "tangential_velocity_error",
                "tangential_velocity_error_ratio",
            ],
            "evidence_level": "derived",
            "objective_role": "recovery_progress",
            "direction": "decrease_absolute_error_without_hiding_signed_crossing",
            "threshold": {
                "value": 0.25,
                "comparator": "absolute_ratio_<=",
                "source": "Phase34_recoverability",
            },
            "use_in_future_semantics": "directional_progress_then_exact_component_milestone",
            "non_meaning": "A passing tangential component does not imply full recoverability.",
        },
        {
            "metric_id": "phase34_compatible_recoverability_components",
            "source_fields": [
                "radius_component_pass",
                "radial_velocity_component_pass",
                "tangential_velocity_component_pass",
                "phase34_compatible_recoverability",
            ],
            "evidence_level": "derived",
            "objective_role": "recovery_milestone",
            "direction": "all_three_inclusive_components_pass",
            "threshold": "inherited_component_thresholds_only",
            "use_in_future_semantics": "conjunctive_milestone_not_handoff",
            "non_meaning": "The predicate is not recovery success or handoff readiness.",
        },
        {
            "metric_id": "eligible_target_radius_crossing",
            "source_fields": ["target_radius_crossing", "crossing_recovery_eligible"],
            "evidence_level": "derived_from_consecutive_measured_states",
            "objective_role": "recovery_milestone",
            "direction": "observe_exact_existing_crossing_semantics",
            "threshold": None,
            "use_in_future_semantics": "event_evidence_separate_from_component_pass",
            "non_meaning": "No interpolation or future crossing prediction is implied.",
        },
        {
            "metric_id": "specific_energy_error_proxy",
            "source_fields": ["specific_energy_error", "orbital_energy_or_proxy"],
            "evidence_level": "diagnostic_proxy",
            "objective_role": "diagnostic_only",
            "direction": "descriptively_reduce_absolute_proxy_error",
            "threshold": None,
            "use_in_future_semantics": "analysis_context_only",
            "non_meaning": "It is not an exact invariant, safety gate, or recovery objective.",
        },
        {
            "metric_id": "recovery_success_v0",
            "source_fields": ["recovery_success_v0"],
            "evidence_level": "future_or_externally_supplied_evaluator",
            "objective_role": "future_evaluator",
            "direction": "not_evaluated_for_alternative_one_step_comparisons",
            "threshold": None,
            "use_in_future_semantics": "required_for_outcome_claims",
            "non_meaning": "Directional progress cannot be substituted for this evaluator.",
        },
        {
            "metric_id": "handoff_readiness",
            "source_fields": ["handoff_readiness"],
            "evidence_level": "not_evaluated",
            "objective_role": "future_evaluator",
            "direction": "not_defined",
            "threshold": None,
            "use_in_future_semantics": "required_before_nominal_handoff",
            "non_meaning": "Phase34-compatible recoverability is not handoff readiness.",
        },
    ]


def build_metric_definition() -> dict[str, object]:
    metrics = metric_definitions()
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "metric_count": len(metrics),
            "metrics": metrics,
            "combined_score": None,
            "weights": None,
            "optimizer": None,
            "action_selector": None,
            "policy_status": "not_defined",
        }
    )


def build_conflict_analysis(usefulness_payloads: Mapping[str, bytes]) -> dict[str, object]:
    progress = json.loads(usefulness_payloads["progress_metrics.json"])
    recoverability = json.loads(usefulness_payloads["recoverability_comparison.json"])
    q4 = json.loads(usefulness_payloads["decision_matrix.json"])[
        "Q4_higher_risk_better_orbital_progress"
    ]
    zero = progress["per_action"]["zero_action_reference_v0"]["component_results"]
    velocity = progress["per_action"]["velocity_opposed_thrust_v0"]["component_results"]
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "conflicts": [
                {
                    "conflict_id": "radius_closure_vs_radial_arrival_condition",
                    "metrics": ["absolute_radius_gap", "radial_velocity_component"],
                    "mechanism": (
                        "Fast target-directed radial motion can close radius gap while "
                        "remaining outside the radial-velocity recoverability component."
                    ),
                    "stage2a_i_evidence": {
                        "zero_radius_gap_improved_count": zero["radius_gap_improvement"][
                            "improved_count"
                        ],
                        "zero_radial_component_worsened_count": zero[
                            "radial_component_improvement"
                        ]["worsened_count"],
                    },
                    "resolution_status": "unresolved_without_future_semantics",
                },
                {
                    "conflict_id": "speed_headroom_vs_radius_progress",
                    "metrics": ["overspeed_headroom", "absolute_radius_gap"],
                    "mechanism": (
                        "Actions with more speed headroom can produce less one-step radius-gap "
                        "improvement than a higher-ratio safe alternative."
                    ),
                    "stage2a_i_evidence": "observed_at_three_fully_matched_states",
                    "resolution_status": "must_not_be_reduced_to_one_weighted_score_today",
                },
                {
                    "conflict_id": "lower_speed_ratio_vs_tangential_progress",
                    "metrics": [
                        "predicted_action_speed_feasibility",
                        "absolute_tangential_velocity_error",
                    ],
                    "mechanism": (
                        "Tangential correction had a higher but safe ratio than velocity-"
                        "opposed thrust while improving tangential error more."
                    ),
                    "stage2a_i_evidence_count": q4["qualified_evidence_count"],
                    "resolution_status": "component_tradeoff_observed",
                },
                {
                    "conflict_id": "radial_progress_vs_tangential_progress",
                    "metrics": [
                        "radial_velocity_component",
                        "absolute_tangential_velocity_error",
                    ],
                    "mechanism": (
                        "Velocity-opposed thrust improved the radial component at all three "
                        "states; tangential correction produced greater tangential improvement."
                    ),
                    "stage2a_i_evidence": {
                        "velocity_radial_improved_count": velocity[
                            "radial_component_improvement"
                        ]["improved_count"],
                        "fully_matched_state_count": 3,
                    },
                    "resolution_status": "no_global_action_ordering_supported",
                },
                {
                    "conflict_id": "directional_progress_vs_recoverability_milestone",
                    "metrics": [
                        "component_directional_progress",
                        "phase34_compatible_recoverability_components",
                    ],
                    "mechanism": (
                        "Multiple components improved while the combined predicate remained false."
                    ),
                    "stage2a_i_evidence": {
                        "full_observations": progress["exact_one_step_observation_count"],
                        "combined_predicate_true": recoverability[
                            "combined_predicate_true_observation_count"
                        ],
                    },
                    "resolution_status": "progress_must_not_be_called_recovery",
                },
                {
                    "conflict_id": "component_progress_vs_crossing",
                    "metrics": ["component_directional_progress", "eligible_target_radius_crossing"],
                    "mechanism": "One-step component improvement occurred without eligible crossing.",
                    "stage2a_i_evidence": {
                        "eligible_crossing_observations": recoverability[
                            "eligible_crossing_observation_count"
                        ]
                    },
                    "resolution_status": "crossing_remains_separate_event_evidence",
                },
                {
                    "conflict_id": "diagnostic_energy_vs_authoritative_objectives",
                    "metrics": ["specific_energy_error_proxy", "recoverability_milestones"],
                    "mechanism": (
                        "The diagnostic proxy can improve even when no recoverability component "
                        "transition or crossing occurs."
                    ),
                    "stage2a_i_evidence": "proxy_improved_for_all_nine_observations",
                    "resolution_status": "exclude_proxy_from_action_policy",
                },
            ],
            "conflict_count": 7,
            "combined_conflict_score": None,
            "conflict_weights": None,
            "conflict_resolution_policy": None,
        }
    )


def build_unknown_evidence() -> dict[str, object]:
    unknowns = [
        (
            "alternative_specific_multistep_recovery_outcome",
            "future_recovery_evaluator",
            "One-step progress cannot establish eventual recovery.",
        ),
        (
            "handoff_readiness",
            "future_handoff_evaluator",
            "No compatible externally valid handoff-ready evidence exists.",
        ),
        (
            "available_correction_authority",
            "future_authority_evaluator",
            "Stage 0A declares correction authority unsupported.",
        ),
        (
            "future_crossing_or_time_to_crossing",
            "future_trajectory_evaluator",
            "Current crossing evidence is measured pairwise and does not extrapolate.",
        ),
        (
            "active_intervention_release_readiness",
            "future_release_evaluator",
            "No release condition has active authority or validation.",
        ),
        (
            "action_cost_and_resource_tradeoff",
            "future_cost_evaluator",
            "The frozen audit does not define fuel, effort, or wear utility.",
        ),
        (
            "progress_persistence_and_noise_scale",
            "future_multitrace_calibration",
            "Three exact states cannot validate persistence or a noise threshold.",
        ),
        (
            "causal_action_effect",
            "future_matched_counterfactual_experiment",
            "Ballistic state evolution can create progress under zero action.",
        ),
        (
            "general_cross_case_action_ordering",
            "future_multicase_evaluator",
            "Nine observations from three states do not support a universal ordering.",
        ),
    ]
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "unknown_count": len(unknowns),
            "unknowns": [
                {
                    "evidence_id": evidence_id,
                    "status": "not_evaluated",
                    "required_future_evaluator": evaluator,
                    "reason": reason,
                    "may_be_converted_to_false": False,
                    "may_be_used_as_positive_action_evidence": False,
                }
                for evidence_id, evaluator, reason in unknowns
            ],
            "unsupported_evidence_remains_unsupported": True,
        }
    )


def build_objective_inventory(
    snapshot: Mapping[str, str],
    metric_document: Mapping[str, object],
    conflict_document: Mapping[str, object],
    unknown_document: Mapping[str, object],
    report_hashes: Mapping[str, str],
) -> dict[str, object]:
    metrics = metric_document["metrics"]
    roles: dict[str, list[str]] = {}
    for metric in metrics:
        roles.setdefault(str(metric["objective_role"]), []).append(str(metric["metric_id"]))
    return _with_hash(
        {
            "audit_id": AUDIT_ID,
            "schema_version": SCHEMA_VERSION,
            "completed_date": COMPLETED_DATE,
            "audit_classification": "frozen_offline_objective_semantics_audit",
            "source_repository_head": SOURCE_HEAD,
            "source_usefulness_manifest_hash": USEFULNESS_MANIFEST_HASH,
            "source_post_veto_manifest_hash": POST_VETO_MANIFEST_HASH,
            "source_guard_evidence_manifest_hash": GUARD_EVIDENCE_MANIFEST_HASH,
            "source_instrumentation_manifest_hash": INSTRUMENTATION_MANIFEST_HASH,
            "source_snapshot": dict(snapshot),
            "metric_count": metric_document["metric_count"],
            "metrics_by_role": {key: sorted(value) for key, value in sorted(roles.items())},
            "conflict_count": conflict_document["conflict_count"],
            "unknown_evidence_count": unknown_document["unknown_count"],
            "future_action_selection_semantics_recommendation": "lexicographic_safety_first",
            "recommendation_status": "conceptual_not_a_policy",
            "future_semantic_tiers": [
                {
                    "tier": 0,
                    "role": "evidence_validity",
                    "meaning": "Invalid or unavailable required safety evidence blocks eligibility.",
                },
                {
                    "tier": 1,
                    "role": "safety_feasibility",
                    "meaning": "Final Veto and the frozen strict overspeed boundary remain hard constraints.",
                },
                {
                    "tier": 2,
                    "role": "recovery_milestones",
                    "meaning": "Preserve component passes and eligible crossing as separate exact evidence.",
                },
                {
                    "tier": 3,
                    "role": "component_wise_progress",
                    "meaning": "Compare radius, radial, tangential, and headroom without weights.",
                },
                {
                    "tier": "diagnostic",
                    "role": "energy_proxy_context",
                    "meaning": "Never use the diagnostic proxy as an authoritative gate.",
                },
            ],
            "why_not_safety_first_only": "Safe alternatives can have materially different mission-progress evidence.",
            "why_not_weighted_multi_objective": (
                "Weights, exchange rates, and conflict resolution lack frozen scientific support."
            ),
            "unresolved_tie_behavior": "not_evaluated",
            "combined_score": None,
            "weights": None,
            "optimizer": None,
            "policy": None,
            "action_selection": None,
            "physical_executions": 0,
            "controller_executions": 0,
            "Final_Veto_modified": False,
            "thresholds_tuned": False,
            "Stage_2A_authority_granted": False,
            "staged_recovery_execution": "not_authorized",
            "artifact_filenames": list(ALL_FILENAMES),
            "report_hashes": dict(report_hashes),
            "report_bundle_hash": canonical_sha256(report_hashes),
            "claim_restrictions": [
                "no_action_selection",
                "no_recovery_success_claim",
                "no_optimality_claim",
                "no_universal_metric_ordering",
                "no_formal_safety_claim",
                "no_active_authority_claim",
            ],
        }
    )


def build_payloads(repository_root: Path) -> dict[str, bytes]:
    validated = validate_sources(repository_root)
    usefulness_payloads = validated["usefulness_payloads"]
    metric = build_metric_definition()
    conflict = build_conflict_analysis(usefulness_payloads)
    unknown = build_unknown_evidence()
    summary = f"""# Stage 2A Recovery Objective Definition Audit v0

Completed: {COMPLETED_DATE}

## Status

Frozen offline objective-semantics audit completed. Physical executions: 0. Controller
executions: 0. Stage 2A authority remains unauthorized.

## Safety Constraints

Evidence validity is a prerequisite. Proposed-action safety is defined by the existing
Final Veto semantics: predicted speed ratio above `1.90` is overspeed, while a proposal at
or below `1.90` is clear under this one hazard predicate. Realized overspeed remains a
separate measured-state hazard. Signed headroom is a safety margin, not a mission score.

## Recovery Progress

Recovery progress is component-wise: decreasing absolute radius gap, target-directed
radial motion together with decreasing absolute radial-velocity ratio, decreasing
absolute tangential error, and increasing overspeed headroom. The exact Phase34-compatible
component predicate and eligible target-radius crossing are milestones, not continuous
scores. The specific-energy quantity remains a diagnostic proxy only.

## Conflicts

The evidence contains real objective conflicts. Closing radius gap can coexist with a
worsening radial arrival component. Lower predicted speed ratio can trade against radius
or tangential progress. Directional progress can occur without the combined recoverability
predicate or eligible crossing. These conflicts are preserved rather than weighted away.

## Future Evaluators

Future recovery success, handoff readiness, correction authority, crossing prediction,
active-intervention release readiness, action cost, progress persistence, causal action
effect, and general cross-case ordering remain `not_evaluated` or unsupported.

## Recommended Semantics

For a future separately authorized design, the most defensible structure is
**lexicographic with safety first**: validate evidence, enforce Final Veto feasibility,
preserve recoverability and crossing milestones, then compare component-wise progress.
This is a conceptual semantics recommendation, not a policy. It defines no combined
score, weights, optimizer, tie-break, action selector, or active authority.

## Claim Restrictions

This audit does not demonstrate recovery success, select an action, establish an optimal
metric ordering, validate active thresholds, prove formal safety, or authorize Stage 2A.
"""
    provisional = {
        "metric_definition.json": metric,
        "conflict_analysis.json": conflict,
        "unknown_evidence.json": unknown,
    }
    payloads = {
        name: json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
        for name, value in provisional.items()
    }
    payloads["summary.md"] = summary.encode("ascii")
    report_hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in payloads}
    inventory = build_objective_inventory(
        validated["snapshot"], metric, conflict, unknown, report_hashes
    )
    payloads["objective_inventory.json"] = (
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    return payloads


def validate_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if set(payloads) != set(ALL_FILENAMES):
        raise RecoveryObjectiveDefinitionAuditError("audit artifact set mismatch")
    documents = {
        name: json.loads(payloads[name]) for name in JSON_REPORTS
    }
    for name, document in documents.items():
        if document.get("canonical_payload_hash") != _canonical_document(document):
            raise RecoveryObjectiveDefinitionAuditError(f"canonical hash mismatch: {name}")
    inventory = documents["objective_inventory.json"]
    report_hashes = {
        name: hashlib.sha256(payloads[name]).hexdigest()
        for name in ALL_FILENAMES
        if name != "objective_inventory.json"
    }
    if (
        inventory["report_hashes"] != report_hashes
        or inventory["report_bundle_hash"] != canonical_sha256(report_hashes)
    ):
        raise RecoveryObjectiveDefinitionAuditError("report artifact hash mismatch")
    metric = documents["metric_definition.json"]
    conflict = documents["conflict_analysis.json"]
    unknown = documents["unknown_evidence.json"]
    if (
        metric["combined_score"] is not None
        or metric["weights"] is not None
        or metric["optimizer"] is not None
        or metric["action_selector"] is not None
        or conflict["conflict_resolution_policy"] is not None
        or unknown["unknown_count"] != 9
        or any(row["status"] != "not_evaluated" for row in unknown["unknowns"])
        or inventory["future_action_selection_semantics_recommendation"]
        != "lexicographic_safety_first"
        or inventory["recommendation_status"] != "conceptual_not_a_policy"
        or inventory["physical_executions"] != 0
        or inventory["controller_executions"] != 0
        or inventory["Stage_2A_authority_granted"] is not False
    ):
        raise RecoveryObjectiveDefinitionAuditError("objective semantics contract mismatch")
    return inventory


def publish_payloads(repository_root: Path, payloads: Mapping[str, bytes]) -> Path:
    target = repository_root / OUTPUT_PATH
    if target.exists():
        raise RecoveryObjectiveDefinitionAuditError("audit output already exists")
    validate_payloads(payloads)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage2a-objective-audit-", dir=target.parent))
    try:
        for name in ALL_FILENAMES:
            (staging / name).write_bytes(payloads[name])
        staged = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
        validate_payloads(staged)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_published_payloads(repository_root: Path) -> dict[str, bytes]:
    target = repository_root / OUTPUT_PATH
    if not target.is_dir():
        raise RecoveryObjectiveDefinitionAuditError("published audit directory is missing")
    names = {path.name for path in target.iterdir() if path.is_file()}
    if names != set(ALL_FILENAMES):
        raise RecoveryObjectiveDefinitionAuditError("published artifact set mismatch")
    payloads = {name: (target / name).read_bytes() for name in ALL_FILENAMES}
    validate_payloads(payloads)
    validate_sources(repository_root)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Stage 2A recovery-objective semantics audit."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_audit)):
        parser.print_help()
        return 0
    try:
        if args.plan:
            print(f"audit_id={AUDIT_ID}")
            print(f"output_path={OUTPUT_PATH.as_posix()}")
            print("physical_executions=0")
            print("controller_executions=0")
            print("execution_enabled=false")
            return 0
        validate_sources(ROOT)
        if args.validate_only:
            print(
                "STAGE2A_RECOVERY_OBJECTIVE_DEFINITION_STATIC: passed; "
                "physical_executions=0; controller_executions=0; write_performed=false"
            )
            return 0
        payloads = build_payloads(ROOT)
        target = publish_payloads(ROOT, payloads)
        inventory = validate_payloads(payloads)
        print(
            "STAGE2A_RECOVERY_OBJECTIVE_DEFINITION_AUDIT: published; "
            f"path={target.relative_to(ROOT).as_posix()}; "
            f"inventory_hash={inventory['canonical_payload_hash']}; physical_executions=0"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
