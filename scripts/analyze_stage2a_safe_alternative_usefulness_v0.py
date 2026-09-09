from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage2a_post_veto_alternative_audit_v0 as post_veto  # noqa: E402
from scripts.analyze_stage2a_hazard_trigger_relevance_v0 import (  # noqa: E402
    STAGE1B_PATH,
    canonical_sha256,
    directory_aggregate_hash,
)


AUDIT_ID = "stage2a_safe_alternative_usefulness_v0"
SCHEMA_VERSION = "stage2a_safe_alternative_usefulness_audit_v0"
COMPLETED_DATE = "2026-09-09"
THRESHOLD = 1.90
OUTPUT_PATH = Path("analysis/stage2a_safe_alternative_usefulness_v0")
POST_VETO_PATH = Path("analysis/stage2a_post_veto_alternative_audit_v0")
TRIGGER_RELEVANCE_PATH = Path("analysis/stage2a_hazard_trigger_relevance_v0")
POST_VETO_MANIFEST_HASH = "6000f10cf924781cd051808a435804ca272d819e76b76c69a15c02e9d91d278e"
TRIGGER_RELEVANCE_MANIFEST_HASH = (
    "9f446503be008fe4b6a3051d8c98737f822673364cbcebcddee89579364dfa7f"
)
SOURCE_HEAD = "b1c201dae14b186f4d73e4758ce81c3927732ee9"

ALTERNATIVE_IDS = (
    "zero_action_reference_v0",
    "velocity_opposed_thrust_v0",
    "tangential_error_correction_v0",
)
PROGRESS_COMPONENTS = (
    "radius_gap_improvement",
    "radial_component_improvement",
    "absolute_tangential_error_improvement",
    "overspeed_headroom_improvement",
    "diagnostic_energy_proxy_error_improvement",
)
REPORT_FILENAMES = (
    "alternative_comparison.json",
    "progress_metrics.json",
    "recoverability_comparison.json",
    "decision_matrix.json",
    "summary.md",
)
ALL_FILENAMES = ("audit_manifest.json", *REPORT_FILENAMES)


class SafeAlternativeUsefulnessAuditError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SafeAlternativeUsefulnessAuditError(f"expected JSON object: {path.as_posix()}")
    return value


def _canonical_manifest(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("canonical_manifest_hash", None)
    return canonical_sha256(payload)


def source_snapshot(repository_root: Path) -> dict[str, str]:
    return {
        "stage2a_post_veto_alternative_audit": directory_aggregate_hash(
            repository_root / POST_VETO_PATH
        ),
        "stage2a_trigger_relevance_audit": directory_aggregate_hash(
            repository_root / TRIGGER_RELEVANCE_PATH
        ),
        **post_veto.source_snapshot(repository_root),
    }


def validate_sources(repository_root: Path) -> dict[str, str]:
    post_veto.validate_sources(repository_root)
    post_payloads = post_veto.load_published_payloads(repository_root)
    post_manifest = json.loads(post_payloads["audit_manifest.json"])
    if (
        post_manifest.get("canonical_manifest_hash") != POST_VETO_MANIFEST_HASH
        or post_manifest.get("physical_executions") != 0
        or post_manifest.get("Stage_2A_authority_granted") is not False
    ):
        raise SafeAlternativeUsefulnessAuditError("Stage 2A-H source identity mismatch")

    trigger_manifest = _json(repository_root / TRIGGER_RELEVANCE_PATH / "audit_manifest.json")
    if (
        trigger_manifest.get("canonical_manifest_hash") != TRIGGER_RELEVANCE_MANIFEST_HASH
        or _canonical_manifest(trigger_manifest) != TRIGGER_RELEVANCE_MANIFEST_HASH
        or trigger_manifest.get("trigger_a_observation_count") != 0
        or trigger_manifest.get("physical_executions") != 0
        or trigger_manifest.get("Stage_2A_authority_granted") is not False
    ):
        raise SafeAlternativeUsefulnessAuditError("Stage 2A-T source identity mismatch")
    return source_snapshot(repository_root)


def _fields(observation: Mapping[str, object]) -> dict[str, dict[str, object]]:
    encoded = observation.get("fields")
    if not isinstance(encoded, list):
        raise SafeAlternativeUsefulnessAuditError("instrumentation fields are missing")
    result: dict[str, dict[str, object]] = {}
    for item in encoded:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
            raise SafeAlternativeUsefulnessAuditError("invalid instrumentation field encoding")
        result[str(item[0])] = item[1]
    return result


def _valid_number(fields: Mapping[str, Mapping[str, object]], name: str) -> float:
    field = fields.get(name)
    if (
        not isinstance(field, Mapping)
        or field.get("valid") is not True
        or not isinstance(field.get("value"), (int, float))
        or not math.isfinite(float(field["value"]))
    ):
        raise SafeAlternativeUsefulnessAuditError(f"required numeric field unavailable: {name}")
    return float(field["value"])


def _valid_bool(fields: Mapping[str, Mapping[str, object]], name: str) -> bool:
    field = fields.get(name)
    if not isinstance(field, Mapping) or field.get("valid") is not True:
        raise SafeAlternativeUsefulnessAuditError(f"required boolean field unavailable: {name}")
    value = field.get("value")
    if not isinstance(value, bool):
        raise SafeAlternativeUsefulnessAuditError(f"required boolean field malformed: {name}")
    return value


def _state_values(fields: Mapping[str, Mapping[str, object]]) -> list[float]:
    return [
        _valid_number(fields, name)
        for name in ("position_x", "position_y", "velocity_x", "velocity_y")
    ]


def _direction(value: float) -> str:
    if value > 0.0:
        return "improved"
    if value < 0.0:
        return "worsened"
    return "unchanged"


def _progress_record(event: Mapping[str, object]) -> dict[str, object]:
    pre = _fields(event["pre_observation"])
    post = _fields(event["post_observation"])
    predicted = _fields(event["predicted_observation"])
    progress = _fields({"fields": event["progress_sample"]})

    predicted_ratio = _valid_number(predicted, "predicted_speed_ratio")
    pre_radial_ratio = _valid_number(pre, "radial_velocity_ratio")
    post_radial_ratio = _valid_number(post, "radial_velocity_ratio")
    pre_tangential = _valid_number(pre, "tangential_velocity_error")
    post_tangential = _valid_number(post, "tangential_velocity_error")
    pre_energy_error = _valid_number(pre, "specific_energy_error")
    post_energy_error = _valid_number(post, "specific_energy_error")
    radius_improvement = -_valid_number(progress, "delta_absolute_target_radius_error")
    radial_improvement = abs(pre_radial_ratio) - abs(post_radial_ratio)
    tangential_improvement = abs(pre_tangential) - abs(post_tangential)
    headroom_improvement = _valid_number(progress, "delta_overspeed_headroom")
    energy_improvement = abs(pre_energy_error) - abs(post_energy_error)
    signed_radius_error = _valid_number(pre, "signed_target_radius_error")
    radial_velocity = _valid_number(pre, "radial_velocity")

    component_values = {
        "radius_gap_improvement": radius_improvement,
        "radial_component_improvement": radial_improvement,
        "absolute_tangential_error_improvement": tangential_improvement,
        "overspeed_headroom_improvement": headroom_improvement,
        "diagnostic_energy_proxy_error_improvement": energy_improvement,
    }
    return {
        "source_event_index": event["event_index"],
        "source_event_hash": event["canonical_event_sha256"],
        "source_pre_state_hash": event["pre_state_hash"],
        "action": event["executed_action"],
        "predicted_speed_ratio": predicted_ratio,
        "Final_Veto_status": event["monitor_decision"],
        "safe_under_frozen_threshold": (
            predicted_ratio <= THRESHOLD and event["monitor_decision"] == "allow"
        ),
        "radius_progress": {
            "delta_absolute_target_radius_error": _valid_number(
                progress, "delta_absolute_target_radius_error"
            ),
            "improvement_magnitude": radius_improvement,
            "status": _direction(radius_improvement),
        },
        "radial_velocity_component": {
            "pre_radial_velocity": radial_velocity,
            "post_radial_velocity": _valid_number(post, "radial_velocity"),
            "pre_radial_velocity_ratio": pre_radial_ratio,
            "post_radial_velocity_ratio": post_radial_ratio,
            "radial_direction_toward_target": signed_radius_error * radial_velocity < 0.0,
            "absolute_ratio_improvement": radial_improvement,
            "status": _direction(radial_improvement),
        },
        "tangential_velocity_error": {
            "pre_signed_error": pre_tangential,
            "post_signed_error": post_tangential,
            "absolute_error_improvement": tangential_improvement,
            "status": _direction(tangential_improvement),
        },
        "overspeed_headroom": {
            "delta": headroom_improvement,
            "status": _direction(headroom_improvement),
        },
        "recoverability_components": {
            "pre": {
                "radius_component_pass": _valid_bool(pre, "radius_component_pass"),
                "radial_velocity_component_pass": _valid_bool(
                    pre, "radial_velocity_component_pass"
                ),
                "tangential_velocity_component_pass": _valid_bool(
                    pre, "tangential_velocity_component_pass"
                ),
                "phase34_compatible_recoverability": _valid_bool(
                    pre, "phase34_compatible_recoverability"
                ),
            },
            "post": {
                "radius_component_pass": _valid_bool(post, "radius_component_pass"),
                "radial_velocity_component_pass": _valid_bool(
                    post, "radial_velocity_component_pass"
                ),
                "tangential_velocity_component_pass": _valid_bool(
                    post, "tangential_velocity_component_pass"
                ),
                "phase34_compatible_recoverability": _valid_bool(
                    post, "phase34_compatible_recoverability"
                ),
            },
        },
        "energy_diagnostic_proxy": {
            "evidence_level": "diagnostic_proxy",
            "pre_specific_energy_error": pre_energy_error,
            "post_specific_energy_error": post_energy_error,
            "absolute_error_improvement": energy_improvement,
            "status": _direction(energy_improvement),
            "not_a_conserved_invariant_claim": True,
        },
        "crossing_evidence": {
            "target_radius_crossing": _valid_bool(post, "target_radius_crossing"),
            "crossing_recovery_eligible": _valid_bool(post, "crossing_recovery_eligible"),
            "future_crossing_inferred": False,
        },
        "component_progress": {
            name: {"improvement": value, "status": _direction(value)}
            for name, value in component_values.items()
        },
        "combined_progress_score": None,
        "combined_progress_score_status": "not_evaluated",
        "future_recovery_success": None,
        "future_recovery_success_status": "not_evaluated",
        "available_fields": sorted(
            {
                "predicted_speed_ratio",
                "Final_Veto_status",
                "radius_progress",
                "radial_velocity_component",
                "tangential_velocity_error",
                "recoverability_components",
                "energy_diagnostic_proxy",
                "crossing_evidence",
            }
        ),
        "unknown_fields": [
            "future_recovery_success",
            "general_action_usefulness",
            "causal_action_effect",
        ],
    }


def _stage1b_transition_records(repository_root: Path) -> list[dict[str, object]]:
    root = repository_root / STAGE1B_PATH
    index = _json(root / "trace_index.json")
    records: list[dict[str, object]] = []
    for trace in index["traces"]:
        if trace.get("explicit_abort") is True:
            continue
        path = root / str(trace["trace_path"])
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            wrapper = json.loads(line)
            event = wrapper["source_event"]
            if event["event_type"] != "transition":
                continue
            pre = _fields(event["pre_observation"])
            records.append(
                {
                    "case_id": trace["case_id"],
                    "action_identity": trace["branch_id"],
                    "state_values": _state_values(pre),
                    "source_artifact": path.relative_to(repository_root).as_posix(),
                    "source_locator": f"jsonl_line:{line_number}",
                    "progress": _progress_record(event),
                }
            )
    return records


def _not_evaluated_usefulness(reason: str) -> dict[str, object]:
    return {
        "usefulness_evidence_status": "not_evaluated",
        "reason": reason,
        "radius_progress": None,
        "radial_velocity_component": None,
        "tangential_velocity_error": None,
        "recoverability_components": None,
        "energy_diagnostic_proxy": None,
        "crossing_evidence": None,
        "component_progress": None,
        "combined_progress_score": None,
        "future_recovery_success": None,
        "available_fields": ["predicted_speed_ratio", "Final_Veto_status"],
        "unknown_fields": [
            "radius_progress",
            "radial_velocity_component",
            "tangential_velocity_error",
            "recoverability_components",
            "energy_diagnostic_proxy",
            "crossing_evidence",
            "future_recovery_success",
        ],
    }


def build_alternative_comparison(repository_root: Path) -> dict[str, object]:
    source = post_veto.build_exact_state_comparisons(repository_root)
    transitions = _stage1b_transition_records(repository_root)
    comparisons: list[dict[str, object]] = []
    full_count = 0
    safety_only_count = 0
    for boundary in source["comparisons"]:
        alternatives: list[dict[str, object]] = []
        for source_alternative in boundary["alternatives"]:
            action_id = str(source_alternative["action_identity"])
            if action_id not in ALTERNATIVE_IDS:
                continue
            matches = [
                record
                for record in transitions
                if record["case_id"] == boundary["case_id"]
                and record["action_identity"] == action_id
                and record["state_values"] == boundary["state_values"]
            ]
            if len(matches) > 1:
                raise SafeAlternativeUsefulnessAuditError("duplicate exact-state usefulness match")
            row = {
                "action_identity": action_id,
                "action": source_alternative["action"],
                "predicted_speed_ratio": source_alternative["predicted_speed_ratio"],
                "Final_Veto_status": source_alternative["allowed_or_rejected_status"],
                "safe_under_frozen_threshold": source_alternative[
                    "safe_under_frozen_threshold"
                ],
                "safety_evidence_source": source_alternative["available_evidence"],
            }
            if matches:
                progress = matches[0]["progress"]
                if (
                    progress["predicted_speed_ratio"] != row["predicted_speed_ratio"]
                    or progress["Final_Veto_status"] != row["Final_Veto_status"]
                ):
                    raise SafeAlternativeUsefulnessAuditError(
                        "Stage 2A-H and Stage 1B alternative evidence differ"
                    )
                row.update(
                    {
                        "usefulness_evidence_status": "available_exact_one_step",
                        "usefulness_evidence_source": matches[0]["source_artifact"],
                        "usefulness_source_locator": matches[0]["source_locator"],
                        **progress,
                    }
                )
                full_count += 1
            else:
                row.update(_not_evaluated_usefulness("no_exact_state_progress_record"))
                safety_only_count += 1
            alternatives.append(row)
        comparisons.append(
            {
                "case_id": boundary["case_id"],
                "branch_step": boundary["branch_step"],
                "state_identity": boundary["state_identity"],
                "nominal_proposal": boundary["nominal_proposal"],
                "alternatives": alternatives,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "exact_veto_state_count": len(comparisons),
        "full_usefulness_alternative_observation_count": full_count,
        "safety_only_alternative_observation_count": safety_only_count,
        "full_three_action_comparison_state_count": sum(
            all(row["usefulness_evidence_status"] == "available_exact_one_step" for row in item["alternatives"])
            for item in comparisons
        ),
        "comparisons": comparisons,
        "unknown_value_policy": "not_evaluated",
        "one_step_evidence_does_not_establish_future_recovery": True,
    }


def _available_rows(comparison: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        row
        for item in comparison["comparisons"]
        for row in item["alternatives"]
        if row["usefulness_evidence_status"] == "available_exact_one_step"
    ]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return None if x_scale == 0.0 or y_scale == 0.0 else numerator / (x_scale * y_scale)


def _average_ranks(values: Iterable[float]) -> list[float]:
    materialized = list(values)
    result = [0.0] * len(materialized)
    ordered = sorted(range(len(materialized)), key=lambda index: materialized[index])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and materialized[ordered[end]] == materialized[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in ordered[cursor:end]:
            result[index] = average
        cursor = end
    return result


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _component_improvement(row: Mapping[str, object], component: str) -> float:
    return float(row["component_progress"][component]["improvement"])


def build_progress_metrics(comparison: Mapping[str, object]) -> dict[str, object]:
    rows = _available_rows(comparison)
    per_action: dict[str, object] = {}
    for action_id in ALTERNATIVE_IDS:
        subset = [row for row in rows if row["action_identity"] == action_id]
        per_action[action_id] = {
            "evaluated_exact_state_count": len(subset),
            "safe_count": sum(row["safe_under_frozen_threshold"] is True for row in subset),
            "component_results": {
                component: {
                    "improved_count": sum(
                        row["component_progress"][component]["status"] == "improved"
                        for row in subset
                    ),
                    "unchanged_count": sum(
                        row["component_progress"][component]["status"] == "unchanged"
                        for row in subset
                    ),
                    "worsened_count": sum(
                        row["component_progress"][component]["status"] == "worsened"
                        for row in subset
                    ),
                }
                for component in PROGRESS_COMPONENTS
            },
            "radial_direction_toward_target_count": sum(
                row["radial_velocity_component"]["radial_direction_toward_target"] is True
                for row in subset
            ),
            "combined_progress_score": None,
        }

    tradeoffs: list[dict[str, object]] = []
    for boundary in comparison["comparisons"]:
        available = {
            row["action_identity"]: row
            for row in boundary["alternatives"]
            if row["usefulness_evidence_status"] == "available_exact_one_step"
        }
        for higher_id in ALTERNATIVE_IDS:
            for lower_id in ALTERNATIVE_IDS:
                if higher_id >= lower_id or higher_id not in available or lower_id not in available:
                    continue
                first = available[higher_id]
                second = available[lower_id]
                if first["predicted_speed_ratio"] == second["predicted_speed_ratio"]:
                    continue
                higher, lower = (
                    (first, second)
                    if first["predicted_speed_ratio"] > second["predicted_speed_ratio"]
                    else (second, first)
                )
                better = [
                    component
                    for component in PROGRESS_COMPONENTS
                    if _component_improvement(higher, component)
                    > _component_improvement(lower, component)
                ]
                worse = [
                    component
                    for component in PROGRESS_COMPONENTS
                    if _component_improvement(higher, component)
                    < _component_improvement(lower, component)
                ]
                if better:
                    tradeoffs.append(
                        {
                            "case_id": boundary["case_id"],
                            "state_identity": boundary["state_identity"],
                            "higher_predicted_ratio_action": higher["action_identity"],
                            "higher_predicted_speed_ratio": higher["predicted_speed_ratio"],
                            "lower_predicted_ratio_action": lower["action_identity"],
                            "lower_predicted_speed_ratio": lower["predicted_speed_ratio"],
                            "components_better_for_higher_ratio_action": better,
                            "components_worse_for_higher_ratio_action": worse,
                            "both_actions_safe": (
                                higher["safe_under_frozen_threshold"] is True
                                and lower["safe_under_frozen_threshold"] is True
                            ),
                        }
                    )

    ratios = [float(row["predicted_speed_ratio"]) for row in rows]
    correlations = {}
    for component in PROGRESS_COMPONENTS:
        improvements = [_component_improvement(row, component) for row in rows]
        correlations[component] = {
            "pearson_predicted_ratio_vs_improvement": _pearson(ratios, improvements),
            "spearman_predicted_ratio_vs_improvement": _spearman(ratios, improvements),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "exact_one_step_observation_count": len(rows),
        "independent_exact_state_count": comparison[
            "full_three_action_comparison_state_count"
        ],
        "per_action": per_action,
        "descriptive_correlations": {
            "sample_count": len(rows),
            "independent_state_count": comparison[
                "full_three_action_comparison_state_count"
            ],
            "metrics": correlations,
            "statistical_inference_authorized": False,
            "general_correlation_claim_authorized": False,
            "limitation": (
                "Nine action observations share only three exact initial states; these "
                "coefficients are descriptive summaries, not independent-sample evidence."
            ),
        },
        "higher_predicted_ratio_with_better_component_progress": tradeoffs,
        "higher_ratio_tradeoff_count": len(tradeoffs),
        "slightly_higher_risk_threshold_defined": False,
        "combined_scalar_progress_score_created": False,
    }


def build_recoverability_comparison(comparison: Mapping[str, object]) -> dict[str, object]:
    rows = _available_rows(comparison)
    per_action: dict[str, object] = {}
    for action_id in ALTERNATIVE_IDS:
        subset = [row for row in rows if row["action_identity"] == action_id]
        per_action[action_id] = {
            "evaluated_exact_state_count": len(subset),
            "post_radius_component_pass_count": sum(
                row["recoverability_components"]["post"]["radius_component_pass"]
                for row in subset
            ),
            "post_radial_velocity_component_pass_count": sum(
                row["recoverability_components"]["post"][
                    "radial_velocity_component_pass"
                ]
                for row in subset
            ),
            "post_tangential_velocity_component_pass_count": sum(
                row["recoverability_components"]["post"][
                    "tangential_velocity_component_pass"
                ]
                for row in subset
            ),
            "post_phase34_compatible_recoverability_count": sum(
                row["recoverability_components"]["post"][
                    "phase34_compatible_recoverability"
                ]
                for row in subset
            ),
            "eligible_crossing_count": sum(
                row["crossing_evidence"]["crossing_recovery_eligible"] for row in subset
            ),
            "future_recovery_success": "not_evaluated",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "predicate_semantics": "exact_existing_Phase34_compatible_components",
        "potentially_recoverable_interpretation": (
            "A true combined predicate would be current-state evidence only, not a "
            "prediction of future recovery success."
        ),
        "per_action": per_action,
        "combined_predicate_true_observation_count": sum(
            row["recoverability_components"]["post"][
                "phase34_compatible_recoverability"
            ]
            for row in rows
        ),
        "eligible_crossing_observation_count": sum(
            row["crossing_evidence"]["crossing_recovery_eligible"] for row in rows
        ),
        "recovery_success_claim_authorized": False,
        "future_success_unknown": True,
    }


def build_decision_matrix(
    comparison: Mapping[str, object],
    progress: Mapping[str, object],
    recoverability: Mapping[str, object],
) -> dict[str, object]:
    velocity_tangential_tradeoffs = [
        row
        for row in progress["higher_predicted_ratio_with_better_component_progress"]
        if {
            row["higher_predicted_ratio_action"],
            row["lower_predicted_ratio_action"],
        }
        == {"velocity_opposed_thrust_v0", "tangential_error_correction_v0"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_dimensions": [
            "safe",
            "component_wise_mission_progress",
            "phase34_compatible_recoverability_evidence",
            "unknown",
        ],
        "Q1_zero_action_safe_or_useful": {
            "answer": "safe_with_mixed_one_step_progress_evidence",
            "evidence": (
                "At three exact veto states zero action was safe and improved radius gap, "
                "absolute tangential error, overspeed headroom, and the diagnostic energy-"
                "proxy error, while its radial-velocity component worsened slightly."
            ),
            "recoverability_result": (
                "No one-step combined Phase34-compatible recoverability or eligible crossing "
                "was observed."
            ),
            "causal_usefulness_claim": "not_authorized",
        },
        "Q2_active_alternatives_vs_zero": {
            "answer": "component_specific_advantages_without_a_global_winner",
            "evidence": (
                "Active alternatives produced larger tangential-error, headroom, and "
                "diagnostic-energy improvements; velocity-opposed thrust also improved the "
                "radial component. Zero action retained slightly greater one-step radius-gap "
                "improvement in the three matched stress states."
            ),
            "controller_superiority_claim": "not_authorized",
        },
        "Q3_lower_speed_ratio_and_recovery": {
            "answer": "not_a_general_recovery_correlation",
            "evidence": (
                "Lower predicted ratio aligns with several component improvements but not "
                "with radius-gap improvement in this nine-observation, three-state sample. "
                "No combined recoverability predicate became true."
            ),
            "descriptive_statistics_only": True,
        },
        "Q4_higher_risk_better_orbital_progress": {
            "answer": len(velocity_tangential_tradeoffs) > 0,
            "qualified_evidence_count": len(velocity_tangential_tradeoffs),
            "evidence": (
                "At each of the three fully matched states, tangential correction had a "
                "higher but still safe predicted ratio than velocity-opposed thrust and "
                "greater radius-gap and tangential-error improvement."
            ),
            "risk_wording_limitation": (
                "Higher predicted speed ratio is only one frozen hazard metric; no general "
                "risk score or 'slightly higher' threshold is defined."
            ),
        },
        "Q5_future_replacement_objective": {
            "answer": "safety_gate_then_explicit_usefulness_evidence",
            "safety_role": "hard proposal-level eligibility gate under Final Veto",
            "usefulness_role": (
                "A future separately authorized selector should compare declared orbital "
                "progress and recoverability components after safety is satisfied."
            ),
            "policy_frozen": False,
            "weights_or_combined_score_defined": False,
            "active_authority_granted": False,
        },
        "evidence_scope": {
            "exact_veto_states": comparison["exact_veto_state_count"],
            "full_three_action_states": comparison[
                "full_three_action_comparison_state_count"
            ],
            "full_action_observations": comparison[
                "full_usefulness_alternative_observation_count"
            ],
            "combined_recoverability_true": recoverability[
                "combined_predicate_true_observation_count"
            ],
        },
        "action_selection_authorized": False,
        "Stage_2A_authority_granted": False,
    }


def build_payloads(repository_root: Path) -> dict[str, bytes]:
    snapshot = validate_sources(repository_root)
    comparison = build_alternative_comparison(repository_root)
    progress = build_progress_metrics(comparison)
    recoverability = build_recoverability_comparison(comparison)
    decision = build_decision_matrix(comparison, progress, recoverability)
    summary = f"""# Stage 2A Safe Alternative Usefulness Audit v0

Completed: {COMPLETED_DATE}

## Status

Frozen offline evidence audit completed. Physical executions: 0. Controller executions: 0.
Stage 2A authority remains unauthorized.

## Evidence Scope

Stage 2A-H provides four exact nominal-veto states. Three join exactly to measured
Stage 1B one-step records for zero action, velocity-opposed thrust, and tangential-error
correction, yielding nine full usefulness observations. The angle-155 state has only
zero-action safety evidence; its progress and recoverability fields remain
`not_evaluated`.

## Findings

Zero action was safe at all four exact veto states. At the three states with measured
one-step progress it improved radius gap, absolute tangential error, overspeed headroom,
and diagnostic energy-proxy error, but slightly worsened the radial-velocity component.
It produced no one-step combined Phase34-compatible recoverability and no eligible
crossing. These observations do not establish causal recovery usefulness.

Both active alternatives were safe at the three fully matched states. They produced
larger tangential-error, headroom, and diagnostic-energy improvements than zero action.
Velocity-opposed thrust also improved the radial component, while zero action retained
slightly greater one-step radius-gap improvement. No scalar utility score or universal
best action is supported.

Lower predicted speed ratio did not uniformly correspond to better orbital progress.
In all three full comparisons, tangential correction had a higher but still safe ratio
than velocity-opposed thrust while producing greater radius-gap and tangential-error
improvement. The descriptive correlations use nine action observations from only three
states and do not support general statistical inference.

## Decision Boundary

A future replacement design should retain Final Veto safety as a hard proposal gate and
then evaluate explicitly declared usefulness components. This audit defines no selector,
weights, combined score, threshold, controller change, or active authority.

## Claim Restrictions

This result does not demonstrate recovery success, controller superiority, an optimal
action, a universal replacement policy, formal safety, or deployment readiness. Future
recovery success remains `not_evaluated`.
"""
    documents: dict[str, object] = {
        "alternative_comparison.json": comparison,
        "progress_metrics.json": progress,
        "recoverability_comparison.json": recoverability,
        "decision_matrix.json": decision,
    }
    payloads = {
        name: json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
        for name, value in documents.items()
    }
    payloads["summary.md"] = summary.encode("ascii")
    artifact_hashes = {
        name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES
    }
    manifest = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "audit_classification": "frozen_offline_evidence_audit",
        "source_repository_head": SOURCE_HEAD,
        "source_post_veto_manifest_hash": POST_VETO_MANIFEST_HASH,
        "source_trigger_relevance_manifest_hash": TRIGGER_RELEVANCE_MANIFEST_HASH,
        "source_snapshot": snapshot,
        "overspeed_threshold": THRESHOLD,
        "overspeed_comparator": ">",
        "exact_veto_state_count": comparison["exact_veto_state_count"],
        "full_three_action_comparison_state_count": comparison[
            "full_three_action_comparison_state_count"
        ],
        "full_usefulness_alternative_observation_count": comparison[
            "full_usefulness_alternative_observation_count"
        ],
        "physical_executions": 0,
        "controller_executions": 0,
        "new_trajectories": 0,
        "D1_D2_rerun": False,
        "Final_Veto_modified": False,
        "threshold_changed": False,
        "synthetic_states_created": False,
        "actions_tuned": False,
        "Stage_2A_authority_granted": False,
        "staged_recovery_execution": "not_authorized",
        "artifact_filenames": list(ALL_FILENAMES),
        "artifact_hashes": artifact_hashes,
        "audit_bundle_hash": canonical_sha256(artifact_hashes),
        "claim_restrictions": [
            "no_recovery_success_claim",
            "no_controller_superiority_claim",
            "no_optimal_action_claim",
            "no_universal_policy_claim",
            "no_formal_safety_claim",
            "no_active_authority_claim",
        ],
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    payloads["audit_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    return payloads


def validate_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if set(payloads) != set(ALL_FILENAMES):
        raise SafeAlternativeUsefulnessAuditError("audit artifact set mismatch")
    manifest = json.loads(payloads["audit_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise SafeAlternativeUsefulnessAuditError("audit manifest canonical hash mismatch")
    hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES}
    if manifest["artifact_hashes"] != hashes or manifest["audit_bundle_hash"] != canonical_sha256(hashes):
        raise SafeAlternativeUsefulnessAuditError("audit artifact hash mismatch")
    comparison = json.loads(payloads["alternative_comparison.json"])
    progress = json.loads(payloads["progress_metrics.json"])
    recoverability = json.loads(payloads["recoverability_comparison.json"])
    decision = json.loads(payloads["decision_matrix.json"])
    if (
        comparison["exact_veto_state_count"] != 4
        or comparison["full_three_action_comparison_state_count"] != 3
        or comparison["full_usefulness_alternative_observation_count"] != 9
        or progress["exact_one_step_observation_count"] != 9
        or progress["combined_scalar_progress_score_created"] is not False
        or recoverability["recovery_success_claim_authorized"] is not False
        or decision["action_selection_authorized"] is not False
        or manifest["physical_executions"] != 0
        or manifest["controller_executions"] != 0
        or manifest["Stage_2A_authority_granted"] is not False
    ):
        raise SafeAlternativeUsefulnessAuditError("frozen usefulness result mismatch")
    return {**manifest, "canonical_manifest_hash": supplied}


def publish_payloads(repository_root: Path, payloads: Mapping[str, bytes]) -> Path:
    target = repository_root / OUTPUT_PATH
    if target.exists():
        raise SafeAlternativeUsefulnessAuditError("audit output already exists")
    validate_payloads(payloads)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage2a-usefulness-audit-", dir=target.parent))
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
        raise SafeAlternativeUsefulnessAuditError("published audit directory is missing")
    names = {path.name for path in target.iterdir() if path.is_file()}
    if names != set(ALL_FILENAMES):
        raise SafeAlternativeUsefulnessAuditError("published artifact set mismatch")
    payloads = {name: (target / name).read_bytes() for name in ALL_FILENAMES}
    validate_payloads(payloads)
    validate_sources(repository_root)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Stage 2A safe-alternative usefulness audit."
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
                "STAGE2A_SAFE_ALTERNATIVE_USEFULNESS_STATIC: passed; "
                "physical_executions=0; controller_executions=0; write_performed=false"
            )
            return 0
        payloads = build_payloads(ROOT)
        target = publish_payloads(ROOT, payloads)
        manifest = validate_payloads(payloads)
        print(
            "STAGE2A_SAFE_ALTERNATIVE_USEFULNESS_AUDIT: published; "
            f"path={target.relative_to(ROOT).as_posix()}; "
            f"manifest_hash={manifest['canonical_manifest_hash']}; physical_executions=0"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
