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

from scripts.analyze_stage2a_hazard_trigger_relevance_v0 import (  # noqa: E402
    D2_PATH,
    FINAL_VETO_PATH,
    REGISTRY_PATH,
    STAGE1B_PATH,
    canonical_sha256,
    collect_recovery_proposals,
    directory_aggregate_hash,
    file_sha256,
    validate_sources as validate_trigger_sources,
)


AUDIT_ID = "stage2a_post_veto_alternative_proposal_audit_v0"
SCHEMA_VERSION = "stage2a_post_veto_alternative_proposal_audit_v0"
COMPLETED_DATE = "2026-09-04"
THRESHOLD = 1.90
OUTPUT_PATH = Path("analysis/stage2a_post_veto_alternative_audit_v0")
TRIGGER_AUDIT_PATH = Path("analysis/stage2a_hazard_trigger_relevance_v0")
TRIGGER_AUDIT_MANIFEST_HASH = (
    "9f446503be008fe4b6a3051d8c98737f822673364cbcebcddee89579364dfa7f"
)
SOURCE_HEAD = "be458ac936ac3b8911484b2edc5ac1e3daafd057"

REPORT_FILENAMES = (
    "veto_event_inventory.json",
    "alternative_coverage.json",
    "exact_state_comparisons.json",
    "final_veto_interpretation.json",
    "evidence_matrix.json",
    "summary.md",
)
ALL_FILENAMES = ("audit_manifest.json", *REPORT_FILENAMES)


class PostVetoAuditError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostVetoAuditError(f"expected JSON object: {path.as_posix()}")
    return value


def _canonical_manifest(document: Mapping[str, object], *, exclude: tuple[str, ...] = ()) -> str:
    payload = dict(document)
    payload.pop("canonical_manifest_hash", None)
    for field in exclude:
        payload.pop(field, None)
    return canonical_sha256(payload)


def source_snapshot(repository_root: Path) -> dict[str, str]:
    paths = {
        "final_veto": FINAL_VETO_PATH,
        "stage1b_calibration": STAGE1B_PATH,
        "stage2a_d2": D2_PATH,
        "branch_state_registry": REGISTRY_PATH,
        "stage2a_trigger_relevance_audit": TRIGGER_AUDIT_PATH,
    }
    return {
        key: directory_aggregate_hash(repository_root / value)
        for key, value in sorted(paths.items())
    }


def validate_sources(repository_root: Path) -> dict[str, str]:
    validate_trigger_sources(repository_root)
    trigger_manifest = _json(repository_root / TRIGGER_AUDIT_PATH / "audit_manifest.json")
    if (
        trigger_manifest.get("canonical_manifest_hash") != TRIGGER_AUDIT_MANIFEST_HASH
        or _canonical_manifest(trigger_manifest) != TRIGGER_AUDIT_MANIFEST_HASH
        or trigger_manifest.get("physical_executions") != 0
        or trigger_manifest.get("Stage_2A_authority_granted") is not False
    ):
        raise PostVetoAuditError("Stage 2A-T source audit identity mismatch")

    final_rows = _load_final_veto_rows(repository_root)
    veto_segments = _veto_segments(final_rows)
    if len(veto_segments) != 5 or sum(int(row["step_count"]) for row in veto_segments) != 499877:
        raise PostVetoAuditError("Final Veto compact logical count mismatch")
    for row in veto_segments:
        if (
            row.get("fallback_executed") is not True
            or row.get("fallback_failure") is not False
            or int(row.get("fallback_failure_count", -1)) != 0
            or float(row["maximum_predicted_fallback_speed_ratio"]) > THRESHOLD
        ):
            raise PostVetoAuditError("Final Veto safe fallback segment contract mismatch")
    return source_snapshot(repository_root)


def _load_final_veto_rows(repository_root: Path) -> list[dict[str, object]]:
    path = repository_root / FINAL_VETO_PATH / "decision_log.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _veto_segments(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("event_kind") == "decision_segment"
        and row.get("decision_type") == "veto_action"
        and row.get("veto_status") == "veto"
        and float(row.get("minimum_predicted_nominal_speed_ratio", float("-inf")))
        > THRESHOLD
    ]


def _state_from_artifact(document: Mapping[str, object]) -> list[float]:
    if all(name in document for name in ("position_x", "position_y", "velocity_x", "velocity_y")):
        return [
            float(document[name])
            for name in ("position_x", "position_y", "velocity_x", "velocity_y")
        ]
    position = document.get("position")
    velocity = document.get("velocity")
    if not (
        isinstance(position, list)
        and len(position) == 2
        and isinstance(velocity, list)
        and len(velocity) == 2
    ):
        raise PostVetoAuditError("branch-state Cartesian state is incomplete")
    return [float(position[0]), float(position[1]), float(velocity[0]), float(velocity[1])]


def _registry_veto_boundaries(repository_root: Path) -> list[dict[str, object]]:
    index = _json(repository_root / REGISTRY_PATH / "branch_state_index.json")
    result: list[dict[str, object]] = []
    for member in index["members"]:
        artifact = repository_root / str(member["artifact_path"])
        document = _json(artifact)
        monitor = document.get("monitor_decision")
        if not isinstance(monitor, dict) or monitor.get("decision") != "veto":
            continue
        proposed = document.get("proposed_action", document.get("nominal_proposed_action"))
        predicted = document.get("predicted_speed_ratio")
        result.append(
            {
                "case_id": member["case_id"],
                "branch_step": member["branch_step"],
                "registry_member_id": member["registry_member_id"],
                "registry_state_hash": member["canonical_branch_state_hash"],
                "state_values": _state_from_artifact(document),
                "nominal_action": proposed,
                "nominal_predicted_speed_ratio": predicted,
                "veto_reason": monitor.get("reason"),
                "source_artifact": member["artifact_path"],
            }
        )
    return result


def _d2_veto_boundaries(repository_root: Path) -> list[dict[str, object]]:
    source_cases = _json(repository_root / D2_PATH / "source_case_index.json")["source_cases"]
    trajectories = _json(repository_root / D2_PATH / "source_boundary_index.json")[
        "recovery_trajectories"
    ]
    trajectory_map = {str(item["case_id"]): item for item in trajectories}
    result: list[dict[str, object]] = []
    for source in source_cases:
        if source.get("source_boundary_status") != "available":
            continue
        first = trajectory_map[str(source["case_id"])]["records"][0]
        state = first["current_state"]
        values = [
            float(state[name])
            for name in ("position_x", "position_y", "velocity_x", "velocity_y")
        ]
        result.append(
            {
                "case_id": source["case_id"],
                "branch_step": source["branch_step"],
                "registry_member_id": (
                    "legacy_canonical" if source.get("anchor") is True else None
                ),
                "registry_state_hash": source["boundary_state_hash"],
                "state_values": values,
                "nominal_action": source["nominal_controller_action"],
                "nominal_predicted_speed_ratio": source[
                    "nominal_controller_predicted_speed_ratio"
                ],
                "veto_reason": "predicted_nominal_overspeed",
                "source_artifact": (D2_PATH / "source_case_index.json").as_posix(),
                "zero_action": {
                    "action": first["zero_action"],
                    "predicted_speed_ratio": first["predicted_speed_ratio"],
                    "decision": first["final_veto_decision"],
                },
            }
        )
    return result


def _exact_boundaries(repository_root: Path) -> list[dict[str, object]]:
    entries: dict[tuple[str, int], dict[str, object]] = {}
    for item in _registry_veto_boundaries(repository_root):
        entries[(str(item["case_id"]), int(item["branch_step"]))] = item
    for item in _d2_veto_boundaries(repository_root):
        key = (str(item["case_id"]), int(item["branch_step"]))
        if key in entries:
            if entries[key]["state_values"] != item["state_values"]:
                raise PostVetoAuditError("D2 and registry boundary states differ")
            entries[key]["zero_action"] = item["zero_action"]
            entries[key]["D2_source_artifact"] = item["source_artifact"]
        else:
            entries[key] = item
    return [entries[key] for key in sorted(entries)]


def _stage1b_records(repository_root: Path) -> list[dict[str, object]]:
    return [
        record
        for record in collect_recovery_proposals(repository_root)
        if record["source_class"] == "stage1b_measured_recovery_trace"
    ]


def _explicit_abort_evidence(repository_root: Path) -> dict[str, object]:
    index = _json(repository_root / STAGE1B_PATH / "trace_index.json")
    match = [item for item in index["traces"] if item["branch_id"] == "explicit_abort_v0"]
    if len(match) != 1:
        raise PostVetoAuditError("explicit-abort trace is not unique")
    path = repository_root / STAGE1B_PATH / str(match[0]["trace_path"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [row["source_event"] for row in rows]
    if (
        len(events) != 2
        or events[0]["event_type"] != "initial_snapshot"
        or events[1]["event_type"] != "terminal"
        or events[1]["terminal_reason"] != "explicit_abort"
        or events[1]["transition_executed"] is not False
        or events[1]["proposed_action"] is not None
        or events[1]["executed_action"] is not None
    ):
        raise PostVetoAuditError("explicit-abort frozen terminal semantics mismatch")
    fields = dict(events[0]["pre_observation"]["fields"])
    values = [
        float(fields[name]["value"])
        for name in ("position_x", "position_y", "velocity_x", "velocity_y")
    ]
    return {
        "case_id": match[0]["case_id"],
        "state_values": values,
        "source_artifact": (STAGE1B_PATH / str(match[0]["trace_path"])).as_posix(),
        "event_count": 2,
        "physical_transition_count": 0,
        "terminal_reason": match[0]["terminal_reason"],
    }


def _safe_prediction(value: object, decision: object) -> bool:
    return isinstance(value, (int, float)) and float(value) <= THRESHOLD and decision == "allow"


def build_veto_event_inventory(repository_root: Path) -> dict[str, object]:
    segments = _veto_segments(_load_final_veto_rows(repository_root))
    d2 = _d2_veto_boundaries(repository_root)
    overlap = [
        item
        for item in d2
        if any(
            item["case_id"] == row["case_id"]
            and int(item["branch_step"]) == int(row["start_step"])
            and item["nominal_action"] == row["first_nominal_action"]
            and float(item["nominal_predicted_speed_ratio"])
            == float(row["first_predicted_nominal_speed_ratio"])
            for row in segments
        )
    ]
    compact_count = sum(int(row["step_count"]) for row in segments)
    case_ids = sorted(
        {str(row["case_id"]) for row in segments} | {str(row["case_id"]) for row in d2}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "logical_event_identity_contract": {
            "compact_event": "final_veto:<case_id>:step:<start_step..end_step>",
            "D2_boundary_event": "D2:<case_id>:branch_step:<branch_step>",
            "compact_ranges_are_lossless_for_event_count": True,
            "per_event_Cartesian_state_identity": "not_evaluated",
        },
        "compact_veto_segment_count": len(segments),
        "compact_logical_veto_event_count": compact_count,
        "D2_first_veto_event_count": len(d2),
        "cross_artifact_reproduction_count": len(overlap),
        "duplicate_aware_veto_event_count": compact_count + len(d2) - len(overlap),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "segments": [
            {
                "case_id": row["case_id"],
                "logical_event_identity_range": (
                    f"final_veto:{row['case_id']}:step:{row['start_step']}..{row['end_step']}"
                ),
                "start_step": row["start_step"],
                "end_step": row["end_step"],
                "logical_event_count": row["step_count"],
                "state_identity": None,
                "state_identity_status": "not_evaluated_compact_log",
                "first_nominal_action": row["first_nominal_action"],
                "last_nominal_action": row["last_nominal_action"],
                "nominal_action_per_step": "not_evaluated_compact_log",
                "minimum_nominal_predicted_speed_ratio": row[
                    "minimum_predicted_nominal_speed_ratio"
                ],
                "maximum_nominal_predicted_speed_ratio": row[
                    "maximum_predicted_nominal_speed_ratio"
                ],
                "veto_reason": row["decision_reason"],
                "veto_status": row["veto_status"],
                "zero_action_fallback": {
                    "action_identity": "zero_action_reference_v0",
                    "first_executed_action": row["first_executed_action"],
                    "last_executed_action": row["last_executed_action"],
                    "minimum_predicted_speed_ratio": row[
                        "minimum_predicted_fallback_speed_ratio"
                    ],
                    "maximum_predicted_speed_ratio": row[
                        "maximum_predicted_fallback_speed_ratio"
                    ],
                    "status": "executed_as_declared_fallback",
                    "fallback_failure": row["fallback_failure"],
                },
            }
            for row in segments
        ],
        "D2_first_veto_events": [
            {
                "case_id": item["case_id"],
                "branch_step": item["branch_step"],
                "state_identity": item["registry_state_hash"],
                "state_identity_status": "available_exact",
                "nominal_action": item["nominal_action"],
                "nominal_predicted_speed_ratio": item["nominal_predicted_speed_ratio"],
                "veto_reason": item["veto_reason"],
                "cross_artifact_reproduction": item in overlap,
            }
            for item in d2
        ],
    }


def build_exact_state_comparisons(repository_root: Path) -> dict[str, object]:
    alternatives = _stage1b_records(repository_root)
    abort = _explicit_abort_evidence(repository_root)
    comparisons: list[dict[str, object]] = []
    for boundary in _exact_boundaries(repository_root):
        matches = [
            record
            for record in alternatives
            if record["case_id"] == boundary["case_id"]
            and record["state_values"] == boundary["state_values"]
        ]
        by_action = {str(record["action_identity"]): record for record in matches}
        if "zero_action" in boundary and "zero_action_reference_v0" not in by_action:
            zero = boundary["zero_action"]
            by_action["zero_action_reference_v0"] = {
                "action_identity": "zero_action_reference_v0",
                "action": zero["action"],
                "predicted_speed_ratio": zero["predicted_speed_ratio"],
                "final_veto_decision": zero["decision"],
                "source_class": "D2_zero_action_recovery",
            }
        rows = []
        for action_id in (
            "zero_action_reference_v0",
            "velocity_opposed_thrust_v0",
            "tangential_error_correction_v0",
        ):
            record = by_action.get(action_id)
            if record is None:
                rows.append(
                    {
                        "action_identity": action_id,
                        "action": None,
                        "predicted_speed_ratio": None,
                        "allowed_or_rejected_status": "not_evaluated",
                        "available_evidence": "not_evaluated_at_exact_state",
                        "safe_under_frozen_threshold": None,
                    }
                )
            else:
                predicted = record["predicted_speed_ratio"]
                decision = record["final_veto_decision"]
                rows.append(
                    {
                        "action_identity": action_id,
                        "action": record["action"],
                        "predicted_speed_ratio": predicted,
                        "allowed_or_rejected_status": decision,
                        "available_evidence": record["source_class"],
                        "safe_under_frozen_threshold": _safe_prediction(predicted, decision),
                    }
                )
        abort_here = abort["case_id"] == boundary["case_id"] and abort["state_values"] == boundary[
            "state_values"
        ]
        rows.append(
            {
                "action_identity": "explicit_abort_v0",
                "action": None,
                "predicted_speed_ratio": None,
                "allowed_or_rejected_status": "not_evaluated_terminal_semantics",
                "available_evidence": (
                    "observed_terminal_only_zero_transition_trace"
                    if abort_here
                    else "not_evaluated_at_exact_state"
                ),
                "safe_under_frozen_threshold": None,
                "physical_action_alternative": False,
            }
        )
        comparisons.append(
            {
                "case_id": boundary["case_id"],
                "branch_step": boundary["branch_step"],
                "state_identity": boundary["registry_state_hash"],
                "state_identity_status": "available_exact",
                "state_values": boundary["state_values"],
                "nominal_proposal": {
                    "action": boundary["nominal_action"],
                    "predicted_speed_ratio": boundary["nominal_predicted_speed_ratio"],
                    "Final_Veto_decision": "veto",
                    "veto_reason": boundary["veto_reason"],
                },
                "alternatives": rows,
                "at_least_one_safe_physical_alternative": any(
                    row["safe_under_frozen_threshold"] is True for row in rows
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "exact_state_comparison_count": len(comparisons),
        "comparisons": comparisons,
        "explicit_abort_semantics": (
            "explicit_abort_v0 is terminal semantics with no physical proposal; its "
            "predicted speed ratio and allow/reject status remain not_evaluated"
        ),
    }


def build_alternative_coverage(
    inventory: Mapping[str, object], comparisons: Mapping[str, object]
) -> dict[str, object]:
    total = int(inventory["duplicate_aware_veto_event_count"])
    exact_rows = comparisons["comparisons"]
    exact_by_action: dict[str, list[Mapping[str, object]]] = {
        action: []
        for action in (
            "zero_action_reference_v0",
            "velocity_opposed_thrust_v0",
            "tangential_error_correction_v0",
            "explicit_abort_v0",
        )
    }
    for comparison in exact_rows:
        for row in comparison["alternatives"]:
            exact_by_action[str(row["action_identity"])].append(row)
    segment_count = int(inventory["compact_logical_veto_event_count"])
    d2_unique = int(inventory["D2_first_veto_event_count"]) - int(
        inventory["cross_artifact_reproduction_count"]
    )
    zero_safe = segment_count + d2_unique
    result: dict[str, dict[str, object]] = {}
    for action_id, rows in exact_by_action.items():
        evaluated = [row for row in rows if isinstance(row["predicted_speed_ratio"], (int, float))]
        safe = [row for row in evaluated if row["safe_under_frozen_threshold"] is True]
        if action_id == "zero_action_reference_v0":
            result[action_id] = {
                "logical_events_with_available_prediction": zero_safe,
                "logical_events_safe": zero_safe,
                "logical_events_rejected": 0,
                "logical_events_not_evaluated": 0,
                "maximum_evaluated_predicted_speed_ratio": max(
                    segment["zero_action_fallback"]["maximum_predicted_speed_ratio"]
                    for segment in inventory["segments"]
                ),
                "consistently_at_or_below_1p90_in_available_evidence": True,
                "evidence_scope": "compact_fallback_segments_plus_one_unique_D2_boundary",
            }
        elif action_id == "explicit_abort_v0":
            observed = sum(
                row["available_evidence"] == "observed_terminal_only_zero_transition_trace"
                for row in rows
            )
            result[action_id] = {
                "logical_events_with_available_prediction": 0,
                "logical_events_safe": 0,
                "logical_events_rejected": 0,
                "logical_events_not_evaluated": total,
                "observed_terminal_semantics_count": observed,
                "maximum_evaluated_predicted_speed_ratio": None,
                "consistently_at_or_below_1p90_in_available_evidence": None,
                "physical_action_alternative": False,
                "evidence_scope": "terminal_semantics_not_action_prediction",
            }
        else:
            result[action_id] = {
                "logical_events_with_available_prediction": len(evaluated),
                "logical_events_safe": len(safe),
                "logical_events_rejected": len(evaluated) - len(safe),
                "logical_events_not_evaluated": total - len(evaluated),
                "maximum_evaluated_predicted_speed_ratio": (
                    max(float(row["predicted_speed_ratio"]) for row in evaluated)
                    if evaluated
                    else None
                ),
                "consistently_at_or_below_1p90_in_available_evidence": (
                    len(safe) == len(evaluated) if evaluated else None
                ),
                "evidence_scope": "exact_registry_boundary_states_in_Stage1B_traces",
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "duplicate_aware_veto_event_count": total,
        "veto_events_with_at_least_one_safe_alternative": zero_safe,
        "veto_events_without_safe_alternative_evidence": total - zero_safe,
        "most_frequently_safe_alternative": "zero_action_reference_v0",
        "alternative_results": result,
        "evaluated_physical_alternatives_consistently_at_or_below_1p90": True,
        "general_consistency_claim_authorized": False,
        "limitation": (
            "Zero-action coverage is complete through compact fallback extrema. "
            "Velocity-opposed and tangential alternatives are evaluated at only three "
            "exact veto states; missing predictions remain not_evaluated."
        ),
    }


def build_payloads(repository_root: Path) -> dict[str, bytes]:
    snapshot = validate_sources(repository_root)
    inventory = build_veto_event_inventory(repository_root)
    comparisons = build_exact_state_comparisons(repository_root)
    coverage = build_alternative_coverage(inventory, comparisons)
    interpretation = {
        "schema_version": SCHEMA_VERSION,
        "forced_choice": "action_replacement_opportunity",
        "precise_interpretation": (
            "proposal_level_safety_barrier_with_observed_zero_action_replacement"
        ),
        "terminal_safety_barrier": False,
        "why_not_terminal": (
            "The five frozen Final Veto segments executed zero-action fallback and "
            "continued; veto itself was not a terminal event."
        ),
        "action_replacement_observed": True,
        "replacement_evidence": {
            "logical_veto_events": inventory["compact_logical_veto_event_count"],
            "fallback_action_identity": "zero_action_reference_v0",
            "fallback_failures": 0,
            "maximum_fallback_predicted_speed_ratio": coverage["alternative_results"][
                "zero_action_reference_v0"
            ]["maximum_evaluated_predicted_speed_ratio"],
        },
        "authority_consequence": "none",
        "Stage_2A_authority_granted": False,
        "Final_Veto_role_if_future_replacement_is_considered": (
            "Final Veto remains the proposal-level barrier for any replacement action."
        ),
        "non_claims": [
            "no_controller_superiority",
            "no_recovery_improvement",
            "no_formal_safety",
            "no_general_alternative_safety",
            "no_active_replacement_authority",
        ],
    }
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_rows": [
            {
                "source": (FINAL_VETO_PATH / "decision_log.jsonl").as_posix(),
                "supports": "499877_compact_nominal_veto_and_zero_fallback_decisions",
                "state_identity": "not_evaluated_per_logical_event",
                "alternative_prediction": "segment_minimum_and_maximum_available",
            },
            {
                "source": (D2_PATH / "source_case_index.json").as_posix(),
                "supports": "two_exact_first_veto_boundaries",
                "state_identity": "available_exact",
                "alternative_prediction": "zero_action_available",
            },
            {
                "source": (REGISTRY_PATH / "branch_state_index.json").as_posix(),
                "supports": "three_exact_registered_veto_boundaries",
                "state_identity": "available_exact",
                "alternative_prediction": "joined_to_Stage1B_by_exact_Cartesian_state",
            },
            {
                "source": (STAGE1B_PATH / "traces").as_posix(),
                "supports": "zero_velocity_opposed_tangential_predictions_at_three_veto_states",
                "state_identity": "available_exact_Cartesian",
                "alternative_prediction": "one_step_predicted",
            },
            {
                "source": (
                    STAGE1B_PATH / "traces/legacy_canonical__explicit_abort_v0.jsonl"
                ).as_posix(),
                "supports": "terminal_only_explicit_abort_at_one_exact_state",
                "state_identity": "available_exact_Cartesian",
                "alternative_prediction": "not_evaluated_terminal_semantics",
            },
        ],
        "unknown_value_policy": "not_evaluated",
        "new_physics_inference": False,
    }
    summary = f"""# Stage 2A Post-Veto Alternative Proposal Audit v0

Completed: {COMPLETED_DATE}

## Status

Frozen offline evidence audit completed. Physical executions: 0. Stage 2A authority remains unauthorized.

## Veto Universe

The frozen Final Veto compact log represents 499,877 logical nominal-proposal
rejections. D2 records two exact first-veto boundaries, one reproducing a compact-log
event and one additional angle-155 event. The duplicate-aware universe therefore
contains **{inventory['duplicate_aware_veto_event_count']} veto events across six cases**.

Compact event identity is recoverable as `(case_id, step)`, but per-event Cartesian state
identity is not published and remains `not_evaluated`. Four exact boundary states are
available from D2 and the branch-state registry.

## Safe Alternatives

All **{coverage['veto_events_with_at_least_one_safe_alternative']}** duplicate-aware veto
events have at least one safe alternative in frozen evidence: `zero_action_reference_v0`.
The five Final Veto segments executed that fallback with zero recorded fallback failures,
and their maximum predicted fallback ratio was
`{coverage['alternative_results']['zero_action_reference_v0']['maximum_evaluated_predicted_speed_ratio']}`.

`velocity_opposed_thrust_v0` and `tangential_error_correction_v0` were each evaluated and
allowed at three exact veto boundary states. Every available prediction for those actions
was at or below `1.90`, but the other {coverage['alternative_results']['velocity_opposed_thrust_v0']['logical_events_not_evaluated']}
veto events remain `not_evaluated` for each action. This sparse coverage does not support
a general safety claim.

`explicit_abort_v0` is not a physical alternative proposal. Frozen evidence contains one
terminal-only, zero-transition observation at the canonical boundary; predicted speed and
allow/reject status remain `not_evaluated`.

## Interpretation

Of the two supplied choices, the frozen behavior is better interpreted as an **action
replacement opportunity**, with an important qualification: Final Veto is the
proposal-level safety barrier and the observed replacement is specifically zero action.
It is not a terminal barrier because the vetoed runs continued under fallback.

No evidence here authorizes active Stage 2A replacement, establishes controller
superiority, proves recovery improvement, or proves formal safety.
"""
    documents: dict[str, object] = {
        "veto_event_inventory.json": inventory,
        "alternative_coverage.json": coverage,
        "exact_state_comparisons.json": comparisons,
        "final_veto_interpretation.json": interpretation,
        "evidence_matrix.json": evidence,
    }
    payloads = {
        name: json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
        for name, value in documents.items()
    }
    payloads["summary.md"] = summary.encode("ascii")
    artifact_hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES}
    manifest = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "audit_classification": "frozen_offline_evidence_audit",
        "source_repository_head": SOURCE_HEAD,
        "source_trigger_relevance_manifest_hash": TRIGGER_AUDIT_MANIFEST_HASH,
        "source_snapshot": snapshot,
        "overspeed_threshold": THRESHOLD,
        "overspeed_comparator": ">",
        "duplicate_aware_veto_event_count": inventory["duplicate_aware_veto_event_count"],
        "veto_events_with_safe_alternative": coverage[
            "veto_events_with_at_least_one_safe_alternative"
        ],
        "most_frequently_safe_alternative": coverage["most_frequently_safe_alternative"],
        "Final_Veto_interpretation": interpretation["precise_interpretation"],
        "physical_executions": 0,
        "controller_executions": 0,
        "new_trajectories": 0,
        "D1_D2_rerun": False,
        "Final_Veto_modified": False,
        "Stage_2A_authority_granted": False,
        "staged_recovery_execution": "not_authorized",
        "artifact_filenames": list(ALL_FILENAMES),
        "artifact_hashes": artifact_hashes,
        "audit_bundle_hash": canonical_sha256(artifact_hashes),
        "claim_restrictions": interpretation["non_claims"],
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    payloads["audit_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii") + b"\n"
    )
    return payloads


def validate_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if set(payloads) != set(ALL_FILENAMES):
        raise PostVetoAuditError("audit artifact set mismatch")
    manifest = json.loads(payloads["audit_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise PostVetoAuditError("audit manifest canonical hash mismatch")
    hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES}
    if manifest["artifact_hashes"] != hashes or manifest["audit_bundle_hash"] != canonical_sha256(hashes):
        raise PostVetoAuditError("audit artifact hash mismatch")
    inventory = json.loads(payloads["veto_event_inventory.json"])
    coverage = json.loads(payloads["alternative_coverage.json"])
    comparison = json.loads(payloads["exact_state_comparisons.json"])
    interpretation = json.loads(payloads["final_veto_interpretation.json"])
    if (
        inventory["duplicate_aware_veto_event_count"] != 499878
        or coverage["veto_events_with_at_least_one_safe_alternative"] != 499878
        or coverage["most_frequently_safe_alternative"] != "zero_action_reference_v0"
        or comparison["exact_state_comparison_count"] != 4
        or interpretation["forced_choice"] != "action_replacement_opportunity"
        or manifest["physical_executions"] != 0
        or manifest["Stage_2A_authority_granted"] is not False
    ):
        raise PostVetoAuditError("frozen post-veto audit result mismatch")
    return {**manifest, "canonical_manifest_hash": supplied}


def publish_payloads(repository_root: Path, payloads: Mapping[str, bytes]) -> Path:
    target = repository_root / OUTPUT_PATH
    if target.exists():
        raise PostVetoAuditError("audit output already exists")
    validate_payloads(payloads)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage2a-post-veto-audit-", dir=target.parent))
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
        raise PostVetoAuditError("published audit directory is missing")
    names = {path.name for path in target.iterdir() if path.is_file()}
    if names != set(ALL_FILENAMES):
        raise PostVetoAuditError("published artifact set mismatch")
    payloads = {name: (target / name).read_bytes() for name in ALL_FILENAMES}
    validate_payloads(payloads)
    validate_sources(repository_root)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Stage 2A post-veto alternative audit.")
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
            print("execution_enabled=false")
            return 0
        validate_sources(ROOT)
        if args.validate_only:
            print("STAGE2A_POST_VETO_ALTERNATIVE_STATIC: passed; physical_executions=0; write_performed=false")
            return 0
        payloads = build_payloads(ROOT)
        target = publish_payloads(ROOT, payloads)
        manifest = validate_payloads(payloads)
        print(
            "STAGE2A_POST_VETO_ALTERNATIVE_AUDIT: published; "
            f"path={target.relative_to(ROOT).as_posix()}; "
            f"manifest_hash={manifest['canonical_manifest_hash']}; physical_executions=0"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
